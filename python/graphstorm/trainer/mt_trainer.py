"""
    Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License").
    You may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    GraphStorm trainer for multi-task learning.
"""

import time
import resource
import logging
import torch as th
from torch.nn.parallel import DistributedDataParallel
import dgl

from ..config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                      BUILTIN_TASK_NODE_REGRESSION,
                      BUILTIN_TASK_EDGE_CLASSIFICATION,
                      BUILTIN_TASK_EDGE_REGRESSION,
                      BUILTIN_TASK_LINK_PREDICTION)
from ..model import (do_full_graph_inference,
                     do_mini_batch_inference,GSgnnModelBase, GSgnnModel)
from .gsgnn_trainer import GSgnnTrainer
from ..model import (run_node_mini_batch_predict,
                     run_edge_mini_batch_predict,
                     run_lp_mini_batch_predict)

from ..utils import sys_tracker, rt_profiler, print_mem, get_rank
from ..utils import barrier, is_distributed, get_backend

def run_node_predict_mini_batch(model, data, task_info, mini_batch, device):
    g = data.g
    input_nodes, seeds, blocks = mini_batch
    if not isinstance(input_nodes, dict):
        assert len(g.ntypes) == 1
        input_nodes = {g.ntypes[0]: input_nodes}
    nfeat_fields = task_info.dataloader.node_feat_fields
    label_field = task_info.dataloader.label_field
    input_feats = data.get_node_feats(input_nodes, nfeat_fields, device)
    lbl = data.get_node_feats(seeds, label_field, device)
    blocks = [block.to(device) for block in blocks]
    # TODO: we don't support edge features for now.
    loss = model(task_info.task_id, ((blocks, input_feats, None, input_nodes), lbl))

    return loss

def run_edge_predict_mini_batch(model, data, task_info, mini_batch, device):
    input_nodes, batch_graph, blocks = mini_batch
    if not isinstance(input_nodes, dict):
        assert len(batch_graph.ntypes) == 1
        input_nodes = {batch_graph.ntypes[0]: input_nodes}
    nfeat_fields = task_info.dataloader.node_feat_fields
    input_feats = data.get_node_feats(input_nodes, nfeat_fields, device)

    if task_info.dataloader.decoder_edge_feat_fields is not None:
        input_edges = {etype: batch_graph.edges[etype].data[dgl.EID] \
                for etype in batch_graph.canonical_etypes}
        edge_decoder_feats = \
            data.get_edge_feats(input_edges,
                                task_info.dataloader.decoder_edge_feat_fields,
                                device)
        edge_decoder_feats = {etype: feat.to(th.float32) \
            for etype, feat in edge_decoder_feats.items()}
    else:
        edge_decoder_feats = None

    # retrieving seed edge id from the graph to find labels
    assert len(batch_graph.etypes) == 1
    target_etype = batch_graph.canonical_etypes[0]
    # TODO(zhengda) the data loader should return labels directly.
    seeds = batch_graph.edges[target_etype[1]].data[dgl.EID]

    label_field = task_info.dataloader.label_field
    lbl = data.get_edge_feats({target_etype: seeds}, label_field, device)
    blocks = [block.to(device) for block in blocks]
    batch_graph = batch_graph.to(device)
    rt_profiler.record('train_graph2GPU')

    # TODO(zhengda) we don't support edge features for now.
    loss = model(task_info.task_id,
                    ((blocks, input_feats, None, input_nodes),
                    (batch_graph, edge_decoder_feats, lbl)))
    return loss

def run_link_predict_mini_batch(model, data, task_info, mini_batch, device):
    input_nodes, pos_graph, neg_graph, blocks = mini_batch

    if not isinstance(input_nodes, dict):
        assert len(pos_graph.ntypes) == 1
        input_nodes = {pos_graph.ntypes[0]: input_nodes}

    nfeat_fields = task_info.dataloader.node_feat_fields
    input_feats = data.get_node_feats(input_nodes, nfeat_fields, device)

    if task_info.dataloader.pos_graph_feat_fields is not None:
        input_edges = {etype: pos_graph.edges[etype].data[dgl.EID] \
            for etype in pos_graph.canonical_etypes}
        pos_graph_feats = data.get_edge_feats(input_edges,
                                                task_info.dataloader.pos_graph_feat_fields,
                                                device)
    else:
        pos_graph_feats = None

    pos_graph = pos_graph.to(device)
    neg_graph = neg_graph.to(device)
    blocks = [blk.to(device) for blk in blocks]

    # TODO: we don't support edge features for now.
    loss = model(task_info.task_id,
                    ((blocks, input_feats, None, input_nodes),
                    (pos_graph, neg_graph,pos_graph_feats, None)))
    return loss

def multi_task_mini_batch_predict(
    model, emb, loader, device, return_proba=True, return_label=False):
    """ conduct mini batch prediction on multiple tasks

    Parameters
    ----------
    model: GSgnnMultiTaskModelInterface, GSgnnModel
        Multi-task learning model
    emb : dict of Tensor
        The GNN embeddings
    loader: GSgnnMultiTaskDataLoader
        The mini-batch dataloader.
    device: th.device
        Device used to compute test scores.
    return_proba: bool
        Whether to return all the predictions or the maximum prediction.

    Returns
    -------
    list: prediction results of each task
    """
    dataloaders = loader.dataloaders
    task_infos = loader.task_infos
    task_pool = model.task_pool
    res = {}
    with th.no_grad():
        for dataloader, task_info in zip(dataloaders, task_infos):
            if task_info.task_type in \
            [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
                task_type, decoder, _, _ = task_pool[task_info.task_id]
                assert task_info.task_type == task_type
                preds, labels = \
                    run_node_mini_batch_predict(decoder,
                                                emb,
                                                dataloader,
                                                device,
                                                return_proba,
                                                return_label)
                res[task_info.task_id] = (preds, labels)
            elif task_info.task_type in \
            [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
                task_type, decoder, _, _ = task_pool[task_info.task_id]
                assert task_info.task_type == task_type
                preds, labels = \
                    run_edge_mini_batch_predict(decoder,
                                                emb,
                                                loader,
                                                device,
                                                return_proba,
                                                return_label)
                res[task_info.task_id] = (preds, labels)
            elif task_info.task_type == BUILTIN_TASK_LINK_PREDICTION:
                task_type, decoder, _, _ = task_pool[task_info.task_id]
                assert task_info.task_type == task_type
                ranking = run_lp_mini_batch_predict(decoder, emb, dataloader, device)
                res[task_info.task_id] = ranking
            else:
                raise TypeError("Unknown task %s", task_info)

    return res

class GSgnnMultiTaskLearningTrainer(GSgnnTrainer):
    r""" A trainer for multi-task learning

    This class is used to train models for multi task learning.

    It makes use of the functions provided by `GSgnnTrainer`
    to define two main functions: `fit` that performs the training
    for the model that is provided when the object is created,
    and `eval` that evaluates a provided model against test and
    validation data.

    Parameters
    ----------
    model : GSgnnMultiTaskModel
        The GNN model for node prediction.
    topk_model_to_save : int
        The top K model to save.
    """
    def __init__(self, model, topk_model_to_save=1):
        super(GSgnnMultiTaskLearningTrainer, self).__init__(model, topk_model_to_save)
        assert isinstance(model) and isinstance(model, GSgnnModelBase), \
                "The input model is not a GSgnnModel model. Please implement GSgnnModelBase."

    def _run_mini_batch(self, data, model, task_info, mini_batch, device):
        """ run mini batch for a single task

        Parameters
        ----------
        data: GSgnnData
            Graph data
        task_info: TaskInfo
            task meta information
        mini_batch: tuple
            mini-batch info

        Return
        ------
        loss
        """
        if task_info.task_type in \
            [BUILTIN_TASK_NODE_CLASSIFICATION, BUILTIN_TASK_NODE_REGRESSION]:
            return run_node_predict_mini_batch(model,
                                               data,
                                               task_info,
                                               mini_batch,
                                               device)
        elif task_info.task_type in \
            [BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION]:
            return run_edge_predict_mini_batch(model,
                                               data,
                                               task_info,
                                               mini_batch,
                                               device)
        elif task_info.task_type == BUILTIN_TASK_LINK_PREDICTION:
            return run_link_predict_mini_batch(model,
                                               data,
                                               task_info,
                                               mini_batch,
                                               device)
        else:
            raise TypeError("Unknown task %s", task_info)

    def fit(self, train_loader,
            num_epochs,
            val_loader=None,
            test_loader=None,
            use_mini_batch_infer=True,
            save_model_path=None,
            save_model_frequency=-1,
            save_perf_results_path=None,
            freeze_input_layer_epochs=0,
            max_grad_norm=None,
            grad_norm_type=2.0):
        """ The fit function for multi-task learning.

        Performs the training for `self.model`. Iterates over all the tasks
        and run one mini-batch for each task in an iteration. The loss will be
        accumulated. Performs the backwards step using `self.optimizer`.
        If an evaluator has been assigned to the trainer, it will run evaluation
        at the end of every epoch.

        Parameters
        ----------
        train_loader : GSgnnMultiTaskDataLoader
            The mini-batch sampler for training.
        num_epochs : int
            The max number of epochs to train the model.
        val_loader : GSgnnMultiTaskDataLoader
            The mini-batch sampler for computing validation scores. The validation scores
            are used for selecting models.
        test_loader : GSgnnMultiTaskDataLoader
            The mini-batch sampler for computing test scores.
        use_mini_batch_infer : bool
            Whether or not to use mini-batch inference.
        save_model_path : str
            The path where the model is saved.
        save_model_frequency : int
            The number of iteration to train the model before saving the model.
        save_perf_results_path : str
            The path of the file where the performance results are saved.
        freeze_input_layer_epochs: int
            Freeze the input layer for N epochs. This is commonly used when
            the input layer contains language models.
            Default: 0, no freeze.
        max_grad_norm: float
            Clip the gradient by the max_grad_norm to ensure stability.
            Default: None, no clip.
        grad_norm_type: float
            Norm type for the gradient clip
            Default: 2.0
        """
        # Check the correctness of configurations.
        if self.evaluator is not None:
            assert val_loader is not None, \
                    "The evaluator is provided but validation set is not provided."
        if not use_mini_batch_infer:
            assert isinstance(self._model, GSgnnModel), \
                    "Only GSgnnModel supports full-graph inference."

        # with freeze_input_layer_epochs is 0, computation graph will not be changed.
        static_graph = freeze_input_layer_epochs == 0
        on_cpu = self.device == th.device('cpu')
        if is_distributed():
            model = DistributedDataParallel(self._model,
                                            device_ids=None if on_cpu else [self.device],
                                            output_device=None if on_cpu else self.device,
                                            find_unused_parameters=True,
                                            static_graph=static_graph)
        else:
            model = self._model
        device = model.device
        data = train_loader.data

        # Preparing input layer for training or inference.
        # The input layer can pre-compute node features in the preparing step if needed.
        # For example pre-compute all BERT embeddings
        if freeze_input_layer_epochs > 0:
            self._model.freeze_input_encoder(data)
        # TODO(xiangsx) Support freezing gnn encoder and decoder

        # training loop
        total_steps = 0
        sys_tracker.check('start training')
        g = data.g
        for epoch in range(num_epochs):
            model.train()
            epoch_start = time.time()
            if freeze_input_layer_epochs <= epoch:
                self._model.unfreeze_input_encoder()
            # TODO(xiangsx) Support unfreezing gnn encoder and decoder

            rt_profiler.start_record()
            batch_tic = time.time()
            for i, task_mini_batches in enumerate(train_loader):
                rt_profiler.record('train_sample')
                total_steps += 1

                losses = []
                for (task_info, mini_batch) in task_mini_batches:
                    loss, weight = self._run_mini_batch(data, task_info, mini_batch)
                    losses.append((loss, weight))

                reg_loss = th.tensor(0.).to(device)
                for d_para in model.get_dense_params():
                    reg_loss += d_para.square().sum()
                alpha_l2norm = model.alpha_l2norm

                mt_loss = reg_loss * alpha_l2norm
                mt_loss += loss * weight
                rt_profiler.record('train_forward')
                self.optimizer.zero_grad()
                loss.backward()
                rt_profiler.record('train_backward')
                self.optimizer.step()
                rt_profiler.record('train_step')

                if max_grad_norm is not None:
                    th.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, grad_norm_type)
                self.log_metric("Train loss", loss.item(), total_steps)

                if i % 20 == 0 and get_rank() == 0:
                    rt_profiler.print_stats()
                    logging.info("Epoch %05d | Batch %03d | Train Loss: %.4f | Time: %.4f",
                                 epoch, i, loss.item(), time.time() - batch_tic)

                val_score = None
                if self.evaluator is not None and \
                    self.evaluator.do_eval(total_steps, epoch_end=False):

                    val_score = self.eval(model.module if is_distributed() else model,
                                          data, val_loader, test_loader, total_steps)
                    # TODO(xiangsx): Add early stop support

                # Every n iterations, check to save the top k models. Will save
                # the last k model or all models depends on the setting of top k
                # TODO(xiangsx): support saving the best top k model.
                if save_model_frequency > 0 and \
                    total_steps % save_model_frequency == 0 and \
                    total_steps != 0:

                    if self.evaluator is None or val_score is not None:
                        # We will save the best model when
                        # 1. There is no evaluation, we will keep the
                        #    latest K models.
                        # 2. (TODO) There is evaluaiton, we need to follow the
                        #    guidance of validation score.
                        self.save_topk_models(model, epoch, i, None, save_model_path)

                batch_tic = time.time()
                rt_profiler.record('train_eval')

            # ------- end of an epoch -------
            barrier()
            epoch_time = time.time() - epoch_start
            if get_rank() == 0:
                logging.info("Epoch %d take %.3f seconds", epoch, epoch_time)

            val_score = None
            if self.evaluator is not None and self.evaluator.do_eval(total_steps, epoch_end=True):
                val_score = self.eval(model.module if is_distributed() else model,
                                      data, val_loader, test_loader, total_steps)

            # After each epoch, check to save the top k models.
            # Will either save the last k model or all models
            # depends on the setting of top k.
            self.save_topk_models(model, epoch, None, None, save_model_path)
            rt_profiler.print_stats()
            barrier()



        rt_profiler.save_profile()
        print_mem(device)
        if get_rank() == 0 and self.evaluator is not None:
            # final evaluation
            output = {'best_test_score': self.evaluator.best_test_score,
                       'best_val_score':self.evaluator.best_val_score,
                       'last_test_score': self.evaluator.last_test_score,
                       'last_val_score':self.evaluator.last_val_score,
                       'peak_GPU_mem_alloc_MB': th.cuda.max_memory_allocated(device) / 1024 / 1024,
                       'peak_RAM_mem_alloc_MB': \
                           resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                       'best validation iteration': \
                           self.evaluator.best_iter_num,
                       'best model path': \
                           self.get_best_model_path() if save_model_path is not None else None}
            self.log_params(output)

    def eval(self, model, data, val_loader, test_loader, total_steps,
        use_mini_batch_infer=False, return_proba=True):
        """ do the model evaluation using validation and test sets

        Parameters
        ----------
        model : Pytorch model
            The GNN model.
        data : GSgnnData
            The training dataset
        val_loader: GSNodeDataLoader
            The dataloader for validation data
        test_loader : GSNodeDataLoader
            The dataloader for test data.
        total_steps: int
            Total number of iterations.
        use_mini_batch_infer: bool
            Whether do mini-batch inference
        return_proba: bool
            Whether to return all the predictions or the maximum prediction.

        Returns
        -------
        dict: validation score
        """
        test_start = time.time()
        sys_tracker.check('before prediction')
        model.eval()

        if use_mini_batch_infer:
            emb = do_mini_batch_inference(model, data,
                                          fanout=val_loader.fanout,
                                          task_tracker=self.task_tracker)
        else:
            emb = do_full_graph_inference(model, data,
                                          fanout=val_loader.fanout,
                                          task_tracker=self.task_tracker)
        sys_tracker.check('compute embeddings')

        val_scores = \
            multi_task_mini_batch_predict(model, emb, val_loader, self.device, return_proba) \
            if val_loader is not None else None

        test_scores = \
            multi_task_mini_batch_predict(model, emb, test_loader, self.device, return_proba) \
            if test_loader is not None else None

        sys_tracker.check('after_test_score')
        val_score, test_score = self.evaluator.evaluate(
                val_scores, test_scores, total_steps)
        sys_tracker.check('evaluate validation/test')
        model.train()

        if get_rank() == 0:
            self.log_print_metrics(val_score=val_score,
                                   test_score=test_score,
                                   dur_eval=time.time() - test_start,
                                   total_steps=total_steps)
        return val_score