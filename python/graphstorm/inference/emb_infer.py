"""
    Copyright 2023 Contributors

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Inferer wrapper for embedding generation.
"""
import logging
from graphstorm.config import  (BUILTIN_TASK_NODE_CLASSIFICATION,
                                BUILTIN_TASK_NODE_REGRESSION,
                                BUILTIN_TASK_EDGE_CLASSIFICATION,
                                BUILTIN_TASK_EDGE_REGRESSION,
                                BUILTIN_TASK_LINK_PREDICTION)
from .graphstorm_infer import GSInferrer
from ..model.utils import save_embeddings as save_gsgnn_embeddings
from ..model.gnn import do_full_graph_inference, do_mini_batch_inference
from ..utils import sys_tracker, get_rank, get_world_size, barrier


class GSgnnEmbGenInferer(GSInferrer):
    """ Embedding Generation inferrer.

    This is a high-level inferrer wrapper that can be used directly
    to generate embedding for inferer.

    Parameters
    ----------
    model : GSgnnNodeModel
        The GNN model with different task.
    """
    def infer(self, data, task_type, save_embed_path, eval_fanout,
            use_mini_batch_infer=False,
            node_id_mapping_file=None,
            save_embed_format="pytorch"):
        """ Do Embedding Generating

        Generate node embeddings and save into disk.

        Parameters
        ----------
        data: GSgnnData
            The GraphStorm dataset
        task_type : str
            task_type must be one of graphstorm builtin task types
        save_embed_path : str
            The path where the GNN embeddings will be saved.
        eval_fanout: list of int
            The fanout of each GNN layers used in evaluation and inference.
        use_mini_batch_infer : bool
            Whether to use mini-batch inference when computing node embeddings.
        node_id_mapping_file: str
            Path to the file storing node id mapping generated by the
            graph partition algorithm.
        save_embed_format : str
            Specify the format of saved embeddings.
        """

        device = self.device

        assert save_embed_path is not None, \
            "It requires save embed path for gs_gen_node_embedding"

        sys_tracker.check('start generating embedding')
        self._model.eval()

        # infer ntypes must be sorted for node embedding saving
        if task_type == BUILTIN_TASK_LINK_PREDICTION:
            infer_ntypes = None
        elif task_type in {BUILTIN_TASK_NODE_REGRESSION, BUILTIN_TASK_NODE_CLASSIFICATION}:
            infer_ntypes = sorted(data.infer_idxs)
        elif task_type in {BUILTIN_TASK_EDGE_CLASSIFICATION, BUILTIN_TASK_EDGE_REGRESSION}:
            infer_ntypes = set()
            for etype in data.infer_idxs:
                infer_ntypes.add(etype[0])
                infer_ntypes.add(etype[2])
            infer_ntypes = sorted(infer_ntypes)
        else:
            raise TypeError("Not supported for task type: ", task_type)

        if use_mini_batch_infer:
            embs = do_mini_batch_inference(self._model, data, fanout=eval_fanout,
                                           edge_mask=None,
                                           task_tracker=self.task_tracker,
                                           infer_ntypes=infer_ntypes)
        else:
            embs = do_full_graph_inference(self._model, data, fanout=eval_fanout,
                                           edge_mask=None,
                                           task_tracker=self.task_tracker)
            if infer_ntypes:
                embs = {ntype: embs[ntype] for ntype in infer_ntypes}

        if get_rank() == 0:
            logging.info("save embeddings to %s", save_embed_path)

        save_gsgnn_embeddings(save_embed_path, embs, get_rank(),
            get_world_size(),
            device=device,
            node_id_mapping_file=node_id_mapping_file,
            save_embed_format=save_embed_format)
        barrier()
        sys_tracker.check('save embeddings')