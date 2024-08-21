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

    Entry point for node classification tasks.
"""
import os
import logging
import evaluate

import numpy as np

import graphstorm as gs
from graphstorm.config import get_argument_parser
from graphstorm.config import GSConfig
from graphstorm.config import (BUILTIN_TASK_NODE_CLASSIFICATION,
                               BUILTIN_TASK_NODE_REGRESSION)
from graphstorm.dataloading import GSgnnData
from graphstorm.model.node_decoder import EntityClassifier, EntityRegression
from graphstorm.model.loss_func import (ClassifyLossFunc,
                                        RegressionLossFunc)
from graphstorm.utils import setup_device, get_log_level

import transformers
from transformers import (
    TrainingArguments,
    HfArgumentParser,
)


from config import ModelArguments
from lm_model import load_hf_tokenizer, load_hf_model
from lm_model import LMForGraphNodeTask
from hlc_trainer import HLCNodePredictionTrainer
from dataloader import HierarchiCompressNodeDataLoader

SUPPORTED_MODEL_DICT = {
    "bert": "bert-base-cased",
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "opt-1.3b": "facebook/opt-1.3b",
    "opt-2.7b": "facebook/opt-2.7b",
    "ac-opt-1.3b": "princeton-nlp/AutoCompressor-1.3b-30k",
    "hc-opt-1.3b": "princeton-nlp/AutoCompressor-1.3b-30k",
    "hc-opt-2.7b": "princeton-nlp/AutoCompressor-2.7b-6k",
    "mistral": "mistralai/Mistral-7B-v0.1",
}

def create_node_hierarchical_lm_compress_model(args, gs_config, model_args):
    model_name = args.hnc_lm_model
    compress_mode = args.compress_mode
    num_labels = gs_config.num_classes
    encoder, config = load_hf_model(model_name, compress_mode, num_labels, model_args)


    config.get_all_text_embeddings_batch_size = model_args.get_all_text_embeddings_batch_size
    config.pooling_strategy = model_args.pooling_strategy

    output_name += f"_{config.pooling_strategy}_pooling"
    model = LMForGraphNodeTask(config=config, encoder=encoder)

    if gs_config.task_type == BUILTIN_TASK_NODE_CLASSIFICATION:
            decoder = {}
            loss_func = {}
            for ntype in gs_config.target_ntype:
                decoder[ntype] = EntityClassifier(model.gnn_encoder.out_dims \
                                                if model.gnn_encoder is not None \
                                                else model.node_input_encoder.out_dims,
                                               gs_config.num_classes[ntype],
                                               gs_config.multilabel[ntype])
                loss_func[ntype] = ClassifyLossFunc(gs_config.multilabel[ntype],
                                                gs_config.multilabel_weights[ntype],
                                                gs_config.imbalance_class_weights[ntype])

            model.set_decoder(decoder)
            model.set_loss_func(loss_func)

    elif gs_config.task_type == BUILTIN_TASK_NODE_REGRESSION:
        model.set_decoder(EntityRegression(model.gnn_encoder.out_dims \
                                            if model.gnn_encoder is not None \
                                            else model.node_input_encoder.out_dims))
        model.set_loss_func(RegressionLossFunc())
    else:
        raise ValueError('unknown node task: {}'.format(gs_config.task_type))


def main(args):
    hf_parser = HfArgumentParser((ModelArguments, TrainingArguments))

    gs.initialize(ip_config=args.ip_config, backend="gloo")
    config = GSConfig(args)
    device = setup_device(config.local_rank)

    model_args, training_args = hf_parser.parse_yaml_file(yaml_file=os.path.abspath(args.hf_config_file))

    # Setup huggingface logging
    if training_args.should_log:
        # The default of training_args.log_level is passive,
        # so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    log_level = get_log_level(args.logging_level) \
                if hasattr(args, "logging_level") else logging.INFO
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logging.debug(
        "Process rank: %d, device: %s, n_gpu: %d 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        training_args.fp16)
    model = create_node_hierarchical_lm_compress_model(args, config, model_args)

    train_data = GSgnnData(config.part_config,
                           node_feat_field=config.node_feat_name,
                           label_field=config.label_field)

    f1_macro_metric = evaluate.load("f1")
    accuracy_metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        results = {}
        results.update(f1_macro_metric.compute(predictions=preds, references = labels, average="macro"))
        results.update(accuracy_metric.compute(predictions=preds, references = labels))
        return results

    dataloader = HierarchiCompressNodeDataLoader(train_data,
                                                 target_idx=train_data.train_idxs,
                                                 fanout=config.fanout,
                                                 batch_size=config.batch_size,
                                                 device=device,
                                                 train_task=True,
                                                 tokenizer_max_length=model_args.tokenizer_max_length,
                                                 pad_token_id=config.pad_token_id,
                                                 pin_memory=True)
    fanout = config.eval_fanout if config.use_mini_batch_infer else []
    if len(train_data.val_idxs) > 0:
        val_dataloader = HierarchiCompressNodeDataLoader(train_data,
                                                        target_idx=train_data.val_idxs,
                                                        fanout=fanout,
                                                        batch_size=config.eval_batch_size,
                                                        device=device,
                                                        train_task=False,
                                                        tokenizer_max_length=model_args.tokenizer_max_length,
                                                        pad_token_id=config.pad_token_id,
                                                        pin_memory=True)
    if len(train_data.test_idxs) > 0:
        test_dataloader = HierarchiCompressNodeDataLoader(train_data,
                                                        target_idx=train_data.test_idxs,
                                                        fanout=fanout,
                                                        batch_size=config.eval_batch_size,
                                                        device=device,
                                                        train_task=False,
                                                        tokenizer_max_length=model_args.tokenizer_max_length,
                                                        pad_token_id=config.pad_token_id,
                                                        pin_memory=True)
    trainer = HLCNodePredictionTrainer(model)
    trainer.setup_device(device=device)
    trainer.fit(
        train_data,
        training_args,
        dataloader,
        compute_metrics,
        val_dataloader,
        test_dataloader)

if __name__ == '__main__':
    argparser = get_argument_parser()

    argparser.add_argument("--hnc-lm-model", type=str, required=True,
                           choices=list(SUPPORTED_MODEL_DICT.keys()),
                           help="Language model used by the hierachical neighbor compression "
                           f"model, supported models include {SUPPORTED_MODEL_DICT.keys()}")
    argparser.add_argument("--compress-mode", type=str, default="hc",
                           choices=["hc", "ac", "none"],
                           help="Compression mode. It can be hc for hierarchical compressor, "
                                "ac for auto compressor and none for no compressor")
    argparser.add_argument("--hf-config-file", type=str, required=True,
                           help="Huggingface configuration file path.")

    args, _ = argparser.parse_known_args()
    main(args)
