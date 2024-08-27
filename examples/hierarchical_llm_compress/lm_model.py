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

"""
import os
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple

from dotenv import load_dotenv, find_dotenv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel

from hierarchical_compressor import HierarchicalCompressorModel
from auto_compressor import AutoCompressorModel

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

@dataclass
class NodeClassificationOutput(SequenceClassifierOutput):
    scores: Optional[List[torch.Tensor]] = None

def load_hf_tokenizer(model_args, model_path):
    if model_args.use_auth_token:
        _ = load_dotenv(find_dotenv()) # read local .env file
        token = os.environ["HUGGINGFACE_TOKEN"]

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              use_auth_token=token if model_args.use_auth_token else None)

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    if getattr(tokenizer, "pad_token") is None:
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token='[PAD]'

    return tokenizer

def load_hf_model(model_name, compress_mode, num_labels, model_args, gs_config):
    """ Load language model

    Parameters
    ----------
    model_name: str
        Huggingface model name
    compress_mode: str
        Compression model
    num_labels: int

    model_args: HFArguement
        Huggingface config
    gs_config: GSConfig
        GraphStorm configs
    """
    output_name = model_name
    model_path = SUPPORTED_MODEL_DICT[model_name]
    tokenizer = load_hf_tokenizer(model_args, model_path)

    if model_args.use_auth_token:
        _ = load_dotenv(find_dotenv()) # read local .env file
        token = os.environ["HUGGINGFACE_TOKEN"]
    config = AutoConfig.from_pretrained(model_path,
        num_labels=num_labels,
        use_auth_token=token if model_args.use_auth_token else None)
    config.word_embed_proj_dim = config.hidden_size
    config.padding_side = tokenizer.padding_side
    config.pad_token_id = tokenizer.pad_token_id

    if compress_mode == "hc":
        accumulate_summary = model_args.accumulate_summary
        fanouts = gs_config.fanout
        segment_lengths = model_args.segment_lengths

        output_name += f'_{fanouts}_{segment_lengths}_{model_args.tokenizer_max_length}'
        if accumulate_summary:
            output_name += '_acc_sum'

        config.fanouts = fanouts
        config.accumulate_summary = model_args.accumulate_summary
        config.segment_lengths = model_args.segment_lengths
        config.tokenizer_max_length = model_args.tokenizer_max_length

        config.hc = True
        encoder = HierarchicalCompressorModel.from_pretrained(model_path,
                                                              config=config,
                                                              fanouts=fanouts)
    elif compress_mode == "ac":
        accumulate_summary = model_args.accumulate_summary
        segment_lengths = model_args.segment_lengths

        output_name += f'_{segment_lengths}'
        if accumulate_summary:
            output_name += '_acc_sum'

        config.accumulate_summary = model_args.accumulate_summary
        config.segment_lengths = model_args.segment_lengths

        config.hc = False
        encoder = AutoCompressorModel.from_pretrained(model_path,
                                                    config=config)
    else:
        # compress_mode == "none"
        encoder = AutoModel.from_pretrained(model_path,
                                            config=config,
                                            use_auth_token=token if model_args.use_auth_token else None)
    return encoder, config, output_name

# Pooling function from https://huggingface.co/sentence-transformers/all-mpnet-base-v2
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).type_as(token_embeddings)
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
        torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class LMForGraphTask(PreTrainedModel):
    """ language model for graph task
    """
    def __init__(self, config, encoder):
        super().__init__(config)
        self.encoder = encoder # LM encoder for text encoding

        # Initialize weights and apply final processing
        self.post_init()

    def encode(self,
               batches,
               return_transformer_outputs: bool = False):

        device = self.device

        transformer_outputs = self.encoder(batches, output_hidden_states=True)
        token_embeddings = transformer_outputs.last_hidden_state
        attention_mask = transformer_outputs.attention_mask
        sequence_lengths = transformer_outputs.sequence_lengths

        # Pooling
        if self.config.pooling_strategy == 'mean':
            pooled_embeddings = mean_pooling(token_embeddings, attention_mask)
            # pooled_embeddings = F.normalize(pooled_embeddings, p=2, dim=1)
        elif self.config.pooling_strategy == 'first':
            pooled_embeddings = token_embeddings[:, 0]
        elif self.config.pooling_strategy == 'last':
            if self.config.pad_token_id is None or self.config.padding_side == 'left':
                sequence_lengths = -1
            else:
                sequence_lengths = sequence_lengths
            pooled_embeddings = token_embeddings[
                torch.arange(attention_mask.shape[0], device=device), sequence_lengths]

        if return_transformer_outputs:
            return pooled_embeddings, transformer_outputs
        else:
            return pooled_embeddings

    @torch.no_grad()
    def get_all_text_embeddings(self):
        batch_size = self.config.get_all_text_embeddings_batch_size
        self.encoder.eval()
        embedding_list = []
        for i in range(0, len(self.text), batch_size):
            predict_nids = torch.arange(i, min(i+batch_size, len(self.text)))
            embeddings = self.encode(predict_nids)
            embedding_list.append(embeddings.cpu())

        embeddings = torch.cat(embedding_list, dim=0)

        return embeddings

class LMForGraphNodeTask(LMForGraphTask):
    def __init__(self, config, encoder, target_ntype):
        super().__init__(config, encoder)

        self._target_ntype = target_ntype
        self._hidden_size = config.hidden_size
        # Initialize weights and apply final processing
        self.post_init()

    @property
    def hidden_dim(self):
        return self._hidden_size

    def set_decoder(self, decoder):
        self._decoder = decoder

    def set_loss_func(self, loss_func):
        self._loss_func = loss_func

    def forward(self, predict_nids, labels, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        embeddings, transformer_outputs = \
            self.encode(predict_nids, return_transformer_outputs=True)

        logits = self._decoder[self._target_ntype](embeddings)


        loss = None
        if labels is not None:
            loss = self._loss_func[self._target_ntype](logits, labels)

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return NodeClassificationOutput(
            loss=loss,
            logits=logits
        )


