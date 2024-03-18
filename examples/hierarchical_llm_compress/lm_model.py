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
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel

from .hierarchical_compressor import HierarchicalCompressorModel
from .auto_compressor import AutoCompressorModel

def load_hf_tokenizer(model_args, model_path):
    if model_args.use_auth_token:
        from dotenv import load_dotenv, find_dotenv
        _ = load_dotenv(find_dotenv()) # read local .env file
        token = os.environ["HUGGINGFACE_TOKEN"]

    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              use_auth_token=token if model_args.use_auth_token else None)

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    if getattr(tokenizer, "pad_token") is None:
        # tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token='[PAD]'

def load_hf_model(model_path, compress_mode, num_labels, model_args):
    tokenizer = load_hf_tokenizer(model_args, model_path)

    if model_args.use_auth_token:
        from dotenv import load_dotenv, find_dotenv
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
        fanouts = model_args.fanouts
        segment_lengths = model_args.segment_lengths

        output_name += f'_{fanouts}_{segment_lengths}_{model_args.tokenizer_max_length}'
        if accumulate_summary:
            output_name += '_acc_sum'

        config.fanouts = model_args.fanouts
        config.accumulate_summary = model_args.accumulate_summary
        config.segment_lengths = model_args.segment_lengths
        config.tokenizer_max_length = model_args.tokenizer_max_length

        config.hc = True
        encoder = HierarchicalCompressorModel.from_pretrained(model_path,
                                                            config=config,
                                                            graph=graph,
                                                            ntext=text)
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
    return encoder, config

# Pooling function from https://huggingface.co/sentence-transformers/all-mpnet-base-v2
def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()).type_as(token_embeddings)
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
        torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class HLCNodeModel():

class LMForGraphTask(PreTrainedModel):
    """ language model for graph task
    """
    def __init__(self, config, encoder, text):
        super().__init__(config)
        self.encoder = encoder # LM encoder for text encoding
        self.text = text

        # Initialize weights and apply final processing
        self.post_init()

    def encode(self,
               predict_nids: torch.LongTensor = None,
               return_transformer_outputs: bool = False):

        device = self.device

        inputs = self.text[predict_nids]
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        if self.config.hc:
            transformer_outputs = self.encoder(predict_nids)
            # TODO: add attention mask to transformer_outputs
        else:
            transformer_outputs = self.encoder(input_ids, attention_mask, output_hidden_states=True)
        token_embeddings = transformer_outputs.last_hidden_state

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
                sequence_lengths = (
                    torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(device)
            pooled_embeddings = token_embeddings[
                torch.arange(predict_nids.shape[0], device=device), sequence_lengths]

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
    def __init__(self, config, encoder, text):
        super().__init__(config, encoder, text)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, predict_nids, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        embeddings, transformer_outputs = \
            self.encode(predict_nids, return_transformer_outputs=True)


