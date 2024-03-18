"""
    Copyright 2024 Contributors

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
import dgl
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from typing import Optional, Union, List, Tuple
from auto_compressor import AutoCompressorModel


def rearrange_and_trim(input_ids, attention_mask):
    # Get the maximum sequence length based on the attention mask
    max_len = torch.max(attention_mask.sum(dim=1)).item()
    compacted_input_ids = torch.zeros((input_ids.shape[0], max_len), dtype=torch.long).to(input_ids.device)
    new_attention_mask = torch.zeros((input_ids.shape[0], max_len), dtype=torch.long).to(input_ids.device)

    for i, (batch, mask) in enumerate(zip(input_ids, attention_mask)):
        selected_tokens = batch[mask.bool()]
        num_tokens = len(selected_tokens)

        compacted_input_ids[i, :num_tokens] = selected_tokens
        new_attention_mask[i, :num_tokens] = 1

    return compacted_input_ids, new_attention_mask


class HierarchicalCompressorModel(AutoCompressorModel):
    def __init__(self, config, data, fanouts):
        super().__init__(config)
        self.data = data
        self.sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
        self.fanouts = fanouts
        self.accumulate_summary = config.accumulate_summary
        self.segment_lengths = config.segment_lengths
        self.tokenizer_max_length = config.tokenizer_max_length

        self.pad_token_id = config.pad_token_id
        self.summary_length = config.summary_length
        self.emb_dim = config.word_embed_proj_dim

        # For context inputs padding via masking the placeholder tensor
        self.context_batch_masks = {i: torch.tril(torch.ones(fanout+1, fanout), diagonal=-1).bool() for i, fanout in enumerate(self.fanouts)}

        # Initialize weights and apply final processing
        self.post_init()

        # Disable gradient for the summary embeddings
        self.embed_summary.weight.requires_grad = False


    def compress(self, batches):
        device = self.device
        fanouts = self.fanouts
        tokenizer_max_length = self.tokenizer_max_length
        emb_dim = self.emb_dim
        pad_token_id =  self.pad_token_id
        segment_lengths = self.segment_lengths

        src_ids = batches["src_ids"]
        dst_ids = batches["dst_ids"]
        input_ids = batches["input_ids"]
        attention_masks = batches["attention_mask"]

        # rebuild blocks
        max_nodes_layers = [1 for _ in range(len(fanouts))]
        for i in range(len(fanouts)):
            for j in range(i):
                max_nodes_layers[j] *= self._fanout[i]

        historical_summary = [None for _ in range(src_ids.shape[0])]
        offset = 0
        for i, fanout in enumerate(fanouts):
            context_input_ids_batch = []
            context_attention_mask_batch = []
            num_dst_batch = []
            historical_summary_batch = []
            for batch_id in range(src_ids.shape[0]):
                src = src_ids[batch_id][offset:offset+max_nodes_layers[i]]
                dst = dst_ids[batch_id][offset:offset+max_nodes_layers[i]]
                exist_edges = dst != -1
                src = src[exist_edges]
                dst = dst[exist_edges]
                num_dst = len(dst)

                context_placeholder_shape = \
                    [num_dst * fanout, tokenizer_max_length]
                # fill input_ids with pad_token_id
                context_input_ids = \
                    torch.ones(context_placeholder_shape).long()
                context_input_ids = context_input_ids * pad_token_id
                # fill attention_mask with 0
                context_attention_mask = \
                    torch.zeros(context_placeholder_shape).long()
                # construct the mask according to the actual number of neighbors
                counts = torch.bincount(dst, minlength=num_dst)

                context_batch_mask = self.context_batch_masks[i][counts].view(-1)

                context_input_ids[context_batch_mask] = input_ids[batch_id][src]
                context_attention_mask[context_batch_mask] = attention_masks[batch_id[src]]

                context_input_ids = context_input_ids.view(num_dst, tokenizer_max_length * fanout).to(device)

                context_attention_mask = context_attention_mask.view(num_dst, tokenizer_max_length * fanout).to(device)

                # select historical_summary and pad them by inserting them into a placeholder tensor
                if historical_summary[i] is not None:
                    historical_summary_placeholder_shape = \
                        [num_dst * fanout] + list(historical_summary[i].shape[1:])
                    new_historical_summary = \
                        torch.zeros(historical_summary_placeholder_shape).type(historical_summary[i].dtype).to(device)
                    new_historical_summary[context_batch_mask] = historical_summary[i][src]
                    historical_summary[i] = new_historical_summary.view(num_dst, -1, emb_dim)
                    historical_summary_batch.append(historical_summary[i])


                context_input_ids_batch.append(context_input_ids)
                context_attention_mask_batch.append(context_attention_mask)
                num_dst_batch.append(num_dst)

            context_input_ids_batch = \
                torch.cat(context_input_ids_batch, dim= 0)
            context_attention_mask_batch = \
                torch.cat(context_attention_mask_batch, dim=0)
            historical_summary_batch = None \
                if len(historical_summary_batch) == 0 \
                else torch.cat(historical_summary_batch, dim=0)
            num_dst = sum(num_dst_batch)

            with torch.no_grad():
                context_summary = self._forward(context_input_ids_batch,
                                                context_attention_mask_batch,
                                                return_dict=True,
                                                use_cache=True,
                                                segment_lengths=segment_lengths,
                                                softprompt=historical_summary_batch,
                                                output_softprompt=True).softprompt
            historical_summary = torch.split(context_summary, num_dst_batch, dim=0)
            # break summary

        return historical_summary

    def forward(self,
                batches: dict = None
                ) -> Union[Tuple, SequenceClassifierOutputWithPast]:

        predict_outputs = self.ntext[predict_nids]
        predict_input_ids = predict_outputs['input_ids'].to(self.device)
        predict_attention_mask = predict_outputs['attention_mask'].to(self.device)

        src_ids = batches["src_ids"]
        dst_ids = batches["dst_ids"]
        input_ids = batches["input_ids"]
        attention_masks = batches["attention_mask"]

        # get predict_input_ids and predict_attention_mask

        # TODO: summarize multiple times
        summary = self.compress(batches)
        transformer_outputs = self._forward(predict_input_ids,
                                            predict_attention_mask,
                                            return_dict=True,
                                            use_cache=True,
                                            softprompt=summary)

        transformer_outputs.last_hidden_state = transformer_outputs.last_hiddens

        return transformer_outputs

    def _compress(self, layers):
        for i, layer in enumerate(layers):


            with torch.no_grad():
                context_summary = self._forward(context_input_ids,
                                                context_attention_mask,
                                                return_dict=True,
                                                use_cache=True,
                                                segment_lengths=segment_lengths,
                                                softprompt=historical_summary,
                                                output_softprompt=True).softprompt

    def forward(self, xxx):

        # TODO: summarize multiple times
        summary = self._compress(xxx)