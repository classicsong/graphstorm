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

import torch as th
import dgl
import torch.nn.functional as F

from graphstorm.dataloading import GSgnnNodeDataLoader
from graphstorm.dataloading.utils import trim_data

class HierarchiCompressNodeDataLoader(GSgnnNodeDataLoader):
    """ The minibatch dataloader for node centric tasks
    """
    def __init__(self, dataset,
                 target_idx, fanout, batch_size,
                 device, train_task, tokenizer_max_length,
                 pad_token_id,
                 pin_memory,
                 num_workers=0, drop_last=True):
        self._tokenizer_max_length = tokenizer_max_length
        self._num_workers = num_workers
        self._drop_last = drop_last
        self._pin_memory = pin_memory
        self._pad_token_id = pad_token_id

        super(HierarchiCompressNodeDataLoader, self).__init__(dataset, target_idx, fanout, batch_size, device, train_task)

    def _prepare_dataloader(self, g, target_idx, fanout, batch_size, train_task, device):
        assert len(target_idx) == 1, "Only support single task training with HierarchiCompress."
        assert len(g.ntypes) == 0 and len(g.etypes) == 0, \
                "HierarchiCompressNodeDataLoader only works with homogeneous graph"

        self._fanout = fanout
        self._batch_size = batch_size
        for ntype in target_idx:
            target_idx[ntype] = trim_data(target_idx[ntype], device)
        sampler = dgl.dataloading.MultiLayerNeighborSampler(fanout)
        # each time we only sample 1 data point
        loader = dgl.dataloading.DistNodeDataLoader(g, target_idx, sampler,
            batch_size=1, shuffle=train_task,
            num_workers=self._num_workers, drop_last=self._drop_last)

        return loader

    def __iter__(self):
        self.dataloader.__iter__()
        return self

    def _prepare_input(self, data, input_nodes, seeds, blocks, tokenizer_max_length):
        """ Convert sampled subgraphs (blocks) into sequences

        Parameters
        ----------
        data: GSgnnData
            The dataset
        input_nodes: th.Tensor
            All the nodes in blocks
        seeds: dict of th.Tensor
            Target nodes
        blocks: dgl.Block
            DGL computation graphs (layer by layer)
        tokenizer_max_length: int
            Max sequence length.
        """
        g = data.g
        if not isinstance(input_nodes, dict):
            assert len(g.ntypes) == 1, \
                    "We don't know the input node type, but the graph has more than one node type."
            input_nodes = {g.ntypes[0]: input_nodes}

        labels = data.get_labels(seeds, self.device)
        input_ids = data.get_node_feats(input_nodes, 'input_ids')
        input_ids = input_ids.flatten()
        attention_masks = data.get_node_feats(input_nodes, 'attention_mask')
        attention_masks = attention_masks.flatten()

        max_nodes_layers = [1 for _ in range(len(blocks))]
        for i in range(len(blocks)):
            for j in range(i):
                max_nodes_layers[j] *= self._fanout[i]
        src_ids = th.full((sum(max_nodes_layers),), -1, dtype=th.int64)
        dst_ids = th.full((sum(max_nodes_layers),), -1, dtype=th.int64)

        offset = 0
        for i, block in enumerate(blocks):
            src, dst = block.in_edges() # u and v are relabeled nids
            src_ids[offset:offset+len(src)] = src
            dst_ids[offset:offset+len(dst)] = dst

            offset += max_nodes_layers[i]

        max_num_nids = sum(max_nodes_layers) + 1
        input_ids = \
            F.pad(input_ids,
                  pad=(0, max_num_nids*self._tokenizer_max_length - len(input_ids)))
        attention_masks = \
            F.pad(attention_masks,
                  pad=(0, max_num_nids*self._tokenizer_max_length - len(attention_masks)))

        return (src_ids, dst_ids, input_ids, attention_masks, labels)

    def __next__(self):
        batches = {
            "src_ids": batch[0],
            "dst_ids": batch[1],
            "input_ids": batch[2],
            "attention_mask": batch[3],
            "labels":batch[4]
        }

        for _ in range(self._batch_size):
            input_nodes, seeds, blocks = self.dataloader.__next__()
            batch = self._prepare_input(self.data, input_nodes, seeds, blocks,
                                        tokenizer_max_length=self._tokenizer_max_length)

            if self._pin_memory:
                batch = [data.pin_memory() if data is not None else None for data in list(batch)]

            batches["src_ids"].append(batch[0])
            batches["dst_ids"].append(batch[1])
            batches["input_ids"].append(batch[2])
            batches["attention_mask"].append(batch[3])
            batches["labels"].append(batch[4])

        batches["src_ids"] = th.stack(batches["src_ids"])
        batches["dst_ids"] = th.stack(batches["dst_ids"])
        batches["input_ids"] = th.stack(batches["input_ids"])
        batches["attention_mask"] = th.stack(batches["attention_mask"])
        batches["labels"] = th.stack(batches["labels"])

        # Build batch from sampled graph.
        return batches
