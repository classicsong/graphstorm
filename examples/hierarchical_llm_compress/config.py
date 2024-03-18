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

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default='./.cache/',
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    peft_model: Optional[str] = field(
        default=None,
        metadata={"help": "If not none, perform peft, one of `prompt_tuning` or `prefix_tuning`"},
    )
    peft_num_virtual_tokens: Optional[int] = field(
        default=16,
        metadata={"help": "Number of tokens for one of `prompt_tuning` or `prefix_tuning`"},
    )
    lora_r: Optional[int] = field(
        default=16,
        metadata={"help": "Lora attention dimension"},
    )
    lora_alpha: Optional[int] = field(
        default=16,
        metadata={"help": "The alpha parameter for Lora scaling"},
    )
    tokenizer_max_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max length of input sequence, used for padding and truncation during tokenization"},
    )
    segment_lengths: Optional[int] = field(
        default=512,
        metadata={"help": "Segment lengths for autocompressor"},
    )
    accumulate_summary: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to accumulate summary for autocompressor"},
    )
    get_all_text_embeddings_batch_size: Optional[int] = field(
        default=256,
        metadata={"help": "Batch size for get_all_text_embeddings"},
    )
    pooling_strategy: Optional[str] = field(
        default="mean",
        metadata={"help": "Pooling strategy from token embeddings to sentence embeddings "},
    )

