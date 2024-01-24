# Created by Shangmin Guo, 2024.1.24
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""DataLoader for loading datasets from Huggingface Datasets Hub.

This module is largely based on the [HALOs repo](https://github.com/ContextualAI/HALOs/blob/main/utils.py).

Each function of the form get_{dataset_name} (e.g., get_shp, get_oasst, etc.)
will return a dict of Example objects, indexed by the prompt for the text.

Each Example object has the following attributes:
- the prompt (formatted with config.human_prefix, config.assistant_prefix)
- a list L of generations
- the index in L of the generation that should be the finetuning target
- a list S of the scores for the generations
- for offline pariwise feedback data: pairs of indices (i,j) in L,
  where generation i is preferable to generation j 
- for offline pointwise feedback data: whether each generation is 
  desirable/chosen or undesirable/rejected
- whether to truncate the beginning or end if the maximum number of tokens is 
  exceeded
- the dataset name
- the unformatted prompt (needed for alpaca)
"""

import datasets
import torch
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import re
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from utils import rank0_print, on_rank0, delete_dict
import pandas as pd


@dataclass
class Example:
    """
    Class for an example in a preference or SFT dataset. 
    If you want each prompt to be uniquely associated with an Example instance, 
    save it in a dict.
    """
    prompt: str = ''                          # prompt for the generated texts
    generations: List[str] = field(default_factory=list) # list of generations
    sft_index: int = -1           # which response should be generated for SFT
    scores: List[float] = field(default_factory=list)   # score of generations
    pairs: List[Tuple[int, int]] = field(default_factory=list)  
    # for pariwise feedback data: indices in responses,
    # where i > j in pair (i,j) is a preference
    desirable: List[bool] = field(default_factory=list) 
    # for pointwise feedback data: whether the generation at the corresponding
    # index in generations is desirable 
    truncation_mode: str = 'keep_end'  
    # if truncation needed, keep the beginning (keep_start) or end (keep_end) 
    # (only override default for SHP)
    dataset_name: str = ''
    original_prompt: str = ''
    # the unformatted prompt (needed to recover instruction for AlpacaEval)

    def num_generations(self):
        return len(self.generations)
    
    def remove_extra_spaces(self):
        """
        Remove double spaces in certain datasets, like Anthropic HH, to 
        standardize spacing.
        """
        clean = lambda x: re.sub(r'[ \t]{2,}', ' ', x)
        self.prompt = clean(self.prompt)
        self.generations = list(map(clean, self.generations))


class Dataset:
    """
    A collection of Example instances, indexed by prompt.
    """
    def __init__(self, name):
        self.name = name
        self.data = defaultdict(Example)

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise KeyError("key must be a string")

        if not isinstance(value, Example):
            raise ValueError("value must be a Example")

        self.data[key] = value

    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)



