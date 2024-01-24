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

import random
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Optional
from utils import rank0_print
from .dataset import get_hh, get_hh_harmless, get_hh_helpful
from .dataset import get_oasst, get_shp, get_ultrabin


class DataLoader:
    """
    The base data loader class, similar to the one from the DPO repo.
    Subclass this and overwrite the __iter__ method as needed, since the batch
     elements will be different depending on whether you're doing SFT, aligning 
     with a pairwise loss like DPO, or alignment with a unary loss like KTO. 
    """
    def __init__(
        self, 
        dataset_names: List[str],      # e.g., ['shp', 'oasst'];
        tokenizer,                     # Huggingface tokenizer object
        split: str = 'train',
        batch_size: int = 1,
        max_length: int = 512,      # max length of prompt + response
        max_prompt_length: int = 128,    # max length of prompt alone
        max_prompt_count: int = None,
        n_epochs: Optional[int] = None,
        n_examples: Optional[int] = None, 
        human_prefix: str = '\n<|user|>\n', # marks start of human's turn
        human_suffix: str = '', # marks end of human's turn
        # marks start of assistant's turn
        assistant_prefix: str = '\n<|assistant|>\n',
        assistant_suffix: str = '',   # marks end of assistant's turn
        seed:int = 0,
        **kwargs
    ) -> None:
        torch.manual_seed(seed)
        random.seed(seed)

        self.tokenizer = tokenizer
        self.split = split
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_prompt_count = max_prompt_count
        self.kwargs = kwargs

        assert n_epochs is not None or n_examples is not None, \
            "Must specify either n_epochs or n_examples"
        self.n_epochs = n_epochs
        self.epoch_idx = 0
        self.n_examples = n_examples

        self.full_data = {}

        for name in dataset_names:
            dataset = eval(f"get_{name}")(
                split,
                human_prefix,
                human_suffix,
                assistant_prefix,
                assistant_suffix
            )
            self.full_data.update(dataset.data)

    def collate(self, batch: Dict[str, List]) -> Dict:
        """
        Takes a list of examples (dicts, where values are lists of ints 
        [tokens] or strings [the original texts]) and returns a batch of 
        examples, PyTorch tensors padded to the maximum length. Strings are 
        passed through.
        """
        if self.tokenizer.pad_token_id is None:
            raise Exception("tokenizer's pad_token_id is not specified")

        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('_input_ids') or k.endswith('_attention_mask') \
                or k.endswith('_labels'):
                if 'prompt' in k:
                    # flip prompt so that you are padding to the beginning
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]

                if k.endswith('_input_ids'):
                    padding_value = self.tokenizer.pad_token_id
                elif k.endswith('_labels'):
                    padding_value = -100
                elif k.endswith('_attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(
                    to_pad, batch_first=True, padding_value=padding_value
                )
                if 'prompt' in k:
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    def tokenize_batch_element(
        self,
        prompt: str,
        generation: str,
        truncation_mode: str,
        generation_key: str = 'generation1',
        generation_reward: Optional[float] = None,
        generation_weight: Optional[float] = 1.0,
    ) -> Dict:
        """
        Tokenize a single batch element and truncate if prompt + generation is
        too long. Batch element is turned into Pytorch tensors in self.collate.
        Create the labels for the generation, which are of length equal to the
        sum of the length of the prompt and the generation, with -100 for the 
        prompt tokens.

        Args:
        - prompt: the input/instruction text
        - generation: output text
        - truncation_mode: one of 'keep_start'/'keep_end' 
                          (truncate end/beginning of combined text respectively)
        - generation_key: the key corresponding to the generation 
                          ('generation1' or 'generation2')
        - generation_reward: the reward for the generation
        - generation_weight: the weight for the generation

        Returns:
            A dict of the tokenized prompt, tokenized generation, and the 
            concatenation of the two on all relevant elements (e.g., tokens,
            attention mask, etc.). The generation elements will have keys 
            starting with '{prefix}_' and the concatenated elements will have
            keys starting with '{prefix}_combined_'.
        """
        prompt_token_ids = self.tokenizer.encode(prompt)
        generation_token_ids = self.tokenizer.encode(generation)

        # clip EOS token at end of input
        if len(prompt_token_ids) > 0 and \
            prompt_token_ids[-1] == self.tokenizer.eos_token_id:
            prompt_token_ids.pop()

        # clip BOS token at start of output
        if len(generation_token_ids) > 0 and \
            generation_token_ids[0] == self.tokenizer.bos_token_id:
            generation_token_ids.pop(0)

        # clip EOS at end of output since it will be added later anyway
        if len(generation_token_ids) > 0 and \
            generation_token_ids[-1] == self.tokenizer.eos_token_id:
            generation_token_ids.pop()

        # if combined sequence is too long, first truncate prompt
        if (
            len(prompt_token_ids) + len(generation_token_ids) > self.max_length
        ) and (len(prompt_token_ids) > self.max_prompt_length):
            if truncation_mode == 'keep_start':
                prompt_token_ids = prompt_token_ids[:self.max_prompt_length]
            elif truncation_mode == 'keep_end':
                prompt_token_ids = prompt_token_ids[-self.max_prompt_length:]
            else:
                raise ValueError(f'Unknown truncation mode: {truncation_mode}')

        # then truncate generation if needed
        if (
            len(prompt_token_ids) + len(generation_token_ids) > self.max_length
        ):
            generation_token_ids = generation_token_ids[
                :(self.max_length - len(prompt_token_ids))
            ]

        # reconstitute the prompt and generation
        prompt = self.tokenizer.decode(
            prompt_token_ids, skip_special_tokens=True
        )
        generation = self.tokenizer.decode(
            generation_token_ids,
            skip_special_tokens=True
        ) + ' ' + self.tokenizer.eos_token

        batch_element = {
            'prompt' : prompt,
        }

        for k,v in self.tokenizer(prompt).items():
            batch_element[f'prompt_{k}'] = v

        generation_batch_element = {
            'generation': generation
        }
        for k,v in self.tokenizer(generation).items():
            generation_batch_element[f'generation_{k}'] = v

        # combine the prompt and generation belonging to the same example
        batch_element.update(self.combine_prompt_and_generation(
            batch_element, generation_batch_element, prefix=generation_key
        ))
        batch_element[f'{generation_key}_reward'] = generation_reward
        batch_element[f'{generation_key}_weight'] = generation_weight
  
        return batch_element

    def combine_prompt_and_generation(
        self,
        prompt_dict: Dict,
        generation_dict: Dict,
        prefix: str='generation1'
    ) -> Dict:
        """
        Tokenize the concatenated prompt and generation. 

        Note that you cannot just concatenate the input ids, attention mask, 
        etc. after the fact -- as done in the DPO repo -- because of subtle 
        differences. For example, the ID for 'Well' corresponds to no 
        space ('Well') when at the start of a text but a space ('\n Well) when 
        succeeding a newline. Therefore we could not get the correct token ID 
        for '\nWell' by first tokenizing '\n' then 'Well' then concatenating
        the resulting tokens together.

        The prefix for each concantenated element will be f'{prefix}_combined_'.

        Args:
        - prompt_dict: dict of the prompt text, tokens, attention mask, etc.
        - generation_dict: dict of the generation text, tokens, attention mask,
                           etc.
        - prefix: str to prepend to the the keys of the tokenized 
                  (prompt + generation)

        Returns:
            A dict of the (prompt + generation) text, tokens, attention mask,
            etc, along with the labels for the joint sequence, where the prompt 
            token labels have been set to -100.
        """
        combined_dict = {
            f'{prefix}' : prompt_dict['prompt'] + generation_dict['generation'],
            f'{prefix}_response_only': generation_dict['generation'],
        }

        for k,v in self.tokenizer(
            prompt_dict['prompt'] + generation_dict['generation']
        ).items():
            combined_dict[f'{prefix}_{k}'] = v

        combined_dict[f'{prefix}_labels'] = \
            combined_dict[f'{prefix}_input_ids'][:] 
            # contains both input and response (unpadded)
        combined_dict[f'{prefix}_labels'][
            :len(prompt_dict['prompt_input_ids'])
        ] = [-100] * len(prompt_dict['prompt_input_ids'])

        return combined_dict

    def __iter__(self):
        """Create a flat version of the data and yield batches."""
        raise NotImplementedError


class SFTDataLoader(DataLoader):
    """
    Dataloader for supervised fine-tuning.
    """
    def __iter__(self):
        flat_data = []
        prompts = list(self.full_data.keys())
        # otherwise, will be frontloaded with prompts in same domain
        random.shuffle(prompts) 

        for prompt in prompts:
            flat_data.append(self.full_data[prompt])

        epoch_idx = 0
        example_idx = 0
        done = False

        while True:
            if done: break
            random.shuffle(flat_data)

            batch = []

            for example in flat_data:
                batch_element = self.tokenize_batch_element(
                    # control token will be None for all losses other than csft
                    example.prompt + (
                        self.kwargs.get('chosen_control_token') or ''
                    ),
                    example.generations[example.sft_index],
                    example.truncation_mode,
                    generation_key='generation1',
                    generation_reward=None,
                    generation_weight=1.0,
                )
                batch.append(batch_element)

                if len(batch) == self.batch_size:
                    example_idx += len(batch)
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and \
                        example_idx >= self.n_examples:
                        rank0_print(
                            f'Finished generating {self.n_examples} examples' +\
                            f' on {self.split} split'
                        )
                        done = True
                        break

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break


class PointwiseFeedbackDataLoader(DataLoader):
    """
    Dataloader for losses that require only pointwise feedback (e.g., KTO).

    Since all the datasets have (or imply) pairwise preferences, this function
    assumes all preferred/dispreferred generations are from the desirable/
    undesirable conditional generations given x.
    """
    def get_flat_data(self, prompts):
        """
        Return a flat list of examples given a list of prompts that index 
        self.full_data.
        """
        if self.max_prompt_count:
            num_unique = sum(
                min(self.max_prompt_count, len(self.full_data[prompt].pairs)) \
                for prompt in prompts
            )
        else:
            num_unique = sum(
                len(self.full_data[prompt].pairs) for prompt in prompts
            )

        allowed_desirable = num_unique * self.kwargs.get(
            'frac_unique_desirable', 1.0
        )
        allowed_undesirable = num_unique * self.kwargs.get(
            'frac_unique_undesirable', 1.0
        )
        seen_desirable = 0
        seen_undesirable = 0

        flat_data = []

        for prompt in prompts:
            example = self.full_data[prompt]

            if self.max_prompt_count:
                example.pairs = random.sample(
                    example.pairs,
                    min(self.max_prompt_count, len(example.pairs))
                )

            for i,j in example.pairs:
                if seen_desirable < allowed_desirable:
                    flat_data.append((
                        example, example.generations[i], 'desired'
                    ))
                    seen_desirable += 1

                if seen_undesirable < allowed_undesirable:
                    flat_data.append(
                        (example, example.generations[j], 'undesired')
                    )
                    seen_undesirable += 1

        return flat_data

    def __iter__(self):
        prompts = list(self.full_data.keys()) 
        random.shuffle(prompts)
        # otherwise, will be frontloaded with prompts in same domain
        flat_data = self.get_flat_data(prompts)

        epoch_idx = 0
        example_idx = 0
        done = False

        while True:
            if done: break
            random.shuffle(flat_data)
            # so generations in the same preference are not in the same batch
            batch = []
            example_queue = []

            for example, generation, status in flat_data:
                generation_reward = 1.0 if status == 'desired' else 0.0
                generation_weight = 1.0
                batch_element = self.tokenize_batch_element(
                    example.prompt,
                    generation,
                    example.truncation_mode,
                    generation_key='generation1',
                    generation_reward=generation_reward,
                    generation_weight=generation_weight,
                )
                batch_element['truncation_mode'] = example.truncation_mode
                example_queue.append(batch_element)

                if len(example_queue) >= self.batch_size:
                    while len(batch) < self.batch_size:
                        batch.append(example_queue.pop(0))

                if len(batch) >= self.batch_size:
                    # for estimating the KL term, match up x and y' that are 
                    # not corresponding input-output pairs in the data
                    # for x_i, get a mismatched y' by just picking the
                    # subsequent y_{i+1} in the batch (desirable/undesirable
                    # status does not matter) the respective input IDs,
                    # attention mask, and so on will be prefixed by the term KL
                    indices = list(range(1, len(batch))) + [0]
                    for i in range(len(batch)):
                        batch[i].update(self.tokenize_batch_element(
                            batch[i]['prompt'],
                            batch[indices[i]]['generation1'],
                            batch[i]['truncation_mode'],
                            generation_key='generation2',
                            generation_reward=0.0,
                            generation_weight=1.0,
                        ))

                    example_idx += len(batch)
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and \
                        example_idx >= self.n_examples:
                        rank0_print(
                            f'Finished generating {example_idx} examples ' + \
                            f'on {self.split} split'
                        )
                        done = True
                        break

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break


class PairwiseFeedbackDataLoader(DataLoader):
    """
    Dataloader for losses that do require pairwise feedback (e.g., DPO).
    """
    def __iter__(self):
        flat_data = []
        prompts = list(self.full_data.keys())
        random.shuffle(prompts)
        # otherwise, will be frontloaded with prompts in same domain

        for prompt in prompts:
            example = self.full_data[prompt]

            if self.max_prompt_count:
                example.pairs = random.sample(
                    example.pairs,
                    min(self.max_prompt_count, len(example.pairs))
                )

            for pair in example.pairs:
                flat_data.append((example, pair))

        epoch_idx = 0
        example_idx = 0
        done = False

        while True:
            if done: break
            random.shuffle(flat_data)
            batch = []

            for example, (i,j) in flat_data:
                batch_element = {}
                batch_element.update(self.tokenize_batch_element(
                        example.prompt,
                        example.generations[i],
                        example.truncation_mode,
                        generation_key='generation1',
                        generation_reward=1.0,
                        generation_weight=1.0,
                ))
                batch_element.update(self.tokenize_batch_element(
                    example.prompt,
                    example.generations[j],
                    example.truncation_mode,
                    generation_key='generation2',
                    generation_reward=0.0,
                    generation_weight=1.0,
                ))
                batch.append(batch_element)

                if len(batch) >= self.batch_size:
                    example_idx += len(batch)
                    yield self.collate(batch)
                    batch = []

                    if self.n_examples is not None and \
                        example_idx >= self.n_examples:
                        rank0_print(
                            f'Finished {example_idx} examples on ' + \
                            f'{self.split} split'
                        )
                        done = True
                        break

            epoch_idx += 1
            if self.n_epochs is not None and epoch_idx >= self.n_epochs:
                done = True
                break
