"""This module contains the `BatchFactory` class to create batches of training data.

Each batch is a dictionary of the following format:
{
    'prompt': str, # prompt for the generated texts
    'prompt_token_ids':  tensor, # token ids of the prompt
    'prompt_attention_mask': tensor, # attention mask of the prompt

    'generations1': str, # text of the 1st generation
    'generations1_token_ids': tensor, # token ids of the 1st generation
    'generations1_attention_mask': tensor, # attention mask of the 1st generation
    'generation1_reward': float, # reward of the 1st generation
    'generation1_weight': float, # weight of the 1st generation

    'generations2': str, # text of the 2nd generation
    'generations2_token_ids': tensor, # token ids of the 2nd generation
    'generations2_attention_mask': tensor, # attention mask of the 2nd generation
    'generation2_reward': float, # reward of the 2nd generation
    'generation2_weight': float, # weight of the 2nd generation
}
"""
import random
import torch
from transformers import PreTrainedModel, AutoTokenizer
from typing import Optional, Union

import DataLoader


class BatchFactory:
    def __init__(
        self,
        dataset_name: str, # e..g ['hh', 'shp']
        tokenizer, # Huggingface tokenizer object
        generator: Union[None, PreTrainedModel],
        # None for offline data, otherwise a Huggingface model
        annotator: Union[None, PreTrainedModel],
        # None for offline data, otherwise a Huggingface model
        split: str = 'train',
        batch_size: int = 1,
        max_length: int = 512,
        max_prompt_length: int = 128,
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

        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.split = split
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.human_prefix = human_prefix
        self.human_suffix = human_suffix
        self.assistant_prefix = assistant_prefix
        self.assistant_suffix = assistant_suffix
        self.kwargs = kwargs

        self.generator = generator
        self.annotator = annotator
        self.online = self._get_online_flag()
        self.pairwise = self._get_pariwise_flag()

        assert n_epochs is not None or n_examples is not None, \
            "Must specify either n_epochs or n_examples"
        self.n_epochs = n_epochs
        self.epoch_idx = 0
        self.n_examples = n_examples

        self.data_loader = self._get_dataloader()

    def __iter__(self):
        raise NotImplementedError

    def _get_dataloader(self):
        raise NotImplementedError

    def _get_online_flag(self):
        if self.generator is not None and self.annotator is not None:
            return True
        elif self.generator is None and self.annotator is None:
            return False
        else:
            raise ValueError(
                'Must specify both generator and annotator or neither'
            )

    def _get_pariwise_flag(self):
        class_name = type(self).__name__
        if 'pairwise' in class_name.lower():
            return True
        else:
            return False


class OfflinePairwiseBatchFactory(BatchFactory):
    def __init__(
        self,
        dataset_name: str, # e..g ['hh', 'shp']
        tokenizer, # Huggingface tokenizer object
        generator: Union[None, PreTrainedModel],
        # None for offline data, otherwise a Huggingface model
        annotator: Union[None, PreTrainedModel],
        # None for offline data, otherwise a Huggingface model
        split: str = 'train',
        batch_size: int = 1,
        max_length: int = 512,
        max_prompt_length: int = 128,
        n_epochs: Optional[int] = None,
        n_examples: Optional[int] = None,
        human_prefix: str = '\n<|user|>\n', # marks start of human's turn
        human_suffix: str = '', # marks end of human's turn
         # marks start of assistant's turn
        assistant_prefix: str = '\n<|assistant|>\n',
        assistant_suffix: str = '',   # marks end of assistant's turn
        seed:int = 0,
        **kwargs
    ):
        super().__init__(
            dataset_name,
            tokenizer,
            generator,
            annotator,
            split,
            batch_size,
            max_length,
            max_prompt_length,
            n_epochs,
            n_examples,
            human_prefix,
            human_suffix,
            assistant_prefix,
            assistant_suffix,
            seed,
            **kwargs
        )
        assert self.online is False, \
            'Cannot use OfflinePairwiseBatchFactory for online data'

    def __iter__(self):
        return self.data_loader.__iter__()


class SFTBatchFactory(OfflinePairwiseBatchFactory):
    def _get_dataloader(self):
        return DataLoader.SFTDataLoader(
            self.dataset_name,
            self.tokenizer,
            self.split,
            self.batch_size,
            self.max_length,
            self.max_prompt_length,
            1,
            self.n_epochs,
            self.n_examples,
            self.human_prefix,
            self.human_suffix,
            self.assistant_prefix,
            self.assistant_suffix,
            **self.kwargs
        )


class OfflinePointwiseBatchFactory(OfflinePairwiseBatchFactory):
    def _get_dataloader(self):
        assert self.pairwise is False, \
            'OfflinePointwiseBatchFactory cannot be used for pairwise feedback'
        return DataLoader.PointwiseFeedbackDataLoader(
            self.dataset_name,
            self.tokenizer,
            self.split,
            self.batch_size,
            self.max_length,
            self.max_prompt_length,
            1,
            self.n_epochs,
            self.n_examples,
            human_prefix=self.human_prefix,
            human_suffix=self.human_suffix,
            assistant_prefix=self.assistant_prefix,
            assistant_suffix=self.assistant_suffix,
            **self.kwargs
        )


class OfflinePairwiseBatchFactory(OfflinePairwiseBatchFactory):
    def _get_dataloader(self):
        assert self.pairwise is True, 'OfflinePairtwiseBatchFactory ' + \
            'can only be used for pointwise feedback'
        return DataLoader.PointwiseFeedbackDataLoader(
            self.dataset_name,
            self.tokenizer,
            self.split,
            self.batch_size,
            self.max_length,
            self.max_prompt_length,
            1,
            self.n_epochs,
            self.n_examples,
            human_prefix=self.human_prefix,
            human_suffix=self.human_suffix,
            assistant_prefix=self.assistant_prefix,
            assistant_suffix=self.assistant_suffix,
            **self.kwargs
        )


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    batch_factory = OfflinePairwiseBatchFactory(
        ['hh'], tokenizer, None, None, 'test', n_examples=2
    )
    for batch in batch_factory:
        print(batch)
        break
