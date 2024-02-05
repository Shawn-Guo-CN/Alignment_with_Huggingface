"""This module contains the `BatchFactory` class to create batches of training data.

Each batch is a dictionary of the following format:
{
    'prompt': str, # prompt for the generated texts
    'prompt_token_ids':  tensor, # token ids of the prompt
    'prompt_attention_mask': tensor, # attention mask of the prompt

    'generations1': str, # text of the 1st generation
    'generations1_token_ids': tensor, # token ids of the 1st generation
    'generations1_attention_mask': tensor,
    # attention mask of the 1st generation
    'generation1_reward': float, # reward of the 1st generation
    'generation1_weight': float, # weight of the 1st generation

    'generations2': str, # text of the 2nd generation
    'generations2_token_ids': tensor, # token ids of the 2nd generation
    'generations2_attention_mask': tensor,
    # attention mask of the 2nd generation
    'generation2_reward': float, # reward of the 2nd generation
    'generation2_weight': float, # weight of the 2nd generation
}
"""
import random
import torch
from transformers import PreTrainedModel, AutoTokenizer
from typing import Optional, Union

import DataLoader
from samplereweighter import SampleReweighter


class BatchFactory:
    def __init__(
        self,
        dataset_name: str, # e..g ['hh', 'shp']
        tokenizer, # Huggingface tokenizer object
        generator: Union[None, PreTrainedModel] = None,
        # None for offline data, otherwise a Huggingface model
        annotator: Union[None, PreTrainedModel] = None,
        # None for offline data, otherwise a Huggingface model
        reweighter: Union[None, SampleReweighter] = None,
        # None for on reweighting, otherwise a SampleReweighter object
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
        self.reweighter = reweighter

        assert n_epochs is not None or n_examples is not None, \
            "Must specify either n_epochs or n_examples"
        self.n_epochs = n_epochs
        self.epoch_idx = 0
        self.n_examples = n_examples

        self.online = self._get_online_flag()
        self.pairwise = self._get_pariwise_flag()
        self.reweight = self._get_reweight_flag()
        self._check_type()
        self.data_loader = self._get_dataloader()

    def _check_type(self):
        raise NotImplementedError

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

    def _get_reweight_flag(self):
        if self.reweighter is not None:
            return True
        else:
            return False


class OfflineBatchFactory(BatchFactory):
    """BatchFactory offline data with or without reweighting.

    This class is to generate offline batches for training the policy model.
    The batches are from the `DataLoader` class. If `reweight` is True, the
    `SampleReweighter` class is used to adjust the weights of generations in
    a batch. Otherwise, the batches are yielded directly from the data loader.
    """
    def _check_type(self):
        assert self.online is False, \
            'Cannot use OfflineBatchFactory for online data'
        self._offline_check_type()

    def __iter__(self):
        if self.reweight:
            batch = next(self.data_loader.__iter__())
            batch = self.reweighter(batch)
            return batch
        else:
            return self.data_loader.__iter__()

    def _offline_check_type(self):
        raise NotImplementedError


class SFTBatchFactory(OfflineBatchFactory):
    """BatchFactory for SFT data.

    Each batch is a dictionary of the following format:
    {
        'prompt': str, # prompt for the generated texts
        'prompt_token_ids':  tensor, # token ids of the prompt
        'prompt_attention_mask': tensor, # attention mask of the prompt

        'generations1': str, # text of prompt + 1st generation
        'generation1_response_only': str, # text of the 1st generation only
        'generations1_token_ids': tensor, # token ids of the 1st generation
        'generations1_attention_mask': tensor, # attn mask of the 1st generation
        'generation1_reward': float, # reward of the 1st generation
        'generation1_weight': float, # weight of the 1st generation
    }
    """
    def _get_dataloader(self):
        return DataLoader.SFTDataLoader(
            self.dataset_name,
            self.tokenizer,
            self.split,
            self.batch_size,
            self.max_length,
            self.max_prompt_length,
            self.batch_size,
            self.n_epochs,
            self.n_examples,
            self.human_prefix,
            self.human_suffix,
            self.assistant_prefix,
            self.assistant_suffix,
            **self.kwargs
        )

    def _offline_check_type(self):
        assert self.reweight is False, \
            'SFTBatchFactory should not be used with reweighting.'


class OfflinePointwiseBatchFactory(OfflineBatchFactory):
    """BatchFactory for offline data with pointwise feedback.

    Each batch is a dictionary of the following format:
    {
        'prompt': str, # prompt for the generated texts
        'prompt_token_ids':  tensor, # token ids of the prompt
        'prompt_attention_mask': tensor, # attention mask of the prompt

        'generations1': str, # text of prompt + 1st generation
        'generation1_response_only': str, # text of the 1st generation only
        'generations1_token_ids': tensor, # token ids of the 1st generation
        'generations1_attention_mask': tensor, # attn mask of the 1st generation
        'generation1_reward': float, # reward of the 1st generation
        'generation1_weight': float, # weight of the 1st generation
    }
    """
    def _get_dataloader(self):
        return DataLoader.PointwiseFeedbackDataLoader(
            self.dataset_name,
            self.tokenizer,
            self.split,
            self.batch_size,
            self.max_length,
            self.max_prompt_length,
            self.batch_size,
            self.n_epochs,
            self.n_examples,
            human_prefix=self.human_prefix,
            human_suffix=self.human_suffix,
            assistant_prefix=self.assistant_prefix,
            assistant_suffix=self.assistant_suffix,
            **self.kwargs
        )

    def _offline_check_type(self):
        assert self.pairwise is False, \
            'OfflinePointwiseBatchFactory cannot be used for pairwise feedback'


class OfflinePairwiseBatchFactory(OfflineBatchFactory):
    """BatchFactory for offline data with pointwise feedback.

    Each batch is a dictionary of the following format:
    {
        'prompt': str, # prompt for the generated texts
        'prompt_token_ids':  tensor, # token ids of the prompt
        'prompt_attention_mask': tensor, # attention mask of the prompt

        'generation1': str, # text of prompt + 1st generation
        'generation1_response_only': str, # text of the 1st generation only
        'generation1_token_ids': tensor, # token ids of the 1st generation
        'generation1_attention_mask': tensor, # attn mask of the 1st generation
        'generation1_reward': float, # reward of the 1st generation
        'generation1_weight': float, # weight of the 1st generation

        'generation2': str, # text of prompt + 2nd generation
        'generation2_response_only': str, # text of the 2nd generation only
        'generation2_token_ids': tensor, # token ids of the 2nd generation
        'generation2_attention_mask': tensor, # attn mask of the 2nd generation
        'generation2_reward': float, # reward of the 2nd generation
        'generation2_weight': float, # weight of the 2nd generation
    }
    """
    def _get_dataloader(self):
        return DataLoader.PointwiseFeedbackDataLoader(
            self.dataset_name,
            self.tokenizer,
            self.split,
            self.batch_size,
            self.max_length,
            self.max_prompt_length,
            self.batch_size,
            self.n_epochs,
            self.n_examples,
            human_prefix=self.human_prefix,
            human_suffix=self.human_suffix,
            assistant_prefix=self.assistant_prefix,
            assistant_suffix=self.assistant_suffix,
            **self.kwargs
        )

    def _offline_check_type(self):
        assert self.pairwise is True, \
            'OfflinePairwiseBatchFactory must be used for pairwise feedback'


class OnlineBatchFactory(BatchFactory):
    """BatchFactor with online annotator and generator.

    This class is to generate online batches fro training the policy model.
    The data loader used in this class is `PromptDataLoader` which yields
    only the prompts in the dataset. The responses are generated by the
    generator on the fly. The responses are then annotated by the annotator
    before being returned as a batch.
    """
    def _check_type(self):
        assert self.online is True, \
            'Cannot use OfflinePairwiseBatchFactory for online data'
        self._online_check_type()

    def _get_dataloader(self):
        return DataLoader.PromptDataLoader(
            self.dataset_name,
            self.tokenizer,
            self.split,
            self.batch_size,
            self.max_length,
            self.max_prompt_length,
            self.batch_size,
            self.n_epochs,
            self.n_examples,
            self.human_prefix,
            self.human_suffix,
            self.assistant_prefix,
            self.assistant_suffix,
            **self.kwargs
        )

    def _get_batch(self):
        raise NotImplementedError

    def _get_sample_from_genearator(self):
        raise NotImplementedError

    def __iter__(self):
        if self.reweight:
            batch = self._get_batch()
            batch = self.reweighter(batch)
            return batch
        else:
            return self._get_batch()

    def _online_check_type(self):
        raise NotImplementedError


class OnlinePointwiseBatchFactory(OnlineBatchFactory):
    def _online_check_type(self):
        raise NotImplementedError


class OnlinePairwiseBatchFactory(OnlineBatchFactory):
    def _online_check_type(self):
        raise NotImplementedError

