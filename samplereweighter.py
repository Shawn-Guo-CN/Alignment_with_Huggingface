"""This module contains the `SampleReweighter` class to adjust the 
weights of generations in a batch.

Eatch batch is a dictionary of the following format:
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
     # attention mask of the 2nd  generation
    'generation2_reward': float, # reward of the 2nd generation
    'generation2_weight': float, # weight of the 2nd generation
}
"""

class SampleReweighter(object):
    def __init__(self) -> None:
        raise NotImplementedError


class TensorAnnotator(SampleReweighter):
    # inputs are PyTorch tensors
    def __init__(self) -> None:
        raise NotImplementedError
