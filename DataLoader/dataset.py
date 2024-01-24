"""Functions for getting datasets from Huggingface hub.

This module is largely based on the [HALOs repo](https://github.com/ContextualAI/HALOs/blob/main/utils.py).

Each function below returns an instance of the `Dataset` class defined in 
`dataloader.py`.
"""
import datasets
import pandas as pd
import random
import re
import tqdm
from typing import List

from .dataloader import Dataset
from utils import rank0_print, on_rank0


def get_shp(
    split: str,
    human_prefix: str,
    human_suffix: str,
    assistant_prefix: str,
    assistant_suffix: str
) -> Dataset:
    """
    Load the Stanford Human Preferences dataset from Huggingface and convert it
    into to a Dataset.

    We filter preference pairs to only keep pairs where the score ratio is at 
    least 2 (as in original SHP). For this dataset, the SFT text is the first
    response in SHP for a given prompt. This is because the globally best 
    response cannot be inferred from SHP, but all responses are a good option 
    because they have a positive score.

    As recommended in the SteamSHPs' (reward models) data cards:
        Maximum number of pairs per prompt is 5 
            (in the training data, to avoid overfitting).
        Minimum score ratio of preferred to dispreferred response is 2

    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the 
                        recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice 
                        and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the 
                            recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended 
                            choice and is set in config.yaml)

    Returns:   
        A Dataset instance.
    """
    MAX_PAIRS_PER_PROMPT = 5
    MIN_SCORE_RATIO = 2

    rank0_print(f'Loading SHP dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset('stanfordnlp/SHP', split=split)
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing SHP')

    data = Dataset('shp')

    for row in dataset:
        prompt = human_prefix + row['history'] + human_suffix + assistant_prefix
        responses = [
            row['human_ref_A'] + assistant_suffix,
            row['human_ref_B'] + assistant_suffix,
        ]
        scores = [row['score_A'], row['score_B']]
        score_ratio = max(scores[0] / scores[1], scores[1] / scores[0])

        if score_ratio < MIN_SCORE_RATIO and split == 'train':
            continue

        i,j = data[prompt].num_generations(), data[prompt].num_generations() + 1
        data[prompt].prompt = prompt
        data[prompt].generations.extend(responses)
        data[prompt].pairs.append((i, j) if row['labels'] == 1 else (j, i))
        data[prompt].scores.extend(scores)
        # keep start for SHP because it's single-turn with long prompts
        data[prompt].truncation_mode = 'keep_start' 
        # absolute best response cannot be inferred, so just pick the first
        data[prompt].sft_index = 0  
        data[prompt].dataset_name = 'shp'
        data[prompt].remove_extra_spaces()

    # prevent over-fitting
    if split == 'train':
        for prompt in data:
            data[prompt].pairs = random.sample(
                data[prompt].pairs,
                min(MAX_PAIRS_PER_PROMPT, len(data[prompt].pairs))
            )

    return data


def get_hh(
    split: str,
    human_prefix: str,
    human_suffix: str,
    assistant_prefix: str,
    assistant_suffix: str,
    only_helpful = False,
    only_harmless = False
) -> Dataset:
    """
    Load the Anthropic Helpful-Harmless dataset from Huggingface and convert it 
    into to a Dataset. For this dataset, the SFT text is the preferred response.

    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the 
                        recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice 
                        and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the
                            recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended 
                            choice and is set in config.yaml)
        - only_helpful: only the helpfulness data
        - only_harmless: only the harmlessness data

    Returns:   
        A Dataset instance.
    """
    if only_helpful:
        dataset = datasets.load_dataset(
            'Anthropic/hh-rlhf', split=split, data_dir="helpful-base"
        )
        data = Dataset('Anthropic-HH-helpful')
    elif only_harmless:
        dataset = datasets.load_dataset(
            'Anthropic/hh-rlhf', split=split, data_dir="harmless-base"
        )
        data = Dataset('Anthropic-HH-harmless')
    else:
        rank0_print(f'Loading HH dataset ({split} split) from Huggingface...')
        dataset = datasets.load_dataset('Anthropic/hh-rlhf', split=split)
        data = Dataset('Anthropic-HH')
        
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing HH')

    def split_prompt_and_responses(ex):
        search_term = '\n\nAssistant: '
        search_term_idx = ex['chosen'].rfind(search_term)
        prompt = ex['chosen'][:search_term_idx + len(search_term)]
        chosen_response = ex['chosen'][len(prompt):]
        rejected_response = ex['rejected'][len(prompt):]
        return prompt, chosen_response, rejected_response

    for row in dataset:
        prompt, chosen, rejected = split_prompt_and_responses(row)
        # strip trailing spaces to avoid tokenization issues
        chunks = []
        # turn doesn't always start with \n\n so watch out
        for chunk in re.split(r'\s*(Human:|Assistant:)\s+', prompt): 
            if chunk.startswith('Human'):
                chunk = re.sub(r'\s*Human:\s*', human_prefix, chunk) + \
                        human_suffix
            elif chunk.startswith('Assistant'):
                chunk = re.sub(r'\s*Assistant:\s*', assistant_prefix, chunk) + \
                        assistant_suffix
            else:
                pass

            if chunk != '':
                chunks.append(chunk)

        prompt = ''.join(chunks)
        responses = [chosen + assistant_suffix, rejected + assistant_suffix]
        i,j = data[prompt].num_generations(), data[prompt].num_generations() + 1

        data[prompt].prompt = prompt
        data[prompt].generations.extend(responses)
        data[prompt].pairs.append((i, j))
        data[prompt].sft_index = 0

        if only_helpful:
            data[prompt].dataset_name = 'hh_helpful'
        elif only_harmless:
            data[prompt].dataset_name = 'hh_harmless'
        else:
            data[prompt].dataset_name = 'hh'

        data[prompt].remove_extra_spaces()

    return data


def get_hh_helpful(
    split: str,
    human_prefix: str,
    human_suffix: str,
    assistant_prefix: str, 
    assistant_suffix: str
) -> Dataset:
    rank0_print(
        f'Loading helpful HH dataset ({split} split) from Huggingface...'
    )
    return get_hh(
        split,
        human_prefix,
        human_suffix,
        assistant_prefix,
        assistant_suffix,
        only_helpful=True,
    )


def get_hh_harmless(
    split: str,
    human_prefix: str,
    human_suffix: str,
    assistant_prefix: str,
    assistant_suffix: str
) -> Dataset:
    rank0_print(
        f'Loading harmless HH dataset ({split} split) from Huggingface...'
    )
    return get_hh(
        split,
        human_prefix,
        human_suffix,
        assistant_prefix,
        assistant_suffix,
        only_harmless=True,
    )


def get_oasst(
    split: str,
    human_prefix: str,
    human_suffix: str, 
    assistant_prefix: str,
    assistant_suffix: str
) -> Dataset:
    """
    Load the Open Assistant dataset from Huggingface and convert it into to a 
    Dataset. For this dataset, the SFT text is the preferred response.

    OASST is a dataset of ranked responses (not just pairwise), but since we 
    are working with losses that expect paired preferences, turn a ranking 
    (a, b, c, d, e) into pairwise preferences ((a,b), (b,c), (c,d), (d,e)).

    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the
                        recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice
                        and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the
                            recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended
                            choice and is set in config.yaml)

    Returns:
        A Dataset instance.
    """
    rank0_print(f'Loading OASST dataset ({split} split) from Huggingface...')
    dataset = datasets.load_dataset(
        'OpenAssistant/oasst1',
        split=('validation' if split == 'test' else 'train')
    )
    dataset = dataset.filter(lambda x: x['lang'] == 'en')

    message_indexed_df = pd.DataFrame(dataset).set_index('message_id')
    parent_indexed_df = pd.DataFrame(dataset).set_index('parent_id')

    def get_path_to_root(node: pd.Series):
        if node['parent_id'] is None:
            return [node]
        else:
            parent = message_indexed_df.loc[node['parent_id']]
            return [node] + get_path_to_root(parent)
    
    def turn_path_to_prompt(path: List[pd.Series]):
        prompt = []
        while path != []:
            node = path.pop() # earlier messages are at end of list
            prefix = assistant_prefix if node['role'] == 'assistant' else \
                     human_prefix
            suffix = assistant_suffix if node['role'] == 'assistant' else \
                     human_suffix
            prompt.append(prefix + node['text'] + suffix)
        
        prompt.append(assistant_prefix)
        return "".join(prompt)

    data = Dataset('OASST')

    for row in (
        tqdm.tqdm(dataset, desc='Processing OASST') if on_rank0() else dataset
    ):
        if row['rank'] == 0 or row['rank'] is None:
            continue

        try:
            sibling_df = parent_indexed_df.loc[row['parent_id']]
            next_best_sibling = \
                sibling_df[sibling_df['rank'] == (row['rank'] - 1)].iloc[0]
            path_to_root = get_path_to_root(
                message_indexed_df.loc[next_best_sibling['message_id']]
            )
        except KeyError:
            continue
        except IndexError:
            continue

        prompt = turn_path_to_prompt(path_to_root[1:])
        responses = [
            next_best_sibling['text'] + assistant_suffix,
            row['text'] + assistant_suffix
        ]
        i,j = data[prompt].num_generations(), data[prompt].num_generations() + 1

        data[prompt].prompt = prompt
        data[prompt].generations.extend(responses)
        data[prompt].pairs.append((i,j))
        data[prompt].scores.extend([next_best_sibling['rank'], row['rank']])
        data[prompt].dataset_name = 'oasst'
        data[prompt].remove_extra_spaces()
    
    return data


def get_ultrabin(
    split: str,
    human_prefix: str,
    human_suffix: str,
    assistant_prefix: str,
    assistant_suffix: str
) -> Dataset:
    """
    Load the Ultrafeedback (binarized) dataset from Huggingface and convert it
    into to a Dataset.
    For this dataset, the SFT text is the preferred response.

    Args:
        - split: one of 'test', 'train'
        - human_prefix: marks start of human turn ('<|user|>' is the
                        recommended choice and is set in config.yaml)
        - human_suffix: marks end of human turn ('' is the recommended choice
                        and is set in config.yaml)
        - assistant_prefix: marks start of human turn ('<|assistant|>' is the
                            recommended choice and is set in config.yaml)
        - assistant_suffix: marks end of human turn ('' is the recommended
                            choice and is set in config.yaml)

    Returns:
        A Dataset instance.
    """
    if split == 'train':
        split = 'train_prefs'
    elif split == 'test':
        split = 'test_prefs'
    else:
        raise ValueError()

    rank0_print(
        f'Loading Ultra Binarized dataset ({split} split) from Huggingface...'
    )
    dataset = datasets.load_dataset(
        'HuggingFaceH4/ultrafeedback_binarized', split=split
    )
    if on_rank0():
        dataset = tqdm.tqdm(dataset, desc='Processing Ultrachat Binarized')

    data = Dataset('ultrabin')

    for row in dataset:
        prompt = human_prefix + row['prompt'] + human_suffix + assistant_prefix
        responses = [
            row['chosen'][-1]['content'] + assistant_suffix,
            row['rejected'][-1]['content'] + assistant_suffix,
        ]

        i, j = data[prompt].num_generations(), \
               data[prompt].num_generations() + 1
        data[prompt].prompt = prompt
        data[prompt].generations.extend(responses)
        data[prompt].pairs.append((i, j))
        data[prompt].sft_index = 0
        data[prompt].dataset_name = data.name
        data[prompt].truncation_mode = 'keep_start'
        data[prompt].remove_extra_spaces()

    return data
