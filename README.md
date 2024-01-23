# Alignment_with_Huggingface

> This repo draws from the excellently written [HALOs repo](https://github.com/ContextualAI/HALOs) and [DPO repo](https://github.com/eric-mitchell/direct-preference-optimization). We have preserved many design choices from the orignal.

This repo is to provide a generali framework for aligning large language models(LLMs) with the Transformers and Datasets from Huggingface.
Unlike the [TRL framework](https://huggingface.co/docs/trl/index) from Huggingface, we hereby incorporate the following features:

- Support for modifying the weights of training samples.
- Support for generating responses from the LLM policy.
- Support for getting feedback with online reward model or language model.

A diagram of the data flow from a high level is shown below:

![Data flow.](https://github.com/Shawn-Guo-CN/Alignment_with_Huggingface/blob/main/docs/figs/sys_data_flow.png)

## Components

In the following, we introduce the major components of the framework, but not by the order of the data flow.

### 1. BatchFactory

The `BatchFactory` is a class that generates batches of data for training. It takes a `Dataset` object as input and generates train/test batches to the `Trainer` of the following format:

```python
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
```

Note that the above items are not necessarily all included in the batch. Below is a diagram of data in `BatchFactory`.

![Data flow in BatchFactory.](https://github.com/Shawn-Guo-CN/Alignment_with_Huggingface/blob/main/docs/figs/data_flow.png)

We hereby list the learning tasks and the corresponding batch items as well as the source of them:

- **Supervised fine-tuning**: in such case, there are only `prompt` and `generation1` in the batch. Moreover, the `generation1_reward` is `None` and the `generation1_weight` is always 1.0.
- **Reward modelling in RLHF**: in such case, `prompt`
- **Offline Pointwise preference learning**:
- **Offline Pairwise preference learning**:

### 2. AnnotatingLM

### 3. Trainer

### 4. Loss

### 5. SampleReweighter

