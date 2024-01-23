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

Note that the above items are not necessarily all included in the batch. Below is a diagram of data in `BatchFactory`. Note that the final output batches $\mathbb{B}$ are of 4 different types in the diagram, although they share the same format as above in practice.

![Data flow in BatchFactory.](https://github.com/Shawn-Guo-CN/Alignment_with_Huggingface/blob/main/docs/figs/data_flow.png)

We hereby list the learning tasks and the corresponding batch items as well as the source of them:

- **Supervised fine-tuning**: only `prompt` and `generation1` in the batch. Moreover, the `generation1_reward` is `None` and the `generation1_weight` is always 1.0.
- **Reward modelling in RLHF**: `prompt`, `generation1`, and `generation2` are all included. However, the `generation1_reward` and `generation2_reward` are both `None`. The `generation1_weight` and `generation2_weight` are always 1.0.
- **Offline Pointwise preference learning**: only `prompt` and `generation1` are included. The `generation1_reward` is 1.0 if the `generation1` is a desired response, otherwise 0.0 to indicate that `generation1` is undesired. The `generation1_weight` is always 1.0. (Check out [HALOs repo](https://github.com/ContextualAI/HALOs) for the details of training models with pointwise desirable/undesirable feedback.) 
> Both the generations and annotations are from the precollected and fixed `Dataset`, thus this is an OFFLINE learning setup.
- **Online Pointwise preference learning**: same to the above offline pointwise preference learning, except that the `generation1` is sampled from the LLM policy being training and the `generation1_reward` is obtained from the online *Annotator*. 
> The generations are from the LLM policy being trained and the feedbacks from online annotator, thus this is an ONLINE learning setup.*
- **Offline Pairwise preference learning**: `prompt`, `generation1`, and `generation2` are all included. The `generation1_reward` is 1.0 to indicate that `generation1` is preferred over `generation2`. The `generation1_weight` and `generation2_weight` are always 1.0.
> Like the offline pointwise preference learning setup, the generations and annotations are from the precollected and fixed `Dataset`, thus this is an OFFLINE learning setup.
- **Online Pairwise preference learning**: same to the above offline pairwise preference learning, except that the `generation1` and `generation2` are sampled from the LLM policy being training. `generation1_reward` is 1.0 if `generation1` is preferred over `generation2` by the *online annotator*, otherwise 0.0.
> The generations are from the LLM policy being trained and the feedbacks from online annotator, thus this is an ONLINE learning setup.*

### 2. Dataset



### 3. Annotator

### 4. Trainer

### 5. Loss

### 6. SampleReweighter

