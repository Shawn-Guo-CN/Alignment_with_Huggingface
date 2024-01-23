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

In the following, we introduce the major components of the framework.

