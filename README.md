# AIME API vLLM Worker

 [vLLM](https://docs.vllm.ai) worker for [AIME API Server](https://github.com/aime-team/aime-api-server) to scalable serve large language models on CUDA or ROCM with various quantizations and large context lengths.

Supported models:

- Llama / Qwen / Mistral /  Mixtral / DeepSeek / Bloom / Falcon / Gemma / GPT-NeoX / InternLM / Mamba / Nemotron / Phi / Starcoder

For a full list of current supported models see [here](https://docs.vllm.ai/en/latest/models/supported_models.html)


## How to setup a AIME API vLLM worker with MLC


```bash
mlc create vllm Pytorch 2.5.1 
mlc open vllm

git clone https://github.com/aime-labs/aime-api_vllm.git

cd vllm_worker

pip3 install -r requirements.txt
```

### Download LLM models

The installed AIME worker interface Pip provides the 'awi' command to download model weights:

```bash
awi download-weights {model name} -o /path/to/your/model/weights/
```

e.g. to download LLama3.1 70B fp8 Instruct model:

```bash
awi download-weights neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 -o /path/to/your/model/weights/
```

## How to start a AIME API vLLM worker

### Running an LLM

```bash
python main.py --api_server http://127.0.0.1:7777 --api_auth_key b07e305b50505ca2b3284b4ae5f65d1 --model /path/to/your/model/weights/your_llm/ --job_type job_type_name --max_batch_size 8 --tensor-parallel-size 2
```

### Running Llama 3.3

Starting Llama 3.3 70B worker on 2x RTX 6000 Ada 48GB GPUs:

```bash
python main.py --api_server http://127.0.0.1:7777 --api_auth_key b07e305b50505ca2b3284b4ae5f65d1 --model /path/to/your/model/weights/Llama-3.3-70B-Instruct --job_type llama3_3 --max_batch_size 8 --tensor-parallel-size 2
```


### Running Llama 3.1

Starting Llama 3.1 70B worker with fp8 quantization on 2x RTX 6000 Ada 48GB GPUs:

```bash
python main.py --api_server http://127.0.0.1:7777 --api_auth_key b07e305b50505ca2b3284b4ae5f65d1 --model /path/to/your/model/weights/Llama-3.1-70B-Instruct-fp8 --job_type llama3_1 --max_batch_size 8 --tensor-parallel-size 2
```

### Running Mixtral 8x7B

Starting Mixtral 8x7B worker with fp8 quantization on 2x RTX 6000 Ada 48GB GPUs:

```bash
python3 main.py --api_server http://127.0.0.1:7777 --api_auth_key b07e305b50505ca2b3284b4ae5f65d1 --model /path/to/your/model/weights/Mixtral-8x7B-Instruct-v0.1-hf --job-type mixtral --max_batch_size 8 --tensor-parallel-size 2
```

Copyright (c) AIME GmbH and affiliates. Find more info at https://www.aime.info/api

This software may be used and distributed according to the terms of the MIT LICENSE
