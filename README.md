# AIME API vLLM Worker

 [vLLM](https://docs.vllm.ai) Worker for [AIME API Server](https://github.com/aime-team/aime-api-server) to scalable serve large language models on CUDA or ROCM with various quantizations.

Supported models:

- Llama / Mistral /  Mixtral / Bloom / Falcon / Gemma / GPT-NeoX / InternLM / Mamba / Nemotron / Phi / Qwen / Starcoder

For a full list of current supported models see [here](https://docs.vllm.ai/en/latest/models/supported_models.html)


## How to setup a AIME API vLLM worker with MLC

```bash
mlc-create vllm Pytorch 2.4.0 -d /path/to/your/model/weights
mlc-open vllm
cd /desired/location/of/this/repo
git clone https://github.com/aime-labs/aime-api_vllm
pip3 install -r /desired/location/of/this/repo/aime-api_vllm/requirements_pinned.txt
```

### Download LLM models

python download_weights.py -m {model name} -o {directory to store model}

e.g. to download LLama3.3 70B Instruct model:

```bash
python download_weights.py -m meta-llama/Llama-3.3-70B-Instruct -o /path/to/your/model/weights/
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
