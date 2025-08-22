

import argparse
from typing import List, Tuple, cast
import os
import time
import inspect
from pathlib import Path
import re
import logging

import torch

from aime_api_worker_interface import APIWorkerInterface

import vllm
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages)
from vllm.inputs import TextPrompt, TokensPrompt
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.utils import is_list_of
from vllm.logger import init_logger
from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
from transformers import AutoTokenizer


DEFAULT_WORKER_JOB_TYPE = "llama3"
DEFAULT_WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d1"
VERSION = 3

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


LLM_FAMILIES = [
    'Llama',
    'Mixtral',
    'Qwen',
    'DeepSeek'
]

QUANTIZATIONS = [
    'int1',
    'int4',
    'int8',
    'int16',
    'int32',
    'int64',
    'fp4',
    'fp8',
    'fp16',
    'fp32',
    'bfloat16',
    'uint4',
    'uint8',
    'uint16',
    'uint32',
    'uint64'
]


class VllmWorker():
    def __init__(self):
        self.args = self.load_flags()
        self.tokenizer_workaround = self.get_tokenizer_workaround()
        self.api_worker = APIWorkerInterface(
            self.args.api_server, 
            self.args.job_type, 
            self.args.api_auth_key, 
            self.args.gpu_id, 
            gpu_name=torch.cuda.get_device_name(0), 
            num_gpus=self.args.tensor_parallel_size,
            worker_version=VERSION,
            exit_callback=self.exit_callback,
            model_label=self.args.model_label,
            model_quantization=self.args.model_quantization, 
            model_size=self.args.model_size, 
            model_family=self.args.model_family, 
            model_type=self.args.model_type,
            model_repo_name=Path(self.args.model).name,
            model_path=self.args.model,
            framework='VLLM',
            framework_version=vllm.version.__version__,
            pytorch_version=torch.version.__version__,
            auto_tokenize_inputs=True,
            trust_remote_code=self.args.trust_remote_code,
            use_fast_tokenizer=self.args.use_fast_tokenizer,
            max_batch_size=self.args.max_batch_size,
            max_context_length=self.args.max_model_len
        )
        self.progress_update_data = dict()
        self.last_progress_update = time.time()
        self.logger = self.get_logger()
        self.llm_engine = LLMEngine.from_engine_args(EngineArgs.from_cli_args(self.args))
        try:
            self.run_engine()
        except KeyboardInterrupt:
            self.logger.info('KeyboardInterrupt triggered. Initiating shutdown sequence...')
            self.api_worker.gracefully_exit()

    def get_tokenizer_workaround(self):
        model_name = Path(self.args.model).name
        if model_name == 'Mixtral-8x7B-Instruct-v0.1-FP8-hf':
            try:
                tokenizer_workaround_path = Path(self.args.model) / 'tokenizer_workaround'
                return AutoTokenizer.from_pretrained(tokenizer_workaround_path)
            except OSError:
                exit(f'Error: To use the model {model_name} your model weight folder needs a subfolder called "tokenizer_workaround" containing the files "config.json", "tokenizer_config.json" "tokenizer.json"  and "tokenizer.model" of the Original Mistral Repo.')


    def get_logger(self):
        level = logging.DEBUG if self.args.dev else logging.INFO
        logging.basicConfig(format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s", datefmt="%m-%d %H:%M:%S", level=level)
        return init_logger(__name__)


    def get_sampling_params(self, job_data):
        
        sampling_params_keys = inspect.signature(SamplingParams).parameters.keys()
        sampling_params = {key: job_data[key] for key in sampling_params_keys if key in job_data}
        sampling_params['max_tokens'] = job_data.get('max_gen_tokens')
        return SamplingParams(**sampling_params)


    def get_prompt_input_ids(self, job_data):
        prompt_input = job_data.get('prompt_input') or job_data.get('text')
        chat_context = job_data.get('chat_context')
        if chat_context:
            if not self.validate_chat_context(job_data.get('job_id'), chat_context):
                self.logger.warning('Wrong context shape')
                return
            if prompt_input:
                chat_context.append(
                    {
                        "role": "user", 
                        "content": prompt_input
                    }
                )
            return self.format_chat_context(chat_context)
        else:
            return self.llm_engine.get_tokenizer().encode(prompt_input)
        

    def format_chat_context(
        self,
        chat_context,
        chat_template = None,
        add_generation_prompt = True,
        tools = None
        ):

        if not chat_template:
            chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

        tokenizer = self.llm_engine.get_tokenizer()
        model_config = self.llm_engine.get_model_config()
        conversation, mm_data = parse_chat_messages(
            chat_context,
            model_config,
            tokenizer,
            content_format=None # Fix for vllm update to 0.6.6.post1, only needed for OpenAI
        )
        if self.tokenizer_workaround:
            input_ids = self.tokenizer_workaround.apply_chat_template(
                chat_context,
                tokenize=True,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                tools=tools
            )
        else:
            input_ids = tokenizer.apply_chat_template(
                conversation=conversation,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
                chat_template=chat_template,
                tokenize=True
            )
        return input_ids


    def validate_chat_context(self, job_id, chat_context):
        for item in chat_context:
            if not isinstance(item, dict) or not all(key in item for key in ("role", "content")):
                result = {
                    'error':  f'Dialog has invalid chat context format! Format should be [{{"role": "user/assistant/system", "content": "Message content"}}, ...] but is {chat_context}',
                }
                self.api_worker.send_job_results(result, job_id=job_id)
                return False
        return True


    def get_result(self, request_output):
        if request_output:
            input_length = len(request_output.prompt_token_ids)
            num_generated_tokens = len(request_output.outputs[0].token_ids)
            max_model_len = self.llm_engine.get_model_config().max_model_len
            result = {
                'num_generated_tokens': num_generated_tokens,
                'max_seq_len': max_model_len,
                'prompt_length': input_length,
                'arrival_time': request_output.metrics.arrival_time,
                'finished_time': time.time(),
                'current_context_length': input_length + num_generated_tokens,
                'pending_duration': request_output.metrics.time_in_queue,
                'metrics': {
                    'in_num_tokens': input_length,
                    'out_num_tokens': num_generated_tokens, 
                }
            }
            if request_output.metrics.first_token_time and request_output.metrics.first_scheduled_time:
                result['preprocessing_duration'] = request_output.metrics.first_token_time - request_output.metrics.first_scheduled_time
            if request_output.outputs[0].finish_reason == 'length' and not num_generated_tokens:
                result['error'] = f"The context length {len(request_output.prompt_token_ids)} is exceeding the maximum context length {max_model_len}"
            else:
                result['text'] = request_output.outputs[0].text
            return result

    def update_progress(self):
        now = time.time()
        if (now - self.last_progress_update) > (1.0 / self.args.progress_rate):
            self.last_progress_update = now
            progress_result_batch = list()
            job_id_batch = list()
            num_generated_tokens_batch = list()
            for job_id, progress_result in self.progress_update_data.items():
                progress_result_batch.append(progress_result)
                job_id_batch.append(job_id)
                num_generated_tokens_batch.append(progress_result.get('num_generated_tokens', 0))
            self.progress_update_data.clear()
            self.api_worker.send_batch_progress(
                num_generated_tokens_batch,
                progress_result_batch,
                job_batch_ids=job_id_batch,
                progress_error_callback=self.error_callback
            )


    def add_job_requests(self, job_batch_data):
        for job_data in job_batch_data:
            self.llm_engine.add_request(
                job_data.get('job_id'), 
                TokensPrompt(prompt_token_ids=job_data.get('chat_context') or job_data.get('prompt_input')),
                self.get_sampling_params(job_data)
            )

    def run_engine(self):
        job_request_generator = self.api_worker.job_request_generator(self.args.max_batch_size)
        for job_batch_data in job_request_generator:
            self.add_job_requests(job_batch_data)
            
            for request_output in self.llm_engine.step():
                result = self.get_result(request_output)
                if request_output.finished:
                    self.api_worker.send_job_results(
                        result,
                        job_id=request_output.request_id,
                        wait_for_response=False,
                        error_callback=self.error_callback
                    )
                    self.progress_update_data.pop(request_output.request_id, None)
                else:
                    self.progress_update_data[request_output.request_id] = result
            self.update_progress()
            for job_id in self.api_worker.get_canceled_job_ids():
                self.logger.info(f'Job {job_id} canceled')
                self.llm_engine.abort_request(job_id)


    def error_callback(self, response):
        self.logger.error(response)
        raise response


    def exit_callback(self):
        destroy_model_parallel()
        destroy_distributed_environment()
        del self.llm_engine
        torch.cuda.empty_cache()


    def load_flags(self):
        parser = FlexibleArgumentParser()
        parser.add_argument(
            "--max-batch-size", type=int, default=256,
            help="Maximum batch size"
        )
        parser.add_argument(
            "--job-type", type=str, default=DEFAULT_WORKER_JOB_TYPE,
            help="Worker job type for the API Server"
        )
        parser.add_argument(
            "--model-label", type=str,
            help="Model label to display in client. Default: Name of the directory given in --model"
        )
        parser.add_argument(
            "--model-size", type=str,
            help="Model size (number of parameters). E.g. '8B', '70B'"
        )
        parser.add_argument(
            "--model-family", type=str,
            help="Model family. E.g. 'Llama', 'Mixtral'"
        )
        parser.add_argument(
            "--model-quantization", choices=QUANTIZATIONS,
            help="Model quantization"
        )
        parser.add_argument(
            "--model-type", type=str, default="LLM",
            help="Model type e.g. 'LLM'."
        )
        parser.add_argument(
            "--api-server", type=str, required=True,
            help="Address of the API server"
        )
        parser.add_argument(
            "--gpu-id", type=int, default=0,
            help="ID of the GPU to be used"
        )
        parser.add_argument(
            "--api-auth-key", type=str , default=DEFAULT_WORKER_AUTH_KEY,
            help="API server worker auth key",
        )
        parser.add_argument(
            "--progress-rate", type=int , default=5,
            help="Progress updates per sec to the API Server.",
        )
        parser.add_argument(
            "--dev", action='store_true',
            help="Sets logger level to DEBUG",
        )
        parser.add_argument(
            "--use-fast-tokenizer", action='store_true',
            help="Use fast tokenizer in API Worker Interface",
        )
        parser.add_argument(
            "--context-length", type=int,
            help="ID of the GPU to be used"
        )

        parser = EngineArgs.add_cli_args(parser)
        args = parser.parse_args()
        if not args.enable_chunked_prefill:
            args.enable_chunked_prefill = False # In vllm by default True for max_model_len > 32K --> Worker crash with requests containing longer contexts
        args.model_label = args.model_label or Path(args.model).name
        args.model_size = args.model_size or self.extract_model_size(args)
        args.model_quantization = args.model_quantization or self.extract_quantization(args) or 'fp16'
        args.model_family = args.model_family or self.extract_family(args)
        if args.context_length:
            args.max_model_len = args.context_length
        return args


    def extract_model_size(self, args):
        match = re.search(r'(\d+B)', Path(args.model).name.upper())  # Matches arbitrary number of digits followed by 'B'
        return match.group(1) if match else None
        

    def extract_quantization(self, args):
        for quantization in QUANTIZATIONS:
            if quantization in Path(args.model).name.lower():
                return quantization

    def extract_family(self, args):
        for family in LLM_FAMILIES:
            if family.lower() in Path(args.model).name.lower():
                return family

def main():
    vllm_worker = VllmWorker()


if __name__ == "__main__":
    main()
