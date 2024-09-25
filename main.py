

import argparse
from typing import List, Tuple
import os
import time
import inspect
from multiprocessing.dummy import Pool

from aime_api_worker_interface import APIWorkerInterface

import torch
from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages)
from vllm.inputs import PromptInputs, TextPrompt, TokensPrompt
from vllm.inputs.parse import parse_and_batch_prompt
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.transformers_utils.tokenizer import MistralTokenizer
from vllm.utils import is_list_of

WORKER_JOB_TYPE = "llama3"
DEFAULT_WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d1"
VERSION = 0
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

class VllmWorker():
    def __init__(self):
        self.args = self.load_flags()
        self.pool = Pool()
        self.api_worker = APIWorkerInterface(
            self.args.api_server, 
            self.args.job_type, 
            self.args.api_auth_key, 
            self.args.gpu_id, 
            world_size=WORLD_SIZE, 
            rank=LOCAL_RANK, 
            gpu_name=torch.cuda.get_device_name(), 
            worker_version=VERSION
        )
        self.running_jobs = list()
        self.awaiting_job = False
        self.llm_engine = LLMEngine.from_engine_args(EngineArgs.from_cli_args(self.args))
        self.run_engine()
        

    def get_sampling_params(self, job_data):
        
        sampling_params_keys = inspect.signature(SamplingParams).parameters.keys()
        sampling_params = {key: job_data[key] for key in sampling_params_keys if key in job_data}
        sampling_params['max_tokens'] = job_data.get('max_gen_tokens')
        return SamplingParams(**sampling_params)


    def get_chat_context_prompt(self, job_data):
        prompt_input = job_data.get('prompt_input')
        if prompt_input is None:
            prompt_input = job_data.get('text')
        chat_context = job_data.get('chat_context')
        if chat_context:
            if not self.validate_chat_context(job_data.get('job_id'), chat_context):
                print('Wrong context shape')
                return
            if prompt_input:
                chat_context.append(
                    {
                        "role": "user", 
                        "content": prompt_input
                    }
                )
        return self.format_chat_context(chat_context)


    def update_worker_job_request(self, request_interval=2):
        if not self.awaiting_job:
            self.awaiting_job = True
            self.pool.apply_async(
                self.api_worker.job_batch_request,
                args=[self.args.max_jobs_per_job_request],
                callback=self.add_new_job,
                error_callback=self.job_request_error_callback
            )
        if not self.running_jobs:
            self.pool.apply_async(self.print_idle_string)
        while not self.running_jobs:
            pass


    def print_idle_string(self):
        dot_string = self.api_worker.dot_string_generator()
        while not self.running_jobs:
            print(f'\rWorker idling{next(dot_string)}', end='')
            time.sleep(1)

    def add_new_job(self, job_batch_data):
        self.awaiting_job = False
        self.running_jobs += job_batch_data
                    
        for batch_idx, job_data in enumerate(job_batch_data):
            prompt = self.get_chat_context_prompt(job_data)
            if prompt:
                self.llm_engine.add_request(
                    job_data.get('job_id'), 
                    prompt,
                    self.get_sampling_params(job_data)
                )

    def job_request_error_callback(self, response):
        self.awaiting_job = False
        print(response.json())


    def format_chat_context(
        self,
        chat_context,
        chat_template = None,
        add_generation_prompt = True,
        tools = None
        ):
        tokenizer = self.llm_engine.get_tokenizer_group(TokenizerGroup).tokenizer
        model_config = self.llm_engine.get_model_config()

        conversation, mm_data = parse_chat_messages(chat_context, model_config,
                                                    tokenizer)
        if isinstance(tokenizer, MistralTokenizer):
            prompt = apply_mistral_chat_template(
                tokenizer,
                messages=chat_context,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
            )
        else:
            prompt = apply_hf_chat_template(
                tokenizer,
                conversation=conversation,
                chat_template=chat_template,
                add_generation_prompt=add_generation_prompt,
                tools=tools,
            )

        if is_list_of(prompt, int):
            inputs = TokensPrompt(prompt_token_ids=prompt)
        else:
            inputs = TextPrompt(prompt=prompt)

        if mm_data is not None:
            inputs["multi_modal_data"] = mm_data

        return inputs


    def validate_chat_context(self, job_id, chat_context):
        for item in chat_context:
            if not isinstance(item, dict) or not all(key in item for key in ("role", "content")):
                job_data = self.get_job_data(job_id)
                result = {
                    'error':  f'Dialog has invalid chat context format! Format should be [{{"role": "user/assistant/system", "content": "Message content"}}, ...] but is {chat_context}',
                    'model_name': self.args.job_type
                }
                self.api_worker.send_job_results(result, job_data=job_data)
                self.running_jobs.remove(job_data)
                return False
        return True


    def get_job_data(self, job_id):
        for job_data in self.running_jobs:
            if job_data.get('job_id') == job_id:
                return job_data

    def get_result(self, request_output):
        num_generated_tokens = len(request_output.outputs[0].token_ids)
        return {
            'text': request_output.outputs[0].text,
            'model_name': self.args.job_type,
            'num_generated_tokens': num_generated_tokens,
            'max_seq_len': self.args.max_seq_len_to_capture,
            'current_context_length': len(request_output.prompt_token_ids) + num_generated_tokens
        }


    def run_engine(self):
        while True:
            self.update_worker_job_request()

            request_outputs = self.llm_engine.step()
            progress_result_batch = list()
            job_batch_data = list()
            num_generated_tokens_batch = list()
            for request_output in request_outputs:
                result = self.get_result(request_output)
                job_data = self.get_job_data(request_output.request_id)

                if request_output.finished:
                    self.api_worker.send_job_results(result, job_data)
                    self.running_jobs.remove(job_data)
                else:
                    num_generated_tokens_batch.append(result.get('num_generated_tokens'))
                    progress_result_batch.append(result)
                    job_batch_data.append(job_data)

            if progress_result_batch:
                self.api_worker.send_batch_progress(
                    num_generated_tokens_batch,
                    progress_result_batch,
                    self.progress_callback,
                    self.progress_error_callback,
                    job_batch_data
                )


    def progress_callback(self, response):
        for job_reponse in response.json():
            if job_reponse.get('canceled'):
                print('CANCELED')
                self.llm_engine.abort_request(job_reponse.get('job_id'))

    def progress_error_callback(self, response):
        print(response.json())

    def load_flags(self):
        parser = FlexibleArgumentParser()
        parser.add_argument(
            "--max-seq-len", type=int, default=8192, required=False,
            help="Maximum sequence length",
        )
        parser.add_argument(
            "--max-jobs-per-job-request", type=int, default=512, required=False,
            help="Maximum batch size",
        )
        parser.add_argument(
            "--max-jobs-per-job-results", type=int, default=8, required=False,
            help="Maximum batch size",
        )
        parser.add_argument(
            "--job-type", type=str, required=False,
            help="Worker job type for the API Server"
        )
        parser.add_argument(
            "--api-server", type=str, required=True,
            help="Address of the API server"
        )
        parser.add_argument(
            "--gpu-id", type=int, default=0, required=False,
            help="ID of the GPU to be used"
        )
        parser.add_argument(
            "--api-auth-key", type=str , default=DEFAULT_WORKER_AUTH_KEY, required=False, 
            help="API server worker auth key",
        )
        parser = EngineArgs.add_cli_args(parser)
        return parser.parse_args()

def main():
    vllm_worker = VllmWorker()


if __name__ == "__main__":
    main()
