

import argparse
from typing import List, Tuple
import os
import time
import inspect
import sys

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

DEFAULT_WORKER_JOB_TYPE = "llama3"
DEFAULT_WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d1"
VERSION = 0

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"



class VllmWorker():
    def __init__(self):
        self.args = self.load_flags()
        self.api_worker = APIWorkerInterface(
            self.args.api_server, 
            self.args.job_type, 
            self.args.api_auth_key, 
            self.args.gpu_id, 
            gpu_name=torch.cuda.get_device_name(), 
            worker_version=VERSION
        )
        
        #keyboard.add_hotkey('q', self.exit_worker)

        self.llm_engine = LLMEngine.from_engine_args(EngineArgs.from_cli_args(self.args))

        self.run_engine()




    def get_sampling_params(self, job_data):
        
        sampling_params_keys = inspect.signature(SamplingParams).parameters.keys()
        sampling_params = {key: job_data[key] for key in sampling_params_keys if key in job_data}
        sampling_params['max_tokens'] = job_data.get('max_gen_tokens')
        return SamplingParams(**sampling_params)


    def get_chat_context_prompt(self, job_data):
        print(job_data)
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
        else:
            return prompt_input
        


    def format_chat_context(
        self,
        chat_context,
        chat_template = None,
        add_generation_prompt = True,
        tools = None
        ):
        tokenizer = self.llm_engine.get_tokenizer_group(TokenizerGroup).tokenizer
        model_config = self.llm_engine.get_model_config()
        conversation, mm_data = parse_chat_messages(
            chat_context,
            model_config,
            tokenizer
        )
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
                result = {
                    'error':  f'Dialog has invalid chat context format! Format should be [{{"role": "user/assistant/system", "content": "Message content"}}, ...] but is {chat_context}',
                    'model_name': self.args.job_type
                }
                self.api_worker.send_job_results(result, job_id=job_id)
                return False
        return True


    def get_result(self, request_output):
        num_generated_tokens = len(request_output.outputs[0].token_ids)
        return {
            'text': request_output.outputs[0].text,
            'model_name': self.args.job_type,
            'num_generated_tokens': num_generated_tokens,
            'max_seq_len': self.args.max_seq_len_to_capture,
            'current_context_length': len(request_output.prompt_token_ids) + num_generated_tokens
        }

    def add_job_requests(self, job_batch_data):
        for job_data in job_batch_data:
            prompt = self.get_chat_context_prompt(job_data)
            if prompt:
                self.llm_engine.add_request(
                    job_data.get('job_id'), 
                    prompt,
                    self.get_sampling_params(job_data)
                )


    def run_engine(self):
        job_request_generator = self.api_worker.job_request_generator(self.args.max_batch_size)
        for job_batch_data in job_request_generator:
            self.add_job_requests(job_batch_data)
            
            num_generated_tokens_batch = list()
            progress_result_batch = list()
            job_id_batch = list()
            for request_output in self.llm_engine.step():
                result = self.get_result(request_output)
                if request_output.finished:
                    self.api_worker.send_job_results(
                        result,
                        job_id=request_output.request_id,
                        wait_for_response=False,
                        error_callback=self.error_callback
                        )
                else:
                    num_generated_tokens_batch.append(result.get('num_generated_tokens'))
                    progress_result_batch.append(result)
                    job_id_batch.append(request_output.request_id)

            if progress_result_batch:
                self.api_worker.send_batch_progress(
                    num_generated_tokens_batch,
                    progress_result_batch,
                    job_batch_ids=job_id_batch,
                    progress_error_callback=self.error_callback
                )
            for job_id in self.api_worker.get_canceled_job_ids():
                print(f'Job {job_id} canceled')
                self.llm_engine.abort_request(job_id)

    def error_callback(self, response):
        print(response.json())

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
        parser = EngineArgs.add_cli_args(parser)
        return parser.parse_args()

def main():
    vllm_worker = VllmWorker()


if __name__ == "__main__":
    main()
