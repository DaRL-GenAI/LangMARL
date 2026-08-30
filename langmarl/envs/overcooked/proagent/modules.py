from openai import OpenAI
from rich import print as rprint
import time
import os
from typing import Union
from .utils import convert_messages_to_prompt, retry_with_exponential_backoff
from .llm_config import (
    is_openrouter, get_api_key, get_base_url, get_extra_headers, create_openai_client
)

# Refer to https://platform.openai.com/docs/models/overview
TOKEN_LIMIT_TABLE = {
    "text-davinci-003": 4080,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
}


class Module(object):
    """
    This module is responsible for communicating with GPTs.
    """
    def __init__(self,
                 role_messages,
                 model="gpt-3.5-turbo-0301",
                 retrieval_method="recent_k",
                 K=3):
        '''
        args:
        use_similarity:
        dia_num: the num of dia use need retrival from dialog history
        '''

        self.model = model
        self.retrieval_method = retrieval_method
        self.K = K

        self.chat_model = True if "gpt" in self.model else False
        self.instruction_head_list = role_messages
        self.dialog_history_list = []
        self.current_user_message = None
        self.cache_list = []
        self.client = None  # Will be initialized when api_key is set

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)
    
    def get_cache(self)->list:
        if self.retrieval_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K:]
            else: 
                return []
        else:
            return [] 
           
    @property
    def query_messages(self)->list:
        return self.instruction_head_list + self.cache_list + [self.current_user_message]

    @property
    def prompt_token_length(self) -> int:
        """Estimate the total token length of the current prompt."""
        total_text = ""
        for msg in self.query_messages:
            if isinstance(msg, dict):
                total_text += msg.get("content", "")
        return len(total_text) // 4  # rough estimate: ~4 chars per token
    
    @retry_with_exponential_backoff
    def query(self, key, stop=None, temperature=0.0, debug_mode = 'Y', trace = True):
        # Initialize OpenAI client with the provided API key and timeout
        # Support OpenRouter via llm_config.json
        if self.client is None:
            if is_openrouter():
                self.client = create_openai_client(timeout=60.0)
            else:
                self.client = OpenAI(api_key=key, timeout=60.0)  # 60 second timeout

        rec = self.K
        if trace == True:
            self.K = 0

        self.cache_list = self.get_cache()
        messages = self.query_messages
        if trace == False:
            messages[len(messages) - 1]['content'] += " Based on the failure explanation and scene description, analyze and plan again."
        self.K = rec

        '''
        # ========== 打印：LLM 原始输入 ==========
        print("\n" + "="*80)
        print("🔵 [LLM INPUT] Raw Messages to LLM")
        print("="*80)
        import json
        print(json.dumps(messages, indent=2, ensure_ascii=False))
        print("="*80 + "\n")
        # ==========================================
        '''

        response = None
        get_response = False
        retry_count = 0
        max_retries = 5

        while not get_response:
            if retry_count >= max_retries:
                rprint(f"[red][ERROR][/red]: Query GPT failed for over {max_retries} times!")
                return {}
            try:
                if self.model in ['text-davinci-003']:
                    prompt = convert_messages_to_prompt(messages)
                    response = self.client.completions.create(
                        model=self.model,
                        prompt=prompt,
                        stop=stop,
                        temperature=temperature,
                        max_tokens = 256
                    )
                    time.sleep(10)
                elif 'gpt' in self.model or is_openrouter():
                    # Build extra headers for OpenRouter
                    extra_headers = get_extra_headers() if is_openrouter() else None

                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stop=stop,
                        temperature=temperature,
                        max_tokens = 256,
                        extra_headers=extra_headers if extra_headers else None
                    )
                    time.sleep(10)
                else:
                    raise Exception(f"Model {self.model} not supported.")

                # Validate response is not None and has expected structure
                if response is None:
                    raise ValueError("API returned None response")

                if hasattr(response, 'choices') and response.choices:
                    if response.choices[0].message is None or response.choices[0].message.content is None:
                        raise ValueError("API returned empty message content")
                elif hasattr(response, 'choices') and not response.choices:
                    raise ValueError("API returned empty choices list")

                get_response = True

            except Exception as e:
                retry_count += 1
                rprint(f"[red][OPENAI ERROR][/red] (attempt {retry_count}/{max_retries}):", e)
                time.sleep(20)

        return self.parse_response(response)

    def parse_response(self, response):
        if self.model == 'claude':
            parsed = response
        elif self.model in ['text-davinci-003']:
            parsed = response.choices[0].text
        elif self.model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0301', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-0314', 'gpt-4o-mini'] or is_openrouter():
            # OpenRouter uses same response format as OpenAI chat completions
            parsed = response.choices[0].message.content

        '''
        # ========== 打印：解析后的内容 ==========
        print("\n" + "="*80)
        print("🟡 [LLM PARSED] Parsed Response Content")
        print("="*80)
        print(parsed)
        print("="*80 + "\n")
        # ==========================================
        '''

        return parsed

    def restrict_dialogue(self):
        """
        The limit on token length for gpt-3.5-turbo-0301 is 4096.
        If token length exceeds the limit, we will remove the oldest messages.
        """
        limit = TOKEN_LIMIT_TABLE[self.model]
        print(f'Current token: {self.prompt_token_length}')
        while self.prompt_token_length >= limit:
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            self.cache_list.pop(0)
            print(f'Update token: {self.prompt_token_length}')
        
    def reset(self):
        self.dialog_history_list = []

