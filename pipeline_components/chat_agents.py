#==============#
# Importations #
#==============#
import time
import textwrap
from termcolor import colored

# LLaMA2
import torch
import transformers
from transformers import (AutoTokenizer, AutoModelForCausalLM)
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate, LLMChain


#=================#
# Text Generation #
#=================#

class OpenAIAgent:
    def __init__(
        self,
        FLAGS,
        client,
        new_system_prompt: str = None,
    ) -> None:
        self.FLAGS = FLAGS
        self.client = client
        self.DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
        self.system_prompt = new_system_prompt if new_system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT
        self.chat_history = [{"role": "system", "content": self.system_prompt}]

    def reset(self) -> None:
        self.chat_history = [{"role": "system", "content": self.system_prompt}]

    def chat(self, message: str) -> str:
        self.chat_history.append({"role": "user", "content": message})
        for role in self.chat_history:
            for key, value in role.items():
                print(colored(f'{key}:', "red"), colored(f'{value}', "green"))
        while True:
            try:
                ai_message = self.client.chat.completions.create(
                    model = self.FLAGS.model,
                    messages = self.chat_history,
                    temperature = self.FLAGS.temperature,
                    max_tokens = self.FLAGS.max_tokens,
                    seed = self.FLAGS.seed,
                ).choices[0].message.content

                self.chat_history.append({"role": "assistant", "content": ai_message})

                # If no exception was raised, break out of the loop
                break

            except Exception as e:
                if str(e) == """Error code: 400 - {'error': {'message': "The response was filtered due to the prompt triggering Azure OpenAI's content management policy. Please modify your prompt and retry. To learn more about our content filtering policies please read our documentation: https://go.microsoft.com/fwlink/?linkid=2198766", 'type': None, 'param': 'prompt', 'code': 'content_filter', 'status': 400}}""":
                    ai_message = e
                    print (e)
                    break
                else:
                    sleep_time = 20
                    print(f"OpenAI API Error, retrying in {sleep_time} seconds: {e}")
                    time.sleep(sleep_time)

        return ai_message


class LLaMA2ChatAgent:
    def __init__(
        self,
        FLAGS,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        chat_history: ConversationBufferMemory = ConversationBufferMemory(memory_key="chat_history"),
        new_system_prompt: str = None,
    ) -> None:
        self.FLAGS = FLAGS
        self.tokenizer = tokenizer
        self.model = model
        self.llm = HuggingFacePipeline(
            pipeline = transformers.pipeline(
                task="text-generation",
                model=self.model, tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.FLAGS.max_tokens,
            ),
            model_kwargs = {'temperature': self.FLAGS.temperature}
        )
        self.chat_history = chat_history
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        self.new_system_prompt = new_system_prompt
        self.DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

    def reset(self) -> None:
        self.chat_history.clear()

    def get_prompt(self, instruction: str, new_system_prompt: str):
        SYSTEM_PROMPT = self.B_SYS + new_system_prompt + self.E_SYS
        prompt_template =  self.B_INST + SYSTEM_PROMPT + instruction + self.E_INST
        return prompt_template

    def cut_off_text(self, text: str, prompt: str):
        cutoff_phrase = prompt
        index = text.find(cutoff_phrase)
        if index != -1:
            return text[:index]
        else:
            return text

    def remove_substring(self, string: str, substring: str):
        return string.replace(substring, "")

    def generate(self, text: str):
        prompt = self.get_prompt(instruction=text, new_system_prompt=self.DEFAULT_SYSTEM_PROMPT if self.new_system_prompt is None else self.new_system_prompt)
        print (colored (prompt, "green"))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.FLAGS.max_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=self.FLAGS.temperature,
                do_sample=True
            )
            final_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            final_outputs = self.cut_off_text(final_outputs, '</s>')
            final_outputs = self.remove_substring(final_outputs, prompt)

        return final_outputs#, outputs

    def parse_text(text):
        wrapped_text = textwrap.fill(text, width=100)
        print(wrapped_text +'\n\n')
        # return assistant_text

    def chat(self, message: str) -> str:
        template = self.get_prompt(
            instruction="Chat History:\n\n{chat_history} \n\nUser: {user_input}",
            new_system_prompt=self.DEFAULT_SYSTEM_PROMPT if self.new_system_prompt is None else self.new_system_prompt
        )
        prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True,
            memory=self.chat_history,
        )
        return llm_chain.predict(user_input=message)


class MistralChatAgent:
    def __init__(
        self,
        FLAGS,
        tokenizer: AutoTokenizer,
        model: AutoModelForCausalLM,
        chat_history: ConversationBufferMemory = ConversationBufferMemory(memory_key="chat_history"),
        new_system_prompt: str = None,
    ) -> None:
        self.FLAGS = FLAGS
        self.tokenizer = tokenizer
        self.model = model
        self.llm = HuggingFacePipeline(
            pipeline = transformers.pipeline(
                task="text-generation",
                model=self.model, tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=self.FLAGS.max_tokens,
            ),
            model_kwargs = {'temperature': self.FLAGS.temperature}
        )
        self.chat_history = chat_history
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.new_system_prompt = new_system_prompt
        self.DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


    def reset(self) -> None:
        self.chat_history.clear()

    def get_prompt(self, instruction: str, new_system_prompt: str):
        SYSTEM_PROMPT = new_system_prompt + "\n\n"
        prompt_template = self.B_INST + SYSTEM_PROMPT + instruction + self.E_INST
        return prompt_template

    def cut_off_text(self, text: str, prompt: str):
        cutoff_phrase = prompt
        index = text.find(cutoff_phrase)
        if index != -1:
            return text[:index]
        else:
            return text

    def remove_substring(self, string: str, substring: str):
        return string.replace(substring, "")

    def generate(self, text: str):
        prompt = self.get_prompt(instruction=text, new_system_prompt=self.DEFAULT_SYSTEM_PROMPT if self.new_system_prompt is None else self.new_system_prompt)
        print (colored (prompt, "green"))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.FLAGS.max_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=self.FLAGS.temperature,
                do_sample=True
            )
            final_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            final_outputs = self.cut_off_text(final_outputs, '</s>')
            final_outputs = self.remove_substring(final_outputs, prompt)

        return final_outputs#, outputs

    def chat(self, message: str) -> str:
        template = self.get_prompt(
            instruction="Chat History:\n\n{chat_history} \n\nUser: {user_input}",
            new_system_prompt=self.DEFAULT_SYSTEM_PROMPT if self.new_system_prompt is None else self.new_system_prompt
        )
        prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=template)
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True,
            memory=self.chat_history,
        )
        return llm_chain.predict(user_input=message)