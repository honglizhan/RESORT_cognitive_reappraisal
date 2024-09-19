import os
import json
import pandas as pd
from pipeline_components.chat_agents import (OpenAIAgent, LLaMA2ChatAgent, MistralChatAgent)
from transformers import (AutoTokenizer, AutoModelForCausalLM)
from openai import (OpenAI, AzureOpenAI)

def read_jsonl(jsonl_path):
    """ returns a list of dictionaries """
    with open(jsonl_path) as f:
        lst = f.readlines()
        lst = [
            json.loads(line.strip()) for line in lst if line.strip()
        ]
    return lst

def filter_inputs(input_lst, output_path, input_field: str = "Reddit ID"):
    filtered_input_lst = input_lst[:]

    ## ---- filter out already-generated questions ---
    if os.path.isfile(output_path):
        output_file = read_jsonl(output_path)
        output_lst = [dct[input_field] for dct in output_file]
        for input in input_lst:
            if input[input_field] in output_lst:
                filtered_input_lst.remove(input)
    return filtered_input_lst

def setup_chat_agent(FLAGS):
    my_system_prompt = """Respond with a response in the format requested by the user. Do not acknowledge my request with "sure" or in any other way besides going straight to the answer."""

    if "Llama-2" in FLAGS.model:
        ChatAgent = LLaMA2ChatAgent(
            FLAGS,
            tokenizer=AutoTokenizer.from_pretrained(FLAGS.model, token=os.getenv("HF_AUTH_KEY")),
            model=AutoModelForCausalLM.from_pretrained(FLAGS.model, device_map='auto', torch_dtype=torch.float16, token=os.getenv("HF_AUTH_KEY")),
            new_system_prompt=my_system_prompt)

    elif "gpt" in FLAGS.model:
        if FLAGS.use_azure == True:
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2023-05-15")
        else:
            client = OpenAI()

        ChatAgent = OpenAIAgent(
            FLAGS,
            client=client,
            new_system_prompt=my_system_prompt)

    elif "mistral" in FLAGS.model:
        ChatAgent = MistralChatAgent(
            FLAGS,
            tokenizer=AutoTokenizer.from_pretrained(FLAGS.model),
            model=AutoModelForCausalLM.from_pretrained(FLAGS.model, device_map='auto', torch_dtype=torch.float16),
            new_system_prompt=my_system_prompt)

    else:
        raise ValueError("Model name not supported!")

    return ChatAgent


def get_prompts(path_to_appraisal_questions, path_to_reappraisal_guidance):
    dimensions_df = pd.DataFrame(index = range(1, 25))
    dim_files = [path_to_appraisal_questions, path_to_reappraisal_guidance]

    for dim_file_name in dim_files:
        with open(dim_file_name) as dim_file:
            raw_txt = dim_file.read()
            if 'dim_name' not in dimensions_df.columns:
                dimensions_df['dim_name'] = pd.Series(raw_txt.split('\n'), index = range(1, 25)).apply(lambda line: line[:line.find('=') - 1])
            dimensions_df[dim_file_name[10:-4]] = pd.Series(raw_txt.split('\n'), index = range(1, 25)).apply(lambda line: line[line.find('=') + 1:])
    #display(dimensions_df)
    return dimensions_df