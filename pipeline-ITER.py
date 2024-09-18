#==============#
# Importations #
#==============#
import os
import json
import torch
import random
import numpy as np
from absl import app, flags
from tqdm import tqdm
from transformers import set_seed
from pipeline_components import prompt_loading
import utils

#=============#
# Define Args #
#=============#
FLAGS = flags.FLAGS

### ------ Define the LLM ------
flags.DEFINE_string("model", "meta-llama/Llama-2-13b-chat-hf", "Specify the LLM.")

### ------ Hyper-parameters for LLMs ------
flags.DEFINE_float("temperature", 0.1, "The value used to modulate the next token probabilities.")
flags.DEFINE_integer("max_tokens", 512, "Setting max tokens to generate.")
flags.DEFINE_integer("seed", 36, "Setting seed for reproducible outputs.")

### ------ Loading input/output data path ------
flags.DEFINE_string("input_data_path", "./source_data/r_anger.jsonl", "")
flags.DEFINE_string("output_path", "./model_outputs/iterative_guided_refinement", "")

### ------ Loading prompt path ------
flags.DEFINE_string("path_to_appraisal_questions", "./prompts/appraisal_questions.txt", "File with all appraisal questions.")
flags.DEFINE_string("path_to_reappraisal_guidance", "./prompts/reappraisal_guidance.txt", "File with all re-appraisal guidance.")

### ------ Define experiment mode ------
flags.DEFINE_enum("experiment_mode", "self-refine", ["self-refine", "+appr", "+cons", "+appr +cons"], "Specify experiment mode.")
flags.DEFINE_list("dimensions", [1, 4, 6, 7, 8, 23], "Specify the dimensions to look at.")

### ------ For GPT models, choose to use Azure or openai keys ------
flags.DEFINE_boolean("use_azure", True, "Use Azure OpenAI API instead of OpenAI API.")




def main(argv):
    #==========#
    # Set seed #
    #==========#
    set_seed(FLAGS.seed)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.cuda.manual_seed_all(FLAGS.seed)

    ChatAgent = utils.setup_chat_agent(FLAGS)

    if FLAGS.model == "gpt-4-1106-preview":
        model_name = "gpt-4-turbo"
    else:
        model_name = FLAGS.model.split('/')[-1]

    dimensions_df = utils.get_promts(FLAGS.path_to_appraisal_questions, FLAGS.path_to_reappraisal_guidance)

    path = f"""{FLAGS.output_path}/{FLAGS.experiment_mode}/{FLAGS.input_data_path.split('/')[-1].split('.')[0]}"""
    if not os.path.exists(path):
        os.makedirs(path)
    output_jsonl_file_path = f"{path}/{model_name}.jsonl"

    #==================#
    # Elicit responses #
    #==================#
    eval_set = utils.read_jsonl(FLAGS.input_data_path)
    eval_set = utils.filter_inputs(eval_set, output_jsonl_file_path, input_field="Reddit ID")

    for row in tqdm(eval_set):
        prev_output = ""
        if FLAGS.experiment_mode == "self-refine":
            for i, dim in enumerate(FLAGS.dimensions):
                if (prev_output == ""):
                    baseline_prompt = "Please help the narrator of the text reappraise the situation. Your response should be concise and brief."
                    step2_output = ChatAgent.chat(f"""[Text] {row['Reddit Post']}\n\n[Question] {baseline_prompt}""")
                else:
                    baseline_prompt = prompt_loading.build_iterative_step_baseline(
                        post = row['Reddit Post'], prev_step = prev_output)
                    step2_output = ChatAgent.chat(baseline_prompt) 
                # print(step2_output)
                ChatAgent.reset()

                row[f"intermediate_prompt_step_{i}"] = baseline_prompt
                row[f"reappraisal_output_step_{i}"] = str(step2_output)
                prev_output = str(step2_output)
                
        elif FLAGS.experiment_mode == "+cons":
            for dim in FLAGS.dimensions:
                guidance = dimensions_df.reappraisal_guidance[dim]
                if (prev_output == ""):
                    baseline_prompt = f"""Please help the narrator of the text reappraise the situation. {guidance} Your response should be concise and brief."""
                    step2_output = ChatAgent.chat(f"""[Text] {row['Reddit Post']}\n\n[Question] {baseline_prompt}""")
                else:
                    baseline_prompt = prompt_loading.build_iterative_step_baseline_guideline(
                        post = row['Reddit Post'], prev_step = prev_output, guidance = guidance)
                    step2_output = ChatAgent.chat(baseline_prompt) 

                # print(step2_output)
                ChatAgent.reset()

                row[f"intermediate_prompt_dim_{dim}"] = baseline_prompt
                row[f"intermediate_output_dim_{dim}"] = str(step2_output)
                prev_output = str(step2_output)

        elif FLAGS.experiment_mode == "+appr":
            baseline_prompt = "Based on the analysis above, please help the narrator of the text reappraise the situation. Your response should be concise and brief."
            for dim in FLAGS.dimensions:
                prompt_step1 = prompt_loading.build_appraisal_prompt(
                    text = row['Reddit Post'],
                    appraisal_q = dimensions_df.appraisal_questions[dim])

                # Step 1: Elicit appraisals
                step1_output = ChatAgent.chat(prompt_step1)
                # print(step1_output)
                row[f"appraisal_question_dim_{dim}"] = f"""{dimensions_df.appraisal_questions[dim]} Please provide your answer in the following format: <likert>[]</likert><rationale>[]</rationale>. Your response should be concise and brief."""
                row[f"appraisal_output_dim_{dim}"] = str(step1_output)

                # Step 2: Ask baseline reappraisal prompt
                if (prev_output == ""):
                    step2_output = ChatAgent.chat(baseline_prompt)
                else:
                    baseline_prompt = prompt_loading.build_iterative_step_w_appraisal(
                        post = row['Reddit Post'], appraisal = step1_output, prev_step = prev_output)
                    step2_output = ChatAgent.chat(baseline_prompt)
                
                # print(step2_output)
                ChatAgent.reset()

                row[f"intermediate_prompt_dim_{dim}"] = baseline_prompt
                row[f"intermediate_output_dim_{dim}"] = str(step2_output)
                prev_output = str(step2_output)

        elif FLAGS.experiment_mode == "+appr +cons":
            baseline_prompt = "Based on the analysis above, please help the narrator of the text reappraise the situation. Your response should be concise and brief."
            for dim in FLAGS.dimensions:
                prompt_step1 = prompt_loading.build_appraisal_prompt(
                    text = row['Reddit Post'],
                    appraisal_q = dimensions_df.appraisal_questions[dim])

                # Step 1: Elicit appraisals
                step1_output = ChatAgent.chat(prompt_step1)
                # print(step1_output)
                row[f"appraisal_question_dim_{dim}"] = f"""{dimensions_df.appraisal_questions[dim]} Please provide your answer in the following format: <likert>[]</likert><rationale>[]</rationale>. Your response should be concise and brief."""
                row[f"appraisal_output_dim_{dim}"] = str(step1_output)

                # Step 2.1: Ask to reappraise based on appraisal
                if (prev_output == ""):
                    baseline_prompt = f"""Based on the analysis above, please help the narrator of the text reappraise the situation. {dimensions_df.reappraisal_guidance[dim]} Your response should be concise and brief."""
                    step2_output = ChatAgent.chat(baseline_prompt)
                else:
                    baseline_prompt = prompt_loading.build_iterative_step_w_appraisal(
                        post = row['Reddit Post'], appraisal = step1_output, prev_step = prev_output)
                    step2_output = ChatAgent.chat(baseline_prompt)

                # print(step2_output)
                ChatAgent.reset()

                row[f"intermediate_prompt_.5_dim_{dim}"] = baseline_prompt
                row[f"intermediate_output_.5_dim_{dim}"] = str(step2_output)
                prev_output = str(step2_output)

                # Step 2.2 Ask to reappraise based on appraisal
                guidance = dimensions_df.reappraisal_guidance[dim]
                baseline_prompt = prompt_loading.build_iterative_step_baseline_guideline(
                    post = row['Reddit Post'], prev_step = prev_output, guidance = guidance)
                step2_output = ChatAgent.chat(baseline_prompt) 

                # print(step2_output)
                ChatAgent.reset()

                row[f"intermediate_prompt_dim_{dim}"] = baseline_prompt
                row[f"intermediate_output_dim_{dim}"] = str(step2_output)
                prev_output = str(step2_output)

        else: raise ValueError

        with open(output_jsonl_file_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        ChatAgent.reset()

if __name__ == "__main__":
    app.run(main)