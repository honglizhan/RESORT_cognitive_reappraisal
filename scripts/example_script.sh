#!/bin/bash

# export OPENAI_API_KEY="<your_api_key>"
# export HF_AUTH_KEY="<huggingface_auth_key>"
export AZURE_OPENAI_ENDPOINT="<your_azure_openai_endpoint>"
export AZURE_OPENAI_KEY="<your_azure_openai_key>"

cd ..

python3 -m pipeline-ITER \
    -model="gpt-4-turbo" \
    -input_data_path="./source_data/r_anger.jsonl" \
    -output_path="./model_outputs/iterative_guided_refinement" \
    -experiment_mode="self-refine" \
    -use_azure