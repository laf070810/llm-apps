#!/bin/bash
python tools/general_processor.py --input-dir input_dir --output-dir output_dir --api-type openai --api-base "https://api.deepseek.com" --model "deepseek-reasoner" --api-key "${LLM_API_KEY}" --system-message translator/system_message.txt --prompt translator/prompt.txt
