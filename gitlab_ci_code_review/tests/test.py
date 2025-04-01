import os
import subprocess

if os.environ.get("GITLAB_TOKEN", "") == "" or os.environ.get("LLM_API_KEY", "") == "":
    raise Exception("tokens not given")

os.environ["CI_API_V4_URL"] = "https://git.tsinghua.edu.cn/api/v4"
# os.environ["CI_PROJECT_ID"] = "38131"
os.environ["CI_PROJECT_ID"] = "38835"
os.environ["CI_MERGE_REQUEST_IID"] = "1"

# os.environ["LLM_API_TYPE"] = "openai"
# os.environ["LLM_API_URL"] = "https://api.deepseek.com"
# os.environ["LLM_API_MODEL"] = "deepseek-reasoner"

# os.environ["LLM_API_TYPE"] = "openai"
# os.environ["LLM_API_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# os.environ["LLM_API_MODEL"] = "qwen-plus"

# os.environ["LLM_API_TYPE"] = "ollama"
# os.environ["LLM_API_URL"] = "http://192.168.20.10:8001"
# os.environ["LLM_API_MODEL"] = "qwq-20480ctx"

os.environ["LLM_API_TYPE"] = "dify"
os.environ["LLM_API_URL"] = "http://192.168.20.41:8002/v1"

subprocess.run("python -m gitlab_ci_code_review.code_review", shell=True)
