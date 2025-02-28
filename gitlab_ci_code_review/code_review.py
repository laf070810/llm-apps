import os

import requests

# 配置参数
GITLAB_URL = os.getenv("CI_API_V4_URL")
PROJECT_ID = os.getenv("CI_PROJECT_ID")
MR_IID = os.getenv("CI_MERGE_REQUEST_IID")
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_MODEL = os.getenv("LLM_API_MODEL")
LLM_API_MAXLEN = os.getenv("LLM_API_MAXLEN", "64000")


def fetch_mr_diff():
    """获取 Merge Request 的 Diff 内容"""
    url = f"{GITLAB_URL}/projects/{PROJECT_ID}/merge_requests/{MR_IID}/changes"
    headers = {"PRIVATE-TOKEN": GITLAB_TOKEN}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        changes = response.json().get("changes", [])
        return "\n".join([change.get("diff", "") for change in changes])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching MR diff: {e}")
        return None


def generate_review(diff_content):
    """调用 LLM API 生成代码审查"""
    if not diff_content:
        return "Error: No diff content available for review."

    url = f"{LLM_API_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }

    prompt = f"""请以 Markdown 格式审查以下代码变更：
1. 指出潜在问题
2. 提供优化建议
3. 总结整体质量

变更内容：
{diff_content}
"""

    data = {"model": LLM_API_MODEL, "messages": [{"role": "user", "content": prompt}]}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM API Error: {str(e)}"


def post_comment_to_mr(comment):
    """提交审查结果到 MR 评论"""
    url = f"{GITLAB_URL}/projects/{PROJECT_ID}/merge_requests/{MR_IID}/notes"
    headers = {"PRIVATE-TOKEN": GITLAB_TOKEN, "Content-Type": "application/json"}
    payload = {"body": comment}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print("Comment posted successfully")
    except requests.exceptions.RequestException as e:
        print(f"Error posting comment: {e}")


if __name__ == "__main__":
    # 执行流程
    diff_content = fetch_mr_diff()

    if diff_content:
        print(f"Diff length: {len(diff_content)} characters")
        # 截断过长的 Diff（例如限制为 64000 字符）
        truncated_diff = diff_content[: int(LLM_API_MAXLEN)]
        review = generate_review(truncated_diff)
    else:
        review = "Failed to generate code review due to missing diff."

    review = f"**[AI Code Review]**\n\n{review}"

    post_comment_to_mr(review)
