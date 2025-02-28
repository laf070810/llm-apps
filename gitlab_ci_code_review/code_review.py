import os

import requests

DEFAULT_PROMPT = """# 角色设定
你是一位资深软件开发工程师，正在进行严格的代码审查。请基于提供的代码diff内容，结合以下维度进行专业分析：

# 审查维度
1. 代码质量
- 语法/逻辑错误风险
- 边界条件处理
- 异常处理机制
- 资源管理（内存/连接泄漏风险）
- 并发安全问题
- 其他方面的代码质量问题

2. 可维护性
- 代码可读性（命名规范/注释清晰度）
- 函数复杂度（建议拆分过长的函数）
- 重复代码检测
- 模块化程度
- 其他方面的可维护性改进

3. 安全风险
- 注入攻击可能性（SQL/命令/跨站脚本）
- 敏感数据处理
- 输入验证机制
- 权限控制缺陷
- 其他方面的安全风险

4. 性能优化
- 算法复杂度
- 冗余计算
- 缓存机制
- 批量处理可能性
- 其他方面的性能优化

# 分析要求
1. 逐项检查上述所有维度
2. 指出具体代码位置（如：@@标记的行号范围）
3. 对发现的问题说明潜在影响
4. 提供改进建议并给出示例代码
5. 区分关键问题和优化建议
6. 对整体质量给一个总结评分

# 输出格式
请用Markdown格式来组织报告（但不必使用类似```markdown   ```这样的标记来包裹全文），主要包含以下章节：

## 代码变更总结
总结一下你所审查的代码diff的主要内容（是代码diff本身的总结，而不是你的审查建议的总结）

## 详细问题和建议
逐项检查上述所有维度，按优先级排序，每个问题包含：
- [🔺 严重/⚠️ 重要/🔵 一般/💡 建议] 问题分类
- 位置标记
- 问题描述
- 潜在风险
- 改进建议

## 审查总结
- 整体质量评分（10分制，输出中要包含分数的范围）
- 问题分类数量统计
- 推荐优先处理的问题
- 长期维护建议

# 示例输出
当分析到不安全的内存操作时：
"🔺 [严重] 缓冲区溢出风险
位置：@@ -15,6 +15,7 @@ void process_data()
问题：strcpy直接使用用户输入，未校验长度
风险：可能导致任意代码执行
建议：改用strncpy或增加长度校验：strncpy(buffer, input, sizeof(buffer)-1);"

# 特别要求
1. 对不确定的问题标注[需人工复核]
2. 区分语言特性（如Python应关注异常链，C++注意指针使用）
3. 对安全漏洞提供OWASP参考标准
4. 性能建议需包含复杂度分析

以下是你要审查的代码diff内容：
{diff_content}
"""

# 配置参数
GITLAB_URL = os.getenv("CI_API_V4_URL")
PROJECT_ID = os.getenv("CI_PROJECT_ID")
MR_IID = os.getenv("CI_MERGE_REQUEST_IID")
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_MODEL = os.getenv("LLM_API_MODEL")
LLM_API_MAXLEN = os.getenv("LLM_API_MAXLEN", "64000")
LLM_API_PROMPT = os.getenv("LLM_API_PROMPT", DEFAULT_PROMPT)


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

    url = f"{LLM_API_URL}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
    }

    prompt = LLM_API_PROMPT.format(diff_content=diff_content)
    data = {
        "model": LLM_API_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

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
