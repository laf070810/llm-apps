import os
import re

import requests

from tools.llmapi import get_llm_response

DEFAULT_PROMPT_CN = """# 角色设定
你是一位资深软件开发工程师，正在对一个Merge Request进行严格的代码审查。请基于提供的Merge Request的代码diff内容，结合以下维度进行专业分析：

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

当前仓库的ID是{project_id}，当前Merge Request的目标分支是{target_branch}。在审查的过程中，请**积极**使用工具来获取当前仓库目标分支的相关目录结构，并**积极**使用工具进一步获取目标分支中你想知道的可能相关的代码文件的具体内容。

如果遇到不确定的情况，请自行决定做法，不要请求用户的输入。

当你做出最终回答时，请严格按照上述的输出格式来输出Markdown内容，代码内容记得用Markdown格式的比如```python ```这样的标记来包裹。

当前Merge Request的标题是：{title}


以下是当前Merge Request的描述：
{description}


以下是你要审查的当前Merge Request的代码diff内容：
{diff_content}
"""

DEFAULT_PROMPT_EN = """# Role Setup  
You are a senior software development engineer conducting a rigorous code review of a Merge Request. Please perform a professional analysis based on the provided code diff content of the Merge Request, focusing on the following dimensions:  

# Review Dimensions  
1. **Code Quality**  
   - Syntax/logic error risks  
   - Boundary condition handling  
   - Exception handling mechanisms  
   - Resource management (memory/connection leaks)  
   - Concurrency safety issues  
   - Other code quality issues  

2. **Maintainability**  
   - Code readability (naming conventions, comment clarity)  
   - Function complexity (suggest splitting overly long functions)  
   - Duplicate code detection  
   - Modularity  
   - Other maintainability improvements  

3. **Security Risks**  
   - Injection vulnerabilities (SQL/command/XSS)  
   - Sensitive data handling  
   - Input validation mechanisms  
   - Permission control flaws  
   - Other security risks  

4. **Performance Optimization**  
   - Algorithm complexity  
   - Redundant computations  
   - Caching mechanisms  
   - Batch processing opportunities  
   - Other performance optimizations  

# Analysis Requirements  
1. Inspect all dimensions listed above item-by-item  
2. Indicate specific code locations (e.g., @@ line number ranges)  
3. Explain potential impacts of identified issues  
4. Provide improvement suggestions with example code  
5. Differentiate between critical issues and optimization recommendations  
6. Provide an overall quality score  

# Output Format  
Organize the report in Markdown (without wrapping the entire content in ```markdown ``` blocks). Include the following sections:  

## Code Change Summary  
Summarize the main content of the code diff being reviewed (focus on the diff itself, not your recommendations).  

## Detailed Issues and Recommendations  
List items in priority order. For each issue:  
- [🔺 Critical/⚠️ Important/🔵 Minor/💡 Suggestion] Issue category  
- Location markers  
- Description  
- Potential risks  
- Improvement suggestions  

## Review Summary  
- Overall quality score (0-10 scale)  
- Issue category statistics  
- Priority issues to address  
- Long-term maintenance recommendations  

# Example Output  
For an unsafe memory operation:
🔺 [Critical] Buffer overflow risk  
Location: @@ -15,6 +15,7 @@ void process_data()  
Issue: `strcpy` used with user input without length validation  
Risk: Arbitrary code execution  
Suggestion: Replace with `strncpy` or add length checks:  
```c  
strncpy(buffer, input, sizeof(buffer)-1);  
```

# Special Requirements  
1. Label uncertain issues with **[Needs manual verification]**  
2. Consider language-specific traits (e.g., exception chaining in Python, pointer usage in C++)  
3. Reference OWASP standards for security vulnerabilities  
4. Include complexity analysis for performance suggestions  

Current repository ID: `{project_id}`  
Merge Request target branch: `{target_branch}`  
**Actively use tools** to:  
1. Retrieve the target branch's directory structure  
2. Fetch relevant code files from the target branch as needed  

Proceed autonomously if uncertain.  

Merge Request Title: {title}

Merge Request Description:  
{description}  

Code Diff Content:  
{diff_content}  
"""

# 配置参数
GITLAB_URL = os.getenv("CI_API_V4_URL")
PROJECT_ID = os.getenv("CI_PROJECT_ID")
MR_IID = os.getenv("CI_MERGE_REQUEST_IID")
GITLAB_TOKEN = os.getenv("GITLAB_TOKEN")
LLM_API_TYPE = os.getenv("LLM_API_TYPE", "openai")
LLM_API_URL = os.getenv("LLM_API_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_API_MODEL = os.getenv("LLM_API_MODEL")
LLM_API_MAXLEN = os.getenv("LLM_API_MAXLEN", "64000")
LLM_API_PROMPT = os.getenv("LLM_API_PROMPT", DEFAULT_PROMPT_EN)


def fetch_mr_info(field):
    """获取 Merge Request 的 Diff 内容"""
    url = f"{GITLAB_URL}/projects/{PROJECT_ID}/merge_requests/{MR_IID}/changes"
    headers = {"PRIVATE-TOKEN": GITLAB_TOKEN}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        changes = response.json().get(field, [])
        return str(changes)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching MR diff: {e}")
        return None


def generate_review(diff_content):
    """调用 LLM API 生成代码审查"""
    if not diff_content:
        return "Error: No diff content available for review."

    url = f"{LLM_API_URL}"

    target_branch = fetch_mr_info("target_branch")
    title = fetch_mr_info("title")
    description = fetch_mr_info("description")

    prompt = LLM_API_PROMPT.format(
        diff_content=diff_content,
        project_id=PROJECT_ID,
        target_branch=target_branch,
        title=title,
        description=description,
    )
    print(f"\nprompt posted to LLM:\n{prompt}\n")

    try:
        return get_llm_response(
            api_type=LLM_API_TYPE,
            api_base=url,
            api_key=LLM_API_KEY,
            model=LLM_API_MODEL,
            prompt=prompt,
            remove_thinking=True,
        )
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
    diff_content = fetch_mr_info("changes")

    if diff_content:
        print(f"Diff length: {len(diff_content)} characters")
        # 截断过长的 Diff（例如限制为 64000 字符）
        truncated_diff = diff_content[: int(LLM_API_MAXLEN)]
        review = generate_review(truncated_diff)
    else:
        review = "Failed to generate code review due to missing diff."

    review = review.replace("</details>", "</details>\n")
    review = re.sub(r"(<details\b[^>]*?)\s+open(?=\s|>)", r"\1 close", review)

    review = f'<details style="color:gray;background-color: #f8f8f8;padding: 8px;border-radius: 4px;" close> <summary>[AI Code Review]</summary>\n\n{review}\n</details>'
    # review = f"**[AI Code Review]**\n\n{review}"

    post_comment_to_mr(review)
