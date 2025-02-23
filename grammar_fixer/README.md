# Grammar Fixer 脚本使用文档

Grammar Fixer是一个利用LLM对单个或多个文本文件中的英语文本进行语法纠错的工具。输入为要检查的文本文件路径或包含多个文本文件的目录，输出为英语语法纠错后的unified diff文件。

## 主要功能
- ✅ 支持多种文本格式（.txt, .md, .tex, .html等）
- 🔄 自动分块处理大文件（通过--chunk-size参数）
- 🔐 支持OpenAI和本地Ollama两种API模式
- 📝 生成标准的diff补丁文件
- ⏯️ 断点续传功能（--resume参数）
- 📦 缓存管理（--clean-cache参数）
- 📊 实时处理进度显示（带颜色标记）
- 🛠️ 自动检测文本文件编码
- 🌊 流式API响应处理（实时显示修正内容）

## 快速开始

### 安装依赖
1. 使用前需安装依赖：

```bash
pip install requests  # 基础依赖
pip install openai    # 仅在使用OpenAI API时需要
```

### 基本用法
```bash
# 处理单个文件
python grammar_fixer.py document.txt \
  --api [openai|ollama] \
  --api-base [API地址] \
  --model [模型名称] \
  --output doc_fixes.patch

# 处理整个目录
python grammar_fixer.py input_directory \
  --api [openai|ollama] \
  --api-base [API地址] \
  --model [模型名称] \
  --output combined.patch
```

应用补丁文件：
```bash
patch < grammar_fixes.patch
```

取消应用补丁文件：
```bash
patch -R < grammar_fixes.patch
```

### 示例

#### 使用OpenAI API
```bash
python grammar_fixer.py docs \
  --api openai \
  --api-base https://api.openai.com/v1 \
  --api-key sk-xxxx \
  --model gpt-4-turbo-preview \
  --output grammar_fixes.patch
```

#### 使用本地Ollama
```bash
python grammar_fixer.py blog_posts \
  --api ollama \
  --api-base http://localhost:11434 \
  --model llama3:8b \
  --chunk-size 512
```

#### 恢复处理中断的任务
```bash
python grammar_fixer.py docs \
  --api openai \
  --api-base https://api.openai.com/v1 \
  --api-key sk-xxxx \
  --model gpt-4-turbo-preview \
  --output grammar_fixes.patch
  --resume
```

#### 清理缓存重新处理
```bash
python grammar_fixer.py blog_posts \
  --api ollama \
  --api-base http://localhost:11434 \
  --model llama3:8b \
  --chunk-size 512
  --clean-cache
```

## 参数说明
| 参数            | 必选       | 默认值          | 说明                                                                     |
| --------------- | ---------- | --------------- | ------------------------------------------------------------------------ |
| `input_path`    | 是         | -               | 要处理的输入文件或目录路径                                               |
| `--api`         | 是         | -               | API提供商（openai/ollama）                                               |
| `--api-base`    | 是         | -               | API基础地址                                                              |
| `--api-key`     | OpenAI必选 | -               | OpenAI API密钥（通过环境变量OPENAI_API_KEY设置更安全）                   |
| `--model`       | 否         | 根据API自动选择 | 使用的模型名称（OpenAI默认：gpt-4-turbo-preview，Ollama默认：llama3:8b） |
| `--output`      | 否         | combined.patch  | 输出补丁文件名                                                           |
| `--chunk-size`  | 否         | 0               | 文件分块大小（KB），0表示禁用分块                                        |
| `--extensions`  | 否         | .txt,.md,.tex   | 要处理的文件扩展名（逗号分隔）                                           |
| `--resume`      | 否         | true            | 断点续传模式（自动从上次中断处继续），默认启用                           |
| `--clean-cache` | 否         | -               | 清理缓存并退出（不进行文本处理）                                         |

## 注意事项
1. API密钥处理：
   - OpenAI密钥通过--api-key参数传递
   - Ollama本地部署无需密钥
2. 缓存文件：
   - 包含文件处理进度和元数据
   - 存储于项目目录的.cache/文件夹
   - 不应分享给他人或提交到版本控制
3. 补丁安全：
   - **生成后建议用文本编辑器检查变更**
   - **如果想修改生成的diff文件里的错误，不建议直接改diff文件本身，因为改了有可能会导致diff匹配不出来，尤其是在diff文件比较大、需要修改的地方比较多的情况下。在diff文件本身比较大或对diff文件的改动比较多时，最好将文档仓库先用git来commit一下，然后直接应用这个diff文件，再根据git的diff情况来修改**
   - 应用前可以考虑备份原始文件：
     ```bash
     cp -r input_dir/ input_dir_backup/
     ```
   - 支持撤销补丁：`patch -R < grammar_fixes.patch`

## 功能细节
### 文件处理流程
1. 检查输入类型（文件/目录）
2. 如果是目录：扫描目录下所有文件
3. 过滤非文本文件（通过扩展名和二进制检测）
4. 检查文件签名（大小和修改时间）
5. 分块处理（如果启用--chunk-size参数）
6. 通过流式API处理每个文本块
7. 生成语法修正建议（保留原始格式）
8. 组合生成最终diff补丁文件

### 缓存机制
缓存文件存储在`.cache/.gf_cache.json`，记录：
- 文件路径的相对位置
- 文件大小（字节）
- 最后修改时间（UNIX时间戳）
- 已处理的块信息（用于--resume模式）

### 分块处理
当启用--chunk-size参数时：
- 大文件会被分割为指定大小的文本块
- 每个块独立处理并缓存进度
- 保持完整的行结构（不会在行中间分割）
