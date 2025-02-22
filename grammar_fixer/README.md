# Grammar Fixer 脚本使用文档

## 主要功能
- ✅ 支持多种文本格式（.txt, .md, .tex, .html等）
- 🔄 自动分块处理大文件（通过--chunk-size参数）
- 🔐 支持OpenAI和本地Ollama两种API模式
- 📝 生成标准的diff补丁文件
- ⏯️ 断点续传功能（--resume参数）
- 📦 缓存管理（--clean-cache参数）
- 📊 实时处理进度显示
- 🛠️ 自动检测文本文件编码

## 快速开始

### 安装依赖
1. 使用前需安装依赖：
```bash
pip install openai requests
```


### 基本用法
```bash
python grammar_fixer.py input_directory \
  --api [openai|ollama] \
  --api-base [API地址] \
  --model [模型名称] \
  --output [输出文件名]
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
| 参数            | 必选       | 默认值          | 说明                       |
| --------------- | ---------- | --------------- | -------------------------- |
| `input_dir`     | 是         | -               | 要处理的输入目录路径       |
| `--api`         | 是         | -               | API提供商（openai/ollama） |
| `--api-base`    | 是         | -               | API基础地址                |
| `--api-key`     | OpenAI必选 | -               | API密钥                    |
| `--model`       | 否         | 根据API自动选择 | 使用的模型名称             |
| `--output`      | 否         | combined.patch  | 输出补丁文件名             |
| `--chunk-size`  | 否         | 0               | 文件分块大小（KB）         |
| `--extensions`  | 否         | .txt,.md,.tex   | 要处理的文件扩展名         |
| `--resume`      | 否         | -               | 断点续传模式               |
| `--clean-cache` | 否         | -               | 清理缓存重新处理           |

## 功能细节
### 文件处理流程
1. 扫描输入目录
2. 过滤非文本文件
3. 检查文件签名（大小和修改时间）
4. 分块处理（如果启用）
5. 生成语法修正建议
6. 生成diff补丁文件

### 缓存机制
缓存文件存储在`.cache/.gf_cache.json`，记录：
- 文件路径的相对位置
- 文件大小
- 最后修改时间
- 已处理的块信息
