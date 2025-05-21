import argparse
import json
import os
import time
from pathlib import Path

from llmapi import get_llm_response
from tqdm import tqdm


class ChunkProcessor:
    def __init__(self, args):
        self.args = args
        self.input_dir = Path(args.input_dir)
        self.output_dir = Path(args.output_dir)
        self.state_file = self.output_dir / "processing_state.json"
        self.manifest = self._load_manifest()
        self.processed_chunks = self._load_processing_state()

        # 准备prompt模板
        if os.path.isfile(args.prompt):
            with open(args.prompt, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        else:
            self.prompt_template = args.prompt

        # 准备系统提示
        if os.path.isfile(args.system_message):
            with open(args.system_message, "r", encoding="utf-8") as f:
                self.system_message = f.read()
        else:
            self.system_message = args.system_message

    def _load_manifest(self) -> dict:
        manifest_path = self.input_dir / "manifest.json"
        with open(manifest_path, "r") as f:
            return json.load(f)

    def _load_processing_state(self) -> set:
        if self.state_file.exists():
            with open(self.state_file, "r") as f:
                return set(json.load(f)["processed_chunks"])
        return set()

    def _save_processing_state(self):
        state = {
            "processed_chunks": list(self.processed_chunks),
            "timestamp": time.time(),
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f)

    def _get_chunk_content(self, chunk_name: str) -> str:
        chunk_path = self.input_dir / chunk_name
        with open(chunk_path, "r", encoding="utf-8") as f:
            return f.read()

    def _process_chunk(self, chunk_name: str):
        # 读取原始分块内容
        chunk_content = self._get_chunk_content(chunk_name)

        # 构建最终prompt
        final_prompt = self.prompt_template.replace("{{chunk}}", chunk_content)

        # 调用LLM接口
        response = get_llm_response(
            api_type=self.args.api_type,
            api_base=self.args.api_base,
            api_key=self.args.api_key,
            model=self.args.model,
            prompt=final_prompt,
            system_message=self.system_message,
            remove_thinking=self.args.remove_thinking,
            **json.loads(self.args.api_options or "{}"),
        )

        # 保存处理后的分块
        output_path = self.output_dir / chunk_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response)

        # 更新处理状态
        self.processed_chunks.add(chunk_name)
        self._save_processing_state()

    def process_all(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 收集所有需要处理的分块
        all_chunks = []
        for file_entry in self.manifest["files"]:
            all_chunks.extend(file_entry["chunks"])

        # 过滤已处理的分块
        pending_chunks = [c for c in all_chunks if c not in self.processed_chunks]

        # 创建线程安全的进度条和状态锁
        pbar_lock = threading.Lock()
        state_lock = threading.Lock()

        # 使用进度条
        with tqdm(total=len(pending_chunks), desc="Processing chunks") as pbar:
            with ThreadPoolExecutor(max_workers=self.args.concurrency) as executor:
                futures = {
                    executor.submit(
                        self._process_chunk_threadsafe,
                        chunk,
                        pbar,
                        pbar_lock,
                        state_lock,
                    ): chunk
                    for chunk in pending_chunks
                }

                for future in as_completed(futures):
                    chunk = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        with pbar_lock:
                            pbar.write(f"Error processing {chunk}: {str(e)}")

    def _process_chunk_threadsafe(self, chunk, pbar, pbar_lock, state_lock):
        try:
            # 处理分块的核心逻辑保持不变
            self._process_chunk(chunk)
        finally:
            # 原子更新进度条和状态
            with state_lock:
                self.processed_chunks.add(chunk)
                self._save_processing_state()

            with pbar_lock:
                pbar.update(1)


def parse_args():
    parser = argparse.ArgumentParser(description="LLM分块处理器")

    # 必需参数
    parser.add_argument(
        "--input-dir", required=True, help="包含分块文件和manifest.json的输入目录"
    )
    parser.add_argument("--output-dir", required=True, help="处理后的分块输出目录")
    parser.add_argument(
        "--prompt",
        required=True,
        help="提示词模板（直接文本或文件路径），使用{{chunk}}作为分块占位符",
    )

    # API参数
    parser.add_argument("--api-type", required=True, help="API类型")
    parser.add_argument("--api-base", required=True, help="API基础地址")
    parser.add_argument("--api-key", required=True, help="API密钥")
    parser.add_argument("--model", required=True, help="模型名称")

    # 可选参数
    parser.add_argument(
        "--system-message", default="", help="系统消息（直接文本或文件路径）（默认空）"
    )
    parser.add_argument(
        "--remove-thinking",
        action="store_true",
        default=False,
        help="是否移除思维过程（默认不移除）",
    )
    parser.add_argument(
        "--api-options", default="{}", help="额外API选项（JSON格式字符串）"
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="并发处理数 (默认: 1)"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    processor = ChunkProcessor(args)
    processor.process_all()

    # 复制原始manifest到输出目录
    with open(Path(args.input_dir) / "manifest.json", "r") as f:
        manifest = json.load(f)
    with open(Path(args.output_dir) / "manifest.json", "w") as f:
        json.dump(manifest, f)

    print("处理完成！可以使用原始脚本的merge命令恢复文件：")
    print(f"python original_script.py merge -i {args.output_dir} -o <恢复目录>")


if __name__ == "__main__":
    main()
