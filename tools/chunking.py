import argparse
import hashlib
import json
import os
from pathlib import Path

from is_text_file import is_text_file


def find_split_point(content, start, end, encoding="utf-8"):
    # 优先在换行符处分割
    split_pos = content.rfind(b"\n", start, end)
    if split_pos != -1:
        return split_pos + 1  # 包含换行符

    # 其次在常见分隔符处分割（如空格、分号等）
    for delim in [b" ", b";", b")", b"]", b"}"]:
        split_pos = content.rfind(delim, start, end)
        if split_pos != -1:
            return split_pos + 1

    # 最后在指定位置强制分割
    return end


def split_file(file_path, chunk_size, output_dir, rel_path, manifest):
    with open(file_path, "rb") as f:
        content = f.read()

    chunks = []
    current = 0
    total = len(content)
    file_hash = hashlib.md5(rel_path.encode()).hexdigest()

    while current < total:
        end = min(current + chunk_size, total)
        split_pos = find_split_point(content, current, end)
        chunk = content[current:split_pos]
        chunk_num = len(chunks) + 1
        chunk_name = f"{file_hash}_{chunk_num}.chunk"
        chunk_path = os.path.join(output_dir, chunk_name)

        with open(chunk_path, "wb") as f:
            f.write(chunk)

        chunks.append(chunk_name)
        current = split_pos

    manifest_entry = {
        "original_path": rel_path,
        "chunks": chunks,
        "hash": hashlib.md5(content).hexdigest(),
    }
    manifest["files"].append(manifest_entry)


def split_command(args):
    input_path = Path(args.input)
    extensions = args.extensions.split(",")
    chunk_size = args.size
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)
    manifest = {"version": 1, "files": []}

    for item in input_path.rglob("*") if input_path.is_dir() else [input_path]:
        if item.is_file() and item.suffix in extensions:
            if is_text_file(item):
                rel_path = str(item.relative_to(input_path))
                split_file(item, chunk_size, output_dir, rel_path, manifest)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def merge_command(args):
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    with open(input_dir / "manifest.json") as f:
        manifest = json.load(f)

    for file_entry in manifest["files"]:
        original_path = output_dir / file_entry["original_path"]
        original_path.parent.mkdir(parents=True, exist_ok=True)

        full_content = bytearray()
        for chunk_name in file_entry["chunks"]:
            with open(input_dir / chunk_name, "rb") as f:
                full_content.extend(f.read())

        # 校验哈希值
        if hashlib.md5(full_content).hexdigest() != file_entry["hash"]:
            print(f"Warning: Hash mismatch for {file_entry['original_path']}")

        with open(original_path, "wb") as f:
            f.write(full_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="智能文本分块工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Split 命令
    split_parser = subparsers.add_parser("split", help="分割文件")
    split_parser.add_argument("-i", "--input", required=True, help="输入文件或目录")
    split_parser.add_argument(
        "-e", "--extensions", required=True, help="要处理的后缀列表（逗号分隔）"
    )
    split_parser.add_argument(
        "-s", "--size", type=int, required=True, help="分块大小（字节）"
    )
    split_parser.add_argument("-o", "--output", required=True, help="输出目录")

    # Merge 命令
    merge_parser = subparsers.add_parser("merge", help="合并文件")
    merge_parser.add_argument("-i", "--input", required=True, help="包含分块的目录")
    merge_parser.add_argument("-o", "--output", required=True, help="恢复输出目录")

    args = parser.parse_args()

    if args.command == "split":
        split_command(args)
    elif args.command == "merge":
        merge_command(args)
    print("操作完成！")
