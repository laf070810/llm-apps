#!/usr/bin/env python3
import argparse
import glob
import json
import mimetypes
import os
import re
import subprocess
import sys
from typing import Optional

import requests

# Conditional OpenAI import
try:
    import openai
except ImportError:
    openai = None


CACHE_DIR = ".cache"
CACHE_FILENAME = ".gf_cache.json"


def load_cache(cache_dir: str = CACHE_DIR) -> dict:
    """Load processing cache from file"""
    cache_path = os.path.join(cache_dir, CACHE_FILENAME)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_cache(cache_data: dict, cache_dir: str = CACHE_DIR):
    """Save processing cache to file"""
    cache_path = os.path.join(cache_dir, CACHE_FILENAME)
    with open(cache_path, "w") as f:
        json.dump(cache_data, f, indent=2)


def get_file_signature(filepath: str) -> dict:
    """Get unique file signature using mtime and size"""
    stat = os.stat(filepath)
    return {
        "mtime": stat.st_mtime,
        "size": stat.st_size,
        "path": os.path.relpath(filepath, start=os.path.dirname(filepath)),
    }


def split_text_file(content: str, chunk_size: int) -> list[tuple[str, int]]:
    """Split content into chunks with line number tracking"""
    lines = content.split("\n")
    chunks = []
    current_line = 0

    while current_line < len(lines):
        chunk_start_line = current_line + 1  # Lines are 1-based in diffs
        current_chunk = []
        chunk_length = 0

        # Accumulate lines until reaching chunk size
        while current_line < len(lines) and chunk_length < chunk_size * 1024:
            line = lines[current_line]
            current_chunk.append(line)
            chunk_length += len(line.encode("utf-8"))
            current_line += 1

        # Add the chunk with its start line
        if current_chunk:
            chunks.append(("\n".join(current_chunk), chunk_start_line))

    return chunks


def is_text_file(filepath: str) -> bool:
    """Check if a file is text-based using multiple methods"""
    # 1. Check MIME type
    mime, _ = mimetypes.guess_type(filepath)
    if mime and mime.startswith("text/"):
        return True

    # 2. Check file extension
    ext = os.path.splitext(filepath)[1].lower()
    if ext in {".txt", ".md", ".tex", ".html", ".css", ".js"}:
        return True

    # 3. Read first 1024 bytes to check for null bytes
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:
                return False
    except Exception:
        return False

    return True


def process_text_file(
    file_path: str,
    api_base: str,
    api_key: str,
    model: str,
    chunk_size: int = 0,
    api_type: str = "openai",
    cache: Optional[dict] = None,
) -> Optional[str]:
    """Process a single text file with LLM API

    Args:
        file_path: Path to the text file to process
        api_base: Base URL for the API endpoint
        api_key: API authentication key
        model: Model name/tag to use
        chunk_size: Split content into chunks (KB), 0 for no splitting
        api_type: API provider type (openai/ollama)
        cache: Optional cache dictionary for resume functionality

    Returns:
        Generated diff content or None if processing failed
    """
    import time

    start_time = time.time()

    print(f"\033[1;34m\n{'='*40}\nProcessing: {file_path}\n{'='*40}\033[0m")

    with open(file_path, "r") as f:
        content = f.read()

    # Split into chunks if chunk_size specified
    chunks = split_text_file(content, chunk_size) if chunk_size > 0 else [(content, 1)]
    total_chunk_num = len(chunks)

    try:
        file_ext = os.path.splitext(file_path)[1]
        processed_chunks = (
            sum(
                1
                for k in cache
                if k.startswith(f"{get_file_signature(file_path)['path']}_chunk_")
            )
            if cache
            else 0
        )

        # Skip already processed chunks
        chunks = chunks[processed_chunks:]
        if processed_chunks > 0:
            print(
                f"\033[33mSkipping {processed_chunks} already processed chunks\033[0m"
            )

        for i, (chunk_content, start_line) in enumerate(chunks, processed_chunks + 1):
            # Generate system message with file type context
            is_partial = len(chunks) > 1
            system_message = f"""You are a professional English proofreader. Correct grammar, spelling and punctuation errors in the following content. Preserve all formatting and document structure (especially {file_ext} syntax and directives). Respond ONLY with the corrected content without any explanations or additional text. Maintain the original writing style and formatting. DO NOT use any diff formatting in your response."""

            if is_partial:
                system_message += "\nNOTE: This is a partial excerpt. Ensure line numbers match full document."

            response_chunks = []
            # API-specific configuration
            if api_type == "openai":
                if openai is None:
                    raise ImportError(
                        "openai package is required for OpenAI API. Install with: pip install openai"
                    )

                client = openai.OpenAI(base_url=api_base, api_key=api_key)

                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {
                            "role": "user",
                            "content": f"Please correct the grammar in this content:\n\n{chunk_content}",
                        },
                    ],
                    temperature=0.1,
                    max_tokens=4096,
                    stream=True,
                )
                response_chunks = response

            elif api_type == "ollama":
                response = requests.post(
                    f"{api_base}/api/generate",
                    json={
                        "model": model,
                        "prompt": f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\nPlease correct the grammar in this content:\n\n{chunk_content}[/INST]",
                        "stream": True,
                        "options": {"temperature": 0.1, "num_predict": 4096},
                    },
                    headers={"Content-Type": "application/json"},
                    stream=True,
                )
                response.raise_for_status()
                response_chunks = response.iter_lines()

            else:
                raise NotImplementedError(f"unknown API type: {api_type}")

            print(
                f"\n\033[33m[API Request] Model: {model} | Chunk {i}/{total_chunk_num} | Size: {len(chunk_content)/1024:.1f}KB\033[0m"
            )
            print("\033[2mStreaming response...\033[0m")
            print(f"\033[32m[Chunk {i} Response]\033[0m ", end="", flush=True)
            chunk_response = []

            # Process response chunks
            for chunk in response_chunks:
                content = None
                try:
                    if api_type == "openai":
                        content = chunk.choices[0].delta.content
                    elif api_type == "ollama":
                        if chunk:
                            json_chunk = json.loads(chunk)
                            content = json_chunk.get("response", "")
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"\n\033[31mError parsing response: {str(e)}\033[0m")
                    continue

                if content:
                    print(content, end="", flush=True)
                    chunk_response.append(content)

            # Remove any <think>...</think> blocks from response
            cleaned_response = re.sub(
                r"<think>.*?</think>", "", "".join(chunk_response), flags=re.DOTALL
            )

            # Collect corrected content
            chunk_response = ["".join(cleaned_response).strip()]

            # Save the content for this chunk immediately
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR)
            temp_file = os.path.join(
                CACHE_DIR, f"{os.path.basename(file_path)}.chunk_{i}"
            )
            with open(temp_file, "w") as f:
                f.write(chunk_response[0])
            print(f"\n\033[32mSaved temporary patch file: {temp_file}\033[0m")

            # Save cache after each chunk
            if cache is not None:
                file_sig = get_file_signature(file_path)
                cache_key = f"{file_sig['path']}_chunk_{i}"
                cache[cache_key] = file_sig
                save_cache(cache)

        # Combine all chunks into final file
        corrected_content = ""
        for i in range(1, total_chunk_num + 1):
            temp_file = os.path.join(
                CACHE_DIR, f"{os.path.basename(file_path)}.chunk_{i}"
            )
            with open(temp_file, "r") as f:
                corrected_content += f.read() + "\n"

        # Save the final file
        corrected_path = os.path.join(
            CACHE_DIR, f"{os.path.basename(file_path)}.corrected"
        )
        with open(corrected_path, "w") as f:
            f.write(corrected_content)

        diff_content = generate_diff(file_path, corrected_path)
        elapsed = time.time() - start_time

        print(f"\n\033[2m\nProcessing completed in {elapsed:.1f}s\033[0m")
        return diff_content
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def generate_diff(original_path: str, corrected_path: str) -> str:
    """Generate git-style diff between original and corrected files"""
    return subprocess.run(
        f"diff -u {original_path} {corrected_path}", shell=True, capture_output=True
    ).stdout.decode()


def process_file(
    file_path: str,
    api_base: str,
    api_key: str,
    model: str,
    chunk_size: int = 0,
    api_type: str = "openai",
    extensions: Optional[list[str]] = None,
    resume: bool = False,
) -> Optional[str]:
    """Process a single file with LLM API

    Args:
        file_path: Path to the file to process
        api_base: API endpoint URL
        api_key: Authentication key for the API
        model: Model identifier to use
        chunk_size: Maximum chunk size in KB (0 for no chunking)
        api_type: Type of API service (openai/ollama)
        extensions: Allowed file extensions for processing

    Returns:
        Diff content or None if processing skipped/failed
    """
    # Validate file extension
    if extensions and os.path.splitext(file_path)[1].lower() not in extensions:
        print(f"\033[33mSkipping unsupported file type: {file_path}\033[0m")
        return None

    cache = load_cache() if resume else None
    return process_text_file(
        file_path, api_base, api_key, model, chunk_size, api_type, cache
    )


def process_directory(
    input_dir: str,
    api_base: str,
    api_key: str,
    model: str,
    chunk_size: int = 0,
    api_type: str = "openai",
    extensions: Optional[list[str]] = None,
    resume: bool = False,
    clean_cache: bool = False,
) -> list[str]:
    """Process all valid files in a directory

    Args:
        input_dir: Directory to scan for files
        api_base: API endpoint URL
        api_key: Authentication key for the API
        model: Model identifier to use
        chunk_size: Maximum chunk size in KB (0 for no chunking)
        api_type: Type of API service (openai/ollama)
        extensions: Allowed file extensions
        resume: Enable resume mode using cache
        clean_cache: Clear existing cache before processing

    Returns:
        List of generated patch file paths
    """
    patch_files = []

    if clean_cache:
        cache_path = os.path.join(CACHE_DIR, CACHE_FILENAME)
        if os.path.exists(cache_path):
            os.remove(cache_path)

    cache = load_cache() if resume else {}

    # Get all files except .patch files
    all_files = glob.glob(os.path.join(input_dir, "**", "*"), recursive=True)

    # Filter files
    input_files = []
    for f in all_files:
        if (
            f.endswith(".patch")
            or not os.path.isfile(f)
            or (extensions and os.path.splitext(f)[1].lower() not in extensions)
            or not is_text_file(f)
        ):
            continue

        file_sig = get_file_signature(f)
        # Check chunks in resume mode
        if resume:
            processed_chunks = sum(1 for k in cache if k.startswith(file_sig["path"]))
            if processed_chunks > 0:
                print(f"\033[33mResuming from chunk {processed_chunks} for: {f}\033[0m")
                input_files.append(f)
                continue

        # Only add new files not in cache
        if not resume or file_sig["path"] not in cache:
            input_files.append(f)

    total_files = len(input_files)
    print(
        f"\n\033[1;35mFound {total_files} files to process ({len(cache)} cached)\033[0m"
    )

    for i, input_file in enumerate(input_files, 1):
        file_sig = get_file_signature(input_file)
        cache_key = file_sig["path"]
        print(f"\n\033[1;36mProcessing file {i}/{total_files}\033[0m")

        # Get number of processed chunks from cache
        processed_chunks = sum(1 for k in cache if k.startswith(f"{cache_key}_chunk_"))
        if processed_chunks > 0:
            print(f"\033[33mResuming from chunk {processed_chunks+1}\033[0m")

        diff_content = process_text_file(
            input_file, api_base, api_key, model, chunk_size, api_type, cache
        )
        if not diff_content:
            continue

        patch_file = os.path.join(CACHE_DIR, f"{os.path.basename(input_file)}.patch")
        with open(patch_file, "w") as f:
            f.write(diff_content)
        print(f"\033[32mGenerated patch file: {patch_file}\033[0m")
        patch_files.append(patch_file)

        # Update cache only after successful processing
        cache[cache_key] = file_sig
        # Save cache after every file
        save_cache(cache)

    # Final cache save
    save_cache(cache)
    return patch_files


def combine_patches(patch_files: list[str], output_file: str) -> None:
    """Combine multiple patch files into a single output file

    Args:
        patch_files: List of paths to patch files
        output_file: Path for combined output file
    """
    print(f"\n\033[1;35mCombining {len(patch_files)} patch files...\033[0m")
    with open(output_file, "w") as outfile:
        for i, patch_file in enumerate(patch_files, 1):
            print(f"\033[33mMerging ({i}/{len(patch_files)}): {patch_file}\033[0m")
            with open(patch_file, "r") as infile:
                outfile.write(infile.read())
    print(f"\033[1;32mSuccessfully created combined patch: {output_file}\033[0m")


def main():
    """Main entry point for the grammar fixer CLI"""
    parser = argparse.ArgumentParser(
        description="Grammar Fixer - Correct text files using LLM APIs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_path", help="Input path (file or directory) to process")
    parser.add_argument(
        "--extensions",
        default=".txt,.md,.tex",
        help="Comma-separated list of file extensions to process",
    )
    parser.add_argument(
        "--api",
        choices=["openai", "ollama"],
        required=True,
        help="API provider to use",
    )
    parser.add_argument(
        "--api-base",
        required=True,
        help="API endpoint URL (e.g. https://api.openai.com/v1 or http://localhost:11434)",
    )
    parser.add_argument(
        "--api-key",
        help="API key (required for OpenAI)",
    )
    parser.add_argument(
        "--output",
        default="combined.patch",
        help="Output patch file name (for single file processing)",
    )
    parser.add_argument(
        "--model",
        default=(
            "gpt-4-turbo-preview"
            if "--api openai" in " ".join(sys.argv)
            else "llama3:8b"
        ),
        help="Model name to use",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Split large files into chunks (KB), 0 to disable chunking",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous processing using cache",
    )
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Clear existing cache before processing",
    )

    args = parser.parse_args()

    if args.resume and args.clean_cache:
        print("Error: Cannot use both --resume and --clean-cache")
        return

    # Parse extensions
    extensions = (
        [ext.strip().lower() for ext in args.extensions.split(",")]
        if args.extensions
        else []
    )

    if not os.path.exists(args.input_path):
        print(f"Error: {args.input_path} does not exist")
        return

    # Process based on input type
    if os.path.isfile(args.input_path):
        # Single file processing
        diff_content = process_file(
            args.input_path,
            args.api_base,
            args.api_key,
            args.model,
            args.chunk_size,
            args.api,
            extensions,
            args.resume,
        )
        if diff_content:
            with open(args.output, "w") as f:
                f.write(diff_content)
            print(f"\033[1;32mGenerated patch file: {args.output}\033[0m")
    else:
        # Directory processing
        patch_files = process_directory(
            args.input_path,
            args.api_base,
            args.api_key,
            args.model,
            args.chunk_size,
            args.api,
            extensions,
            args.resume,
            args.clean_cache,
        )
        if patch_files:
            combine_patches(patch_files, args.output)
        else:
            print("\033[33mNo files processed\033[0m")

    print(f"\033[1;32mProcessing completed\033[0m")


if __name__ == "__main__":
    main()
