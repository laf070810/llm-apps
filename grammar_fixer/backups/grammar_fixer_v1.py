#!/usr/bin/env python3
import argparse
import difflib
import glob
import os
from typing import Optional

import openai


def process_latex_file(
    file_path: str, api_base: str, api_key: str, model: str, chunk_size: int = 0
) -> Optional[str]:
    """Process a single LaTeX file with OpenAI API"""
    import time

    start_time = time.time()

    print(f"\033[1;34m\n{'='*40}\nProcessing: {file_path}\n{'='*40}\033[0m")

    with open(file_path, "r") as f:
        content = f.read()

    # Split into chunks if chunk_size specified
    chunks = [content]
    if chunk_size > 0:
        chunks = split_latex(content, chunk_size)
        print(f"\033[33mSplit into {len(chunks)} chunks for processing\033[0m")

    client = openai.OpenAI(base_url=api_base, api_key=api_key)

    full_response = []
    try:
        for i, chunk_content in enumerate(chunks, 1):
            print(
                f"\n\033[33m[API Request] Model: {model} | Chunk {i}/{len(chunks)} | Size: {len(chunk_content)/1024:.1f}KB\033[0m"
            )
            print("\033[2mStreaming response...\033[0m")

            is_partial = len(chunks) > 1
            system_message = "You are a professional English proofreader. Correct grammar, spelling and punctuation errors in the following LaTeX content. Preserve all LaTeX commands, math expressions, and document structure. Respond ONLY with the corrected LaTeX content without any explanations or additional text. Maintain the original writing style and formatting."
            if is_partial:
                system_message += " NOTE: This is a partial document excerpt. Focus on local corrections while maintaining consistency with surrounding content. Do NOT add section headers or formatting."

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_message,
                    },
                    {
                        "role": "user",
                        "content": f"Correct the grammar in this LaTeX content. Respond ONLY with the corrected text using the same LaTeX formatting:\n\n{chunk_content}",
                    },
                ],
                temperature=0.1,
                max_tokens=4096,
                stream=True,
            )

            print(f"\033[32m[Chunk {i} Response]\033[0m ", end="", flush=True)
            chunk_response = []

            for chunk in response:
                if chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    print(delta, end="", flush=True)
                    chunk_response.append(delta)

            full_response.append("".join(chunk_response).strip())

        corrected = "\n\n".join(full_response).strip()
        elapsed = time.time() - start_time

        print(f"\n\033[2m\nProcessing completed in {elapsed:.1f}s\033[0m")
        return corrected
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None


def generate_diff(original_path: str, corrected_content: str) -> str:
    """Generate git-style diff using difflib"""
    with open(original_path, "r") as f:
        original_lines = f.readlines()

    corrected_lines = corrected_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        corrected_lines,
        fromfile=original_path,
        tofile=original_path,
        lineterm="",
    )
    return "\n".join(diff)


def split_latex(content: str, chunk_size: int) -> list[str]:
    """Split LaTeX content into meaningful chunks preserving structure"""
    chunks = []
    current_chunk = []
    current_length = 0

    # Split on paragraphs (empty lines) while preserving structure
    paragraphs = content.split("\n\n")

    for para in paragraphs:
        para_length = len(para)
        if current_length + para_length > chunk_size * 1024 and current_chunk:
            # Finalize current chunk
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(para)
        current_length += para_length

        # Check if we need to split mid-paragraph (rare case)
        if current_length > chunk_size * 1024:
            # Fallback to simple split
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def process_directory(
    input_dir: str, api_base: str, api_key: str, model: str, chunk_size: int = 0
) -> list[str]:
    """Process all LaTeX files in directory and return list of patch files"""
    patch_files = []
    tex_files = glob.glob(os.path.join(input_dir, "**", "*.tex"), recursive=True)
    total_files = len(tex_files)

    print(f"\n\033[1;35mFound {total_files} LaTeX files to process\033[0m")

    for i, tex_file in enumerate(tex_files, 1):
        print(f"\n\033[1;36mProcessing file {i}/{total_files}\033[0m")
        corrected = process_latex_file(tex_file, api_base, api_key, model, chunk_size)
        if not corrected:
            continue

        diff_content = generate_diff(tex_file, corrected)
        if not diff_content:
            continue

        patch_file = os.path.join(input_dir, f"{os.path.basename(tex_file)}.patch")
        with open(patch_file, "w") as f:
            f.write(diff_content)
        print(f"\033[32mGenerated patch file: {patch_file}\033[0m")
        patch_files.append(patch_file)

    return patch_files


def combine_patches(patch_files: list[str], output_file: str) -> None:
    """Combine multiple patch files into one"""
    print(f"\n\033[1;35mCombining {len(patch_files)} patch files...\033[0m")
    with open(output_file, "w") as outfile:
        for i, patch_file in enumerate(patch_files, 1):
            print(f"\033[33mMerging ({i}/{len(patch_files)}): {patch_file}\033[0m")
            with open(patch_file, "r") as infile:
                outfile.write(infile.read())
            os.remove(patch_file)
    print(f"\033[1;32mSuccessfully created combined patch: {output_file}\033[0m")


def main():
    parser = argparse.ArgumentParser(description="LaTeX Grammar Fixer")
    parser.add_argument("input_dir", help="Directory containing LaTeX files")
    parser.add_argument("--api-base", required=True, help="OpenAI API base URL")
    parser.add_argument("--api-key", required=True, help="OpenAI API key")
    parser.add_argument(
        "--output", default="combined.patch", help="Output patch file name"
    )
    parser.add_argument(
        "--model", default="gpt-4-turbo-preview", help="OpenAI model to use"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help="Split files into chunks of specified size (KB) for processing. 0 means no splitting",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory")
        return

    patch_files = process_directory(
        args.input_dir, args.api_base, args.api_key, args.model, args.chunk_size
    )
    combine_patches(patch_files, args.output)
    print(f"\033[1;32mSuccessfully generated combined patch file: {args.output}\033[0m")


if __name__ == "__main__":
    main()
