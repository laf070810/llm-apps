import re

# Conditional OpenAI import
try:
    import openai
except ImportError:
    openai = None

# Conditional Ollama import
try:
    import ollama
except ImportError:
    ollama = None

# Conditional dify_client import
try:
    import dify_client
except ImportError:
    dify_client = None


def get_llm_response(
    *,
    api_type,
    api_base,
    api_key,
    model,
    prompt,
    system_message="",
    remove_thinking=False,
    **api_options,
):
    print("\033[2mStreaming response...\033[0m")
    chunk_response = []

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
                    "content": prompt,
                },
            ],
            stream=True,
            **api_options,
        )

        # Process response chunks
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                chunk_response.append(content)

    elif api_type == "ollama":
        if ollama is None:
            raise ImportError(
                "ollama package is required for Ollama API. Install with: pip install ollama"
            )

        client = ollama.Client(host=api_base)

        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            stream=True,
            **api_options,
        )

        # Process response chunks
        for chunk in response:
            content = chunk["message"]["content"]
            if content:
                print(content, end="", flush=True)
                chunk_response.append(content)

    elif api_type == "dify":
        if dify_client is None:
            raise ImportError(
                "dify_client package is required for Dify API. Install with: pip install dify-client"
            )
        import json

        chat_client = dify_client.ChatClient(api_key)
        chat_client.base_url = api_base

        chat_response = chat_client.create_chat_message(
            inputs={}, query=prompt, user="1", response_mode="streaming"
        )
        chat_response.raise_for_status()

        for line in chat_response.iter_lines(decode_unicode=True):
            if not line.startswith("data:"):
                continue

            content = None

            line = line.split("data:", 1)[-1]
            try:
                if line:
                    json_chunk = json.loads(line)
                    content = json_chunk.get("answer")
            except (json.JSONDecodeError, KeyError) as e:
                print(
                    f"\n\033[31mError parsing response: {str(e)}. The response is: {line}\033[0m"
                )
                continue

            if content:
                print(content, end="", flush=True)
                chunk_response.append(content)

    else:
        raise NotImplementedError(f"unknown API type: {api_type}")

    combined_response = "".join(chunk_response)
    if remove_thinking:
        return re.sub(r"<think>.*?</think>", "", combined_response, flags=re.DOTALL)
    else:
        return combined_response
