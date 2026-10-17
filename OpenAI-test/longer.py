import time

from openai import OpenAI

# -----------------------------
# Initialize Client
# -----------------------------
client = OpenAI(
    base_url="https://3s6oczsbq8l8ce-8000.proxy.runpod.net/v1", 
    api_key="empty")

# -----------------------------
# Get Model
# -----------------------------
models = client.models.list()
model_name = models.data[0].id
print(f"\nUsing Model: {model_name}\n")


# -----------------------------
# Stream Printer + Metrics Tracker
# -----------------------------
def stream_response(stream):
    first_token_time = None
    completion_tokens = 0
    prompt_tokens = 0
    start_time = time.time()

    is_thinking = False
    thought_printed_header = False
    content_printed_header = False

    for chunk in stream:
        # Capture usage stats if returned
        if hasattr(chunk, "usage") and chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens
            completion_tokens = chunk.usage.completion_tokens

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # Extract delta attributes
        reasoning_text = getattr(delta, "reasoning_content", None) or getattr(
            delta, "reasoning", None
        )
        content_text = getattr(delta, "content", None)

        # ---------------------------------------------------------
        # Case A: Server sends reasoning in reasoning_content field
        # ---------------------------------------------------------
        if reasoning_text:
            if first_token_time is None:
                first_token_time = time.time()
            if not thought_printed_header:
                print("--- [THINKING PROCESS] ---\n", flush=True)
                thought_printed_header = True

            print(reasoning_text, end="", flush=True)

            if completion_tokens == 0 or not hasattr(chunk, "usage") or not chunk.usage:
                completion_tokens += 1
            continue

        # ---------------------------------------------------------
        # Case B: Server sends reasoning inside <think> tags in content
        # ---------------------------------------------------------
        if content_text:
            if first_token_time is None:
                first_token_time = time.time()

            # Handle start of thinking block
            if "<think>" in content_text:
                is_thinking = True
                if not thought_printed_header:
                    print("--- [THINKING PROCESS] ---\n", flush=True)
                    thought_printed_header = True
                content_text = content_text.replace("<think>", "")

            # Handle end of thinking block
            if "</think>" in content_text:
                parts = content_text.split("</think>")
                print(parts[0], end="", flush=True)
                print("\n\n--- [FINAL ANSWER] ---\n", flush=True)
                is_thinking = False
                content_printed_header = True
                content_text = parts[1]

            # Print current stream slice
            if is_thinking:
                print(content_text, end="", flush=True)
            else:
                if thought_printed_header and not content_printed_header:
                    print("\n\n--- [FINAL ANSWER] ---\n", flush=True)
                    content_printed_header = True
                print(content_text, end="", flush=True)

            if completion_tokens == 0 or not hasattr(chunk, "usage") or not chunk.usage:
                completion_tokens += 1

    end_time = time.time()

    # Performance Metrics
    ttft = (first_token_time - start_time) if first_token_time else 0
    total_time = end_time - start_time
    generation_time = (end_time - first_token_time) if first_token_time else total_time
    throughput = completion_tokens / generation_time if generation_time > 0 else 0
    tpot = (generation_time / completion_tokens) * 1000 if completion_tokens > 0 else 0

    print("\n\n")
    print("====== Inference Metrics ======")
    print(
        f"Prompt Tokens:                  {prompt_tokens if prompt_tokens else 'N/A'}"
    )
    print(f"Output Tokens:                  {completion_tokens}")
    print(f"Total Tokens:                   {prompt_tokens + completion_tokens}")
    print("------------------------------------")
    print(f"Time To First Token (Prefill):  {ttft:.3f} sec")
    print(f"Generation Time (Decode):       {generation_time:.3f} sec")
    print(f"Total Request Time:             {total_time:.3f} sec")
    print("------------------------------------")
    print(f"TPOT:                           {tpot:.2f} ms/token")
    print(f"Output Throughput:              {throughput:.2f} tokens/sec")
    print("====================================\n")


# =====================================================
# LONG ARTICLE GENERATION
# =====================================================

print("=========== LONG ARTICLE GENERATION ===========\n")

stream = client.chat.completions.create(
    model=model_name,
    stream=True,
    max_tokens=10000,  # allow very long output
    stream_options={
        "include_usage": True
    },  # Requests exact token counts from vLLM/OpenAI
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking_token_budget": 256,
    },
    messages=[
        {
            "role": "system",
            "content": (
                "You are an expert technical writer capable of producing "
                "very detailed long-form articles."
            ),
        },
        {
            "role": "user",
            "content": (
                """
Write a VERY detailed 5-page technical article.

Topic:
The Evolution of Large Language Models and Their Impact on Artificial Intelligence

Requirements:

- Minimum length: 3500 words
- Use clear sections and subsections
- Include deep explanations
- Include examples
- Include technical insights
- Include historical context
- Include modern architectures (Transformers, MoE, multimodal models)
- Include future research directions
- Include practical applications

Structure:

1. Introduction
2. History of Language Models
3. Neural Language Models
4. Transformer Architecture
5. Scaling Laws and Model Training
6. Multimodal Models
7. Applications of LLMs
8. Challenges and Limitations
9. Future of AI
10. Conclusion

Write in **academic article style**.

Expand each section extensively with detailed explanations.
Do not summarize early. Continue expanding ideas thoroughly.
"""
            ),
        },
    ],
)

stream_response(stream)

print("=========== TEST COMPLETE ===========")

