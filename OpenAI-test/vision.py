import base64
import time

from openai import OpenAI

# -----------------------------
# 1. Initialize Client
# -----------------------------
client = OpenAI(
    base_url="https://k4afg71fpofwvl-8000.proxy.runpod.net/v1",
    api_key="empty",
)

# -----------------------------
# 2. Encode Image
# -----------------------------
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


IMAGE_PATH = "OpenAI-test/img.jpeg"
base64_image = encode_image(IMAGE_PATH)

# -----------------------------
# 3. Get Model
# -----------------------------
models = client.models.list()
model_name = models.data[0].id
print(f"\nUsing Model: {model_name}\n")


# -----------------------------
# Helper: Stream Printer + Metrics Tracker
# -----------------------------
def stream_response(stream):
    first_token_time = None
    start_time = time.time()

    prompt_tokens = 0
    completion_tokens = 0
    server_usage_found = False
    fallback_token_count = 0

    thinking_header_shown = False
    content_header_shown = False

    for chunk in stream:
        # 1. Capture stream usage stats if returned by the server (e.g., vLLM stream_options)
        if hasattr(chunk, "usage") and chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens or 0
            completion_tokens = chunk.usage.completion_tokens or 0
            server_usage_found = True

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        # 2. Track reasoning tokens (thinking phase)
        reasoning = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None)
        if reasoning:
            if first_token_time is None:
                first_token_time = time.time()
            if not thinking_header_shown:
                print("--- [Thinking / Reasoning] ---")
                thinking_header_shown = True

            print(reasoning, end="", flush=True)
            fallback_token_count += 1

        # 3. Track visible content tokens
        if hasattr(delta, "content") and delta.content:
            if first_token_time is None:
                first_token_time = time.time()
            if not content_header_shown:
                if thinking_header_shown:
                    print("\n\n--- [Final Output] ---")
                content_header_shown = True

            print(delta.content, end="", flush=True)
            fallback_token_count += 1

    end_time = time.time()

    # If backend didn't emit a usage object, fall back to streamed chunk count
    if not server_usage_found or completion_tokens == 0:
        completion_tokens = fallback_token_count

    # 4. Calculate Performance Metrics
    ttft = (first_token_time - start_time) if first_token_time else 0.0
    total_time = end_time - start_time
    generation_time = (end_time - first_token_time) if first_token_time else total_time
    throughput = (completion_tokens / generation_time) if generation_time > 0 else 0.0
    tpot = ((generation_time / completion_tokens) * 1000) if completion_tokens > 0 else 0.0

    print("\n\n" + "=" * 36)
    print("        INFERENCE METRICS")
    print("=" * 36)
    print(f"Prompt (Prefill) Tokens:        {prompt_tokens if prompt_tokens > 0 else 'N/A'}")
    print(f"Output (Decode) Tokens:         {completion_tokens}")
    print(f"Total Tokens:                   {prompt_tokens + completion_tokens if prompt_tokens > 0 else completion_tokens}")
    print("-" * 36)
    print(f"Time To First Token (TTFT):     {ttft:.3f} s")
    print(f"Decode / Generation Time:       {generation_time:.3f} s")
    print(f"Total Request Time:             {total_time:.3f} s")
    print("-" * 36)
    print(f"TPOT (Time Per Output Token):   {tpot:.2f} ms/token")
    print(f"Throughput:                     {throughput:.2f} tokens/s")
    print("=" * 36 + "\n")


# =====================================================
# 1️⃣ THINKING CALL — Deep Vision Analysis
# =====================================================

print("=========== THINKING VISION ANALYSIS ===========\n")

stream1 = client.chat.completions.create(
    model=model_name,
    stream=True,
    max_tokens=2000,
    stream_options={"include_usage": True},
    messages=[
        {
            "role": "system",
            "content": "You are a highly capable vision analysis AI.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Carefully analyze this image. Think step-by-step about objects, "
                        "people, environment, colors, and context. Then provide a detailed explanation "
                        "of what is happening in the image."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)

stream_response(stream1)


# =====================================================
# 2️⃣ THINKING CALL — OCR + Structured Understanding
# =====================================================

print("=========== THINKING OCR + TEXT ANALYSIS ===========\n")

stream2 = client.chat.completions.create(
    model=model_name,
    stream=True,
    max_tokens=2000,
    stream_options={"include_usage": True},
    messages=[
        {
            "role": "system",
            "content": "You are an expert OCR and document analysis system.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Inspect this image carefully and extract any visible text. "
                        "Think step-by-step while scanning different parts of the image. "
                        "List all readable text and explain where it appears."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ],
    extra_body={"chat_template_kwargs": {"enable_thinking": True}},
)

stream_response(stream2)


# =====================================================
# 3️⃣ NON-THINKING CALL — Fast Caption
# =====================================================

print("=========== FAST CAPTION (NO THINKING) ===========\n")

stream3 = client.chat.completions.create(
    model=model_name,
    stream=True,
    max_tokens=500,
    stream_options={"include_usage": True},
    messages=[
        {
            "role": "system",
            "content": "You generate short image captions.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Write a concise one-sentence caption describing this image.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ],
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

stream_response(stream3)

print("=========== ALL STREAMS COMPLETE ===========")