import time
import base64
from openai import OpenAI

# -----------------------------
# Function to encode image
# -----------------------------
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# -----------------------------
# Initialize Client
# -----------------------------
client = OpenAI(
    base_url="https://v5li5nb34ewlq6-8000.proxy.runpod.net/v1",
    api_key="token-abc123"
)

# -----------------------------
# Get Model
# -----------------------------
models = client.models.list()
model_name = models.data[0].id
print(f"\nUsing Model: {model_name}\n")

# -----------------------------
# Stream Printer + Metrics Calculator
# -----------------------------
def stream_response(stream, start_time):
    first_token_time = None
    completion_tokens = 0
    prompt_tokens = 0

    for chunk in stream:
        # Capture exact token counts from the final usage chunk
        if getattr(chunk, "usage", None) is not None:
            completion_tokens = chunk.usage.completion_tokens
            prompt_tokens = chunk.usage.prompt_tokens
            continue

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        if hasattr(delta, "content") and delta.content is not None:
            # Capture Time to First Token (TTFT)
            if first_token_time is None:
                first_token_time = time.time()
            
            text = delta.content
            print(text, end="", flush=True)

    end_time = time.time()
    
    # Calculate Performance Metrics
    ttft = first_token_time - start_time if first_token_time else 0
    total_time = end_time - start_time
    generation_time = end_time - first_token_time if first_token_time else total_time
    throughput = completion_tokens / generation_time if generation_time > 0 else 0
    tpot = (generation_time / completion_tokens) * 1000 if completion_tokens > 0 else 0

    print("\n\n")
    print("====== VLM Inference Metrics ======")
    print(f"Prompt Tokens (includes image): {prompt_tokens}")
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
# MULTIMODAL LONG GENERATION
# =====================================================

print("=========== VLM LONG GENERATION ===========\n")

# 1. Encode the local image
image_path = "./nature.jpg"
base64_image = encode_image(image_path)

# Record start time
start_time = time.time()

# 2. Update the payload for vision
stream = client.chat.completions.create(
    model=model_name,
    stream=True,
    max_tokens=6000,
    stream_options={"include_usage": True}, 
    messages=[
        {
            "role": "system",
            "content": (
                "You are an expert environmental scientist and technical writer "
                "capable of producing very detailed long-form analyses."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "Write a VERY detailed 5-page technical article analyzing this image. "
                        "Expand extensively on the geographical features, ecological systems, "
                        "lighting, composition, and visible flora/fauna. Do not summarize early. "
                        "Continue expanding ideas thoroughly."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
)

stream_response(stream, start_time)

print("=========== TEST COMPLETE ===========")
