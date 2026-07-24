import base64
from openai import OpenAI

# -----------------------------
# 1. Initialize Client
# -----------------------------
client = OpenAI(
    base_url="https://zkje1hzy73h1c5-6006.proxy.runpod.net/v1",
    api_key="empty"
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
# Helper: Stream printer
# -----------------------------
def stream_response(stream):

    for chunk in stream:

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        if hasattr(delta, "reasoning") and delta.reasoning:
            print(delta.reasoning, end="", flush=True)

        if hasattr(delta, "content") and delta.content:
            print(delta.content, end="", flush=True)

    print("\n")


# =====================================================
# 1️⃣ THINKING CALL — Deep Vision Analysis
# =====================================================

print("=========== THINKING VISION ANALYSIS ===========\n")

stream1 = client.chat.completions.create(
    model=model_name,
    stream=True,
    max_tokens=2000,

    messages=[
        {
            "role": "system",
            "content": "You are a highly capable vision analysis AI."
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
    ],

    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": True
        }
    }
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

    messages=[
        {
            "role": "system",
            "content": "You are an expert OCR and document analysis system."
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
    ],

    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": True
        }
    }
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

    messages=[
        {
            "role": "system",
            "content": "You generate short image captions."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Write a concise one-sentence caption describing this image."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],

    extra_body={
        "chat_template_kwargs": {
            "enable_thinking": False
        }
    }
)

stream_response(stream3)

print("=========== ALL STREAMS COMPLETE ===========")