import time
from openai import OpenAI

# -----------------------------
# Initialize Client
# -----------------------------
client = OpenAI(
    base_url="https://bz6gk79748uazb-8000.proxy.runpod.net/v1",
    api_key="token-abc123"
)

# -----------------------------
# Get Model
# -----------------------------
models = client.models.list()
model_name = models.data[0].id
print(f"\nUsing Model: {model_name}\n")


# -----------------------------
# Stream Printer + Token Counter
# -----------------------------
def stream_response(stream):

    token_count = 0
    start = time.time()

    for chunk in stream:

        if not chunk.choices:
            continue

        delta = chunk.choices[0].delta

        if hasattr(delta, "content") and delta.content:
            text = delta.content
            token_count += len(text.split())
            print(text, end="", flush=True)

    end = time.time()

    print("\n")
    print("------ Throughput Stats ------")
    print(f"Approx Tokens: {token_count}")
    print(f"Time Taken: {round(end-start,2)} sec")
    print(f"Approx Tokens/sec: {round(token_count/(end-start),2)}")
    print("------------------------------\n")


# =====================================================
# LONG ARTICLE GENERATION
# =====================================================

print("=========== LONG ARTICLE GENERATION ===========\n")

stream = client.chat.completions.create(
    model=model_name,
    stream=True,
    max_tokens=6000,   # allow very long output

    messages=[
        {
            "role": "system",
            "content": (
                "You are an expert technical writer capable of producing "
                "very detailed long-form articles."
            )
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
            )
        }
    ]
)

stream_response(stream)

print("=========== TEST COMPLETE ===========")