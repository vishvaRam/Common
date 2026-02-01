from openai import OpenAI

# 1. Initialize the client
# vLLM doesn't require a real API key by default, but the client needs a string.
client = OpenAI(
    base_url="https://v40rhrnrine990-8000.proxy.runpod.net/v1",
    api_key="token-abc123" 
)

# 2. List available models (Useful to verify connection)
models = client.models.list()
model_name = models.data[0].id
print(f"Testing model: {model_name}\n")

# 3. Chat Completion Test (Non-streaming)
print("--- Chat Completion ---")
chat_response = client.chat.completions.create(
    model=model_name,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum physics in one sentence."},
    ],
    temperature=0.7
)
print(chat_response.choices[0].message.content)

# 4. Streaming Test
print("\n--- Streaming Response ---")
stream = client.chat.completions.create(
    model=model_name,
    messages=[{"role": "user", "content": "Write a short poem about coding."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)