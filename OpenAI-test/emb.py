from openai import OpenAI

# 1. Initialize the client pointing to your local vLLM server
client = OpenAI(
    base_url="https://cjr5omlu6xka35-8000.proxy.runpod.net/v1",
    api_key="EMPTY",  # vLLM doesn't require a real key by default
)

try:
    # 2. Dynamically fetch the model name from the server API
    models_response = client.models.list()
    if not models_response.data:
        raise RuntimeError("No models found running on the vLLM server!")

    model_name = models_response.data[0].id
    print(f" Detected Model Name: {model_name}\n")

    # 3. Test text inputs
    sample_texts = [
        "What is the capital of France?",
        "Paris is the capital and most populous city of France.",
    ]

    # 4. Request embeddings
    # Note: If you want to leverage your Matryoshka dimensions (e.g., truncated to 512),
    # you can pass: dimensions=512
    response = client.embeddings.create(
        model=model_name,
        input=sample_texts,
        dimensions=1024
    )

    # 5. Output inspection
    print(f" Successfully generated {len(response.data)} embeddings.")
    for idx, item in enumerate(response.data):
        vec = item.embedding
        print(f"\n--- Text [{idx + 1}]: \"{sample_texts[idx]}\" ---")
        print(f"Embedding Vector Length (Dimensions): {len(vec)}")
        print(f"Sample values (first 5): {vec[:5]}")

    if response.usage:
        print(f"\n Prompt Tokens Used: {response.usage.prompt_tokens}")

except Exception as e:
        print(f" Error connecting to vLLM server: {e}")