import base64
from openai import OpenAI

# 1. Initialize the Client
client = OpenAI(
    base_url="https://0nx83tzoswfrvi-8000.proxy.runpod.net/v1",
    api_key="token-abc123"
)

# Function to encode local image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your local image
IMAGE_PATH = "./img.jpeg"

# 2. Get Model Name
models = client.models.list()
model_name = models.data[0].id
print(f"Using Vision Model: {model_name}\n")

# 3. Vision Chat Completion
print("--- Vision Analysis ---")

# Encode the local image
base64_image = encode_image(IMAGE_PATH)

response = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is in this image? Describe it in detail."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        # Standard format for base64 images in OpenAI API
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    max_tokens=300
)

print(response.choices[0].message.content)

# 4. Streaming Vision Test (Optional)
print("\n--- Streaming Vision Response ---")

stream = client.chat.completions.create(
    model=model_name,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract any text visible in this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                }
            ]
        }
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n")
