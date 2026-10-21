import base64

import httpx

RUNPOD_PROXY_URL = "https://kd7xw10b2f1g59-8000.proxy.runpod.net/v1/systemone"
MODEL_ID = "Vishva007/clef-flash-W4A16-AutoRound-GPTQ"
IMAGE_PATH = "Decision-Models-vLLM/cats.jpg"


# 1. Encode image to Data URI
def to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# 2. Build the payload matching the /v1/systemone schema
payload = {
    "model": MODEL_ID,
    "state": "Examine the image and check if it contains a cat.",
    "images": [to_data_uri(IMAGE_PATH)],
    "questions": {
        "is_cat": {
            "type": "choice",
            "instructions": "Determine if a cat is clearly visible in the image.",
            "criteria": {
                "yes": "A cat is present",
                "no": "No cat is visible",
            },
        }
    },
}

# 3. Post to the endpoint
response = httpx.post(
    RUNPOD_PROXY_URL,
    json=payload,
    headers={"Authorization": "Bearer EMPTY"},
    timeout=60.0,
)

print(response.status_code)
print(response.json())