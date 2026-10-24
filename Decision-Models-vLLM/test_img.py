import base64
import json
import time

import httpx

RUNPOD_PROXY_URL = "http://213.173.105.26:18278"
MODEL_ID = "Vishva007/Qwen3.5-9B-NVFP4"
IMAGE_PATH = "Decision-Models-vLLM/cats.jpg"


def to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


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

start_time = time.perf_counter()

response = httpx.post(
    f"{RUNPOD_PROXY_URL}/v1/systemone",
    json=payload,
    headers={"Authorization": "Bearer EMPTY"},
    timeout=60.0,
)

elapsed_time = time.perf_counter() - start_time

print(f"Status Code: {response.status_code}")
print(f"Elapsed Time: {elapsed_time:.3f} seconds\n")

try:
    data = response.json()
    print("Response JSON:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
except Exception:
    print("Response Text:")
    print(response.text)