import base64

import requests

RUNPOD_PROXY_URL = "https://7eo9070yla4fw4-8000.proxy.runpod.net/v1/systemone"
MODEL_ID = "Vishva007/clef-flash-W4A16-AutoRound-GPTQ"
IMAGE_PATH = "Decision-Models-vLLM/cats.jpg"

with open(IMAGE_PATH, "rb") as f:
    b64_img = base64.b64encode(f.read()).decode("utf-8")

payload = {
    "model": MODEL_ID,
    "state": {"photo": "cats_asset"},
    "assets": {
        "cats_asset": {
            "type": "image",
            "data": b64_img
        }
    },
    "questions": {
        "cat_primary": {
            "type": "box",
            "instructions": "the cat in `photo`"
        }
    }
}

headers = {
    "Content-Type": "application/json",
    "TypeSafe-Danger-Zone-Images": "true"
}

resp = requests.post(RUNPOD_PROXY_URL, json=payload, headers=headers)
print("HTTP Status:", resp.status_code)
print("Server Diagnostic Response:", resp.text)