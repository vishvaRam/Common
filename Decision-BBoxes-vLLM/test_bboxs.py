import base64
import json
import os
import time

import requests
from PIL import Image, ImageDraw

IMAGE_PATH = "Decision-BBoxes-vLLM/cats.jpg"
OUTPUT_PATH = "Decision-BBoxes-vLLM/cat_detected.jpg"
BASE_URL = "http://localhost:8000"

# 1. Verify image exists
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Cannot find image: '{IMAGE_PATH}'")

# Start total execution timer
total_start = time.perf_counter()

# 2. Auto-detect active model
t0 = time.perf_counter()
try:
    models_res = requests.get(f"{BASE_URL}/v1/models")
    models_res.raise_for_status()
    served_models = [m["id"] for m in models_res.json()["data"]]
    model_name = served_models[0]
    print(f"Connected to vLLM. Serving model: {model_name}")
except Exception as e:
    model_name = "Vishva007/Qwen3.5-4B-W4A16-AutoRound-GPTQ"
    print(f"Could not auto-detect model ({e}). Using: {model_name}")
print(f"Model lookup took: {(time.perf_counter() - t0):.3f}s\n")

# 3. Read and base64-encode the image
t0 = time.perf_counter()
with open(IMAGE_PATH, "rb") as f:
    encoded_image = base64.b64encode(f.read()).decode("utf-8")

ext = os.path.splitext(IMAGE_PATH)[1].lower().replace(".", "")
mime_type = "jpeg" if ext in ["jpg", "jpeg"] else ext
image_data_url = f"data:image/{mime_type};base64,{encoded_image}"
print(f"Image encoding took: {(time.perf_counter() - t0) * 1000:.2f}ms")

# 4. Multi-instance payload for /v1/systemone
payload = {
    "model": model_name,
    "state": "Detect and locate every cat in the image individually.",
    "images": [image_data_url],
    "questions": {
        "cat_1": {
            "type": "box",
            "instructions": "the cat on the left or primary cat"
        },
        "cat_2": {
            "type": "box",
            "instructions": "the cat on the right or secondary cat"
        },
        "cat_3": {
            "type": "box",
            "instructions": "any other distinct cat in the frame"
        }
    },
    "chat_template_kwargs": {"enable_thinking": False}
}

# 5. Measure server inference request latency
print("Sending request to /v1/systemone...")
inference_start = time.perf_counter()
response = requests.post(f"{BASE_URL}/v1/systemone", json=payload)
inference_duration = time.perf_counter() - inference_start

if not response.ok:
    print(f"\n[HTTP {response.status_code} Error from vLLM]:")
    print(response.text)
    response.raise_for_status()

result = response.json()
print(f"Server inference completed in: {inference_duration:.3f}s\n")

# 6. Pretty print response JSON
print("=== Server JSON Response ===")
print(json.dumps(result, indent=2))
print("============================\n")

# 7. Coordinate parsing & post-processing timing
t0 = time.perf_counter()
img = Image.open(IMAGE_PATH).convert("RGB")
width, height = img.size
draw = ImageDraw.Draw(img)

palette = ["red", "lime", "cyan", "yellow", "magenta", "orange"]
valid_boxes = []

answers = result.get("answers", {})

for idx, (qid, ans) in enumerate(answers.items()):
    box = ans.get("box") if isinstance(ans, dict) else None
    if not box or not isinstance(box, dict):
        continue

    left_norm = box.get("left", 0.0)
    top_norm = box.get("top", 0.0)
    right_norm = box.get("right", 0.0)
    bottom_norm = box.get("bottom", 0.0)

    # Discard non-detections or inverted coordinate predictions
    if right_norm <= left_norm or bottom_norm <= top_norm:
        continue

    # Discard duplicate overlapping boxes
    is_duplicate = False
    for prev in valid_boxes:
        pb = prev["norm"]
        if (
            abs(left_norm - pb["left"]) < 0.03
            and abs(top_norm - pb["top"]) < 0.03
            and abs(right_norm - pb["right"]) < 0.03
            and abs(bottom_norm - pb["bottom"]) < 0.03
        ):
            is_duplicate = True
            break

    if is_duplicate:
        continue

    valid_boxes.append({
        "qid": qid,
        "norm": {"left": left_norm, "top": top_norm, "right": right_norm, "bottom": bottom_norm},
        "color": palette[len(valid_boxes) % len(palette)]
    })

# 8. Draw boxes onto image
for item in valid_boxes:
    b = item["norm"]
    qid = item["qid"]
    color = item["color"]

    left = b["left"] * width
    top = b["top"] * height
    right = b["right"] * width
    bottom = b["bottom"] * height

    draw.rectangle([left, top, right, bottom], outline=color, width=4)
    tag = f"{qid} ({b['left']:.2f}, {b['top']:.2f})"
    draw.rectangle([left, max(0, top - 20), left + (len(tag) * 8), top], fill=color)
    draw.text((left + 3, max(0, top - 18)), tag, fill="black")

    print(f"[{qid}] Coordinates: left={left:.1f}, top={top:.1f}, right={right:.1f}, bottom={bottom:.1f}")

# 9. Save output image
if valid_boxes:
    img.save(OUTPUT_PATH)
    print(f"\nSaved {len(valid_boxes)} detected bounding box(es) to: {OUTPUT_PATH}")
else:
    print("\nNo valid bounding boxes detected.")

postprocess_duration = time.perf_counter() - t0
total_duration = time.perf_counter() - total_start

print(f"Annotation & rendering took: {postprocess_duration * 1000:.2f}ms")
print(f"Total script runtime: {total_duration:.3f}s")


# root@e3c90bbabb78:/workspace# uv run test_bboxs.py 
# Connected to vLLM. Serving model: Vishva007/Qwen3.5-4B-W4A16-AutoRound-GPTQ
# Model lookup took: 0.003s

# Image encoding took: 1.52ms
# Sending request to /v1/systemone...
# Server inference completed in: 1.039s

# === Server JSON Response ===
# {
#   "id": "decision-b5d39eef985329d9",
#   "object": "structured_decision",
#   "created": 1794210418,
#   "model": "Vishva007/Qwen3.5-4B-W4A16-AutoRound-GPTQ",
#   "answers": {
#     "cat_1": {
#       "type": "box",
#       "box": {
#         "left": 0.0,
#         "top": 0.432,
#         "right": 0.333,
#         "bottom": 0.595
#       },
#       "confidence": null
#     },
#     "cat_2": {
#       "type": "box",
#       "box": {
#         "left": 0.663,
#         "top": 0.246,
#         "right": 0.845,
#         "bottom": 0.378
#       },
#       "confidence": null
#     },
#     "cat_3": {
#       "type": "box",
#       "box": {
#         "left": 0.663,
#         "top": 0.246,
#         "right": 0.845,
#         "bottom": 0.378
#       },
#       "confidence": null
#     }
#   },
#   "usage": {
#     "input_tokens": 6529,
#     "output_tokens": 78
#   },
#   "diagnostics": {},
#   "joint": null
# }
# ============================

# [cat_1] Coordinates: left=0.0, top=829.4, right=359.6, bottom=1142.4
# [cat_2] Coordinates: left=716.0, top=472.3, right=912.6, bottom=725.8

# Saved 2 detected bounding box(es) to: cat_detected.jpg
# Annotation & rendering took: 31.92ms
# Total script runtime: 1.077s