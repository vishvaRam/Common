import base64
import os
import time

from PIL import Image
from typesafe_sdk import Box, Choice, TypeSafeClient

RUNPOD_PROXY_URL = "https://7eo9070yla4fw4-8000.proxy.runpod.net/"
MODEL_ID = "Vishva007/clef-flash-W4A16-AutoRound-GPTQ"
IMAGE_PATH = "Decision-Models-vLLM/cats.jpg"


def encode_image(image_path: str) -> str:
    """Encode an image file to a base64 Data URI."""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def print_header(title: str):
    print(f"\n{'=' * 70}\nSCENARIO: {title}\n{'=' * 70}")


def run_text_routing_and_policy(client: TypeSafeClient):
    """
    Scenario 1: Complex Multi-Task Customer Support & Safety Evaluation
    Evaluates intent, policy violation, sentiment, and urgency in one pass.
    """
    print_header("1. Complex Text Routing & Policy Enforcement")

    ticket_text = (
        "I need an immediate chargeback on invoice #INV-9021. Your software completely wiped "
        "our staging table during the last sync. If this isn't refunded by tomorrow morning, "
        "I'm filing a formal fraud complaint with our bank and escalating to corporate counsel."
    ).strip()

    questions = {
        "intent": Choice(
            instructions="Identify the primary business intent.",
            criteria={
                "refund_chargeback": "Explicit demand for money return, credit, or charge reversal",
                "technical_bug": "Report of data loss, sync failure, or system malfunction without financial demand",
                "churn_notice": "Notice of standard subscription termination",
            },
        ),
        "escalation_severity": Choice(
            instructions="Assess operational and legal risk level.",
            criteria={
                "critical": "Mentions legal action, financial penalties, bank disputes, or severe data loss",
                "moderate": "Expressions of anger or frustration without immediate legal or financial threats",
                "routine": "Calm or standard transactional inquiry",
            },
        ),
        "target_team": Choice(
            instructions="Route to the initial response unit.",
            criteria={
                "finance_disputes": "Billing disputes, refunds, and bank chargeback inquiries",
                "sre_data_recovery": "Database restoration, data corruption, and outage remediation",
                "general_support": "Standard ticketing and account maintenance",
            },
        ),
    }

    t0 = time.perf_counter()
    res = client.system_one(state=ticket_text, questions=questions)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"Latency: {elapsed:.2f} ms")
    choices = getattr(res, "choices", getattr(res, "answers", {}))
    for key, val in choices.items():
        print(f"  • {key.upper():<20} -> {val.choice:<20} (Conf: {val.confidence:.1%})")


def run_multimodal_verification_and_grounding(client: TypeSafeClient, image_path: str):
    """
    Scenario 2: Multimodal Verification & Spatial Grounding
    Pairs detection checks with bounding boxes to avoid false-positive crops.
    """
    print_header("2. Multimodal Spatial Grounding (Presence + Box)")

    if not os.path.exists(image_path):
        print(f"Skipping: {image_path} not found.")
        return

    with Image.open(image_path) as img:
        w, h = img.size

    image_uri = encode_image(image_path)
    state = {
        "image": image_uri,
        "metadata": "Living room surveillance snapshot, 1080p, low lighting."
    }

    questions = {
        # Verification check to validate target presence
        "cat_presence": Choice(
            instructions="Is there at least one cat clearly visible in `image`?",
            criteria={
                "present": "One or more cats are visible",
                "absent": "No cats are visible in the scene",
            },
        ),
        # Spatial bounding coordinates
        "primary_cat_box": Box(
            instructions="The most centrally positioned cat in `image`"
        ),
        # Scene context classification
        "indoor_scene_type": Choice(
            instructions="Classify the indoor setting.",
            criteria={
                "living_room": "Sofas, rugs, coffee tables, living space furniture",
                "kitchen": "Counters, stovetops, sinks, kitchen appliances",
                "outdoor_patio": "Decks, yards, patio furniture, gardens",
            },
        ),
    }

    t0 = time.perf_counter()
    res = client.system_one(state=state, questions=questions)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"Latency: {elapsed:.2f} ms")
    answers = getattr(res, "answers", getattr(res, "choices", {}))

    for key, val in answers.items():
        if hasattr(val, "choice"):
            print(f"  • {key.upper():<20} -> {val.choice:<20} (Conf: {val.confidence:.1%})")
        elif hasattr(val, "box") and val.box:
            b = val.box
            l = getattr(b, "left", b.get("left", 0.0) if isinstance(b, dict) else 0.0)
            t = getattr(b, "top", b.get("top", 0.0) if isinstance(b, dict) else 0.0)
            r = getattr(b, "right", b.get("right", 0.0) if isinstance(b, dict) else 0.0)
            bot = getattr(b, "bottom", b.get("bottom", 0.0) if isinstance(b, dict) else 0.0)

            px = (round(l * w), round(t * h), round(r * w), round(bot * h))
            print(f"  • {key.upper():<20} -> Box: [x1={px[0]}, y1={px[1]}, x2={px[2]}, y2={px[3]}] (Conf: {val.confidence:.1%})")


def run_e_commerce_catalog_tagging(client: TypeSafeClient):
    """
    Scenario 3: E-commerce Product Extraction & Cataloging
    Demonstrates domain-specific categorization and metadata inference.
    """
    print_header("3. Product Catalog Tagging & Taxonomy Classification")

    product_description = (
        "Men's Apex Waterproof Trail Runner. Features a Vibram Megagrip outsole, "
        "breathable Gore-Tex membrane, reinforced carbon-fiber toe cap, and zero-drop cushioning. "
        "Engineered for technical alpine terrain and heavy mud."
    )

    questions = {
        "footwear_category": Choice(
            instructions="Select the primary footwear catalog category.",
            criteria={
                "trail_running": "Off-road, technical terrain running and mountain jogging shoes",
                "road_running": "Pavement, track, and marathon lightweight running shoes",
                "hiking_boots": "High-top boots built for backpacking and ankle support",
                "lifestyle_sneakers": "Casual everyday street sneakers",
            },
        ),
        "weatherproofing": Choice(
            instructions="Determine water resistance capability.",
            criteria={
                "waterproof": "Explicitly waterproof, Gore-Tex, or sealed membrane",
                "water_resistant": "DWR coating, splash-resistant mesh",
                "none": "Standard breathable mesh with no water protection",
            },
        ),
        "experience_tier": Choice(
            instructions="Target user technical level.",
            criteria={
                "technical_expert": "Alpine, ultra-distance, reinforced protection (carbon, Vibram)",
                "recreational_entry": "Beginner friendly, everyday park walking or light trails",
            },
        ),
    }

    t0 = time.perf_counter()
    res = client.system_one(state=product_description, questions=questions)
    elapsed = (time.perf_counter() - t0) * 1000

    print(f"Latency: {elapsed:.2f} ms")
    choices = getattr(res, "choices", getattr(res, "answers", {}))
    for key, val in choices.items():
        print(f"  • {key.upper():<20} -> {val.choice:<20} (Conf: {val.confidence:.1%})")


def main():
    # Provide the Danger-Zone header if executing vision/box tasks on supported clusters
    with TypeSafeClient(
        api_key="EMPTY",
        base_url=RUNPOD_PROXY_URL,
        model=MODEL_ID,
    ) as client:

        # 0. Warmup call
        print("Priming server connection and KV cache...")
        client.system_one(
            state="Ping",
            questions={"ping": Choice(instructions="Ping", criteria={"ok": "ok", "err": "err"})},
        )
        print("Warmup complete.")

        # 1. Text Classification & Risk Routing
        run_text_routing_and_policy(client)

        # 2. Multimodal Verification & Spatial Grounding
        run_multimodal_verification_and_grounding(client, IMAGE_PATH)

        # 3. Product Catalog Tagging
        run_e_commerce_catalog_tagging(client)


if __name__ == "__main__":
    main()