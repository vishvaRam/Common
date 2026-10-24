import json
import time

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

RUNPOD_PROXY_URL = "http://213.173.105.26:18278"
MODEL_ID = "Vishva007/Qwen3.5-9B-NVFP4"

client = TypeSafeClient(
    api_key="Empty",
    base_url=RUNPOD_PROXY_URL,
)

ticket = """
I was charged twice for my subscription this month.
Please refund the duplicate payment. This is urgent because
my rent payment is due tomorrow.
"""

start_time = time.perf_counter()

result = client.system_one(
    model=MODEL_ID,
    state=ticket,
    questions={
        # 1. NOUL: Boolean / statement truth verification
        "refund_requested": Noul(
            instructions="Is the customer explicitly requesting a refund?",
        ),
        # 2. SCORE: Rate on an ordered scale
        "urgency": Score(
            instructions="How urgent is this customer's issue?",
            criteria=[
                "0 - Not urgent; can wait several weeks",
                "1 - Low urgency; can wait a few days",
                "2 - Moderate urgency; should be resolved soon",
                "3 - High urgency; requires prompt attention",
                "4 - Critical urgency; immediate action is needed",
            ],
        ),
        # 3. CHOICE: Categorical selection
        "category": Choice(
            instructions="Which department should handle this ticket?",
            criteria={
                "billing": "Duplicate charges, payments, refunds",
                "technical": "Software bugs and technical failures",
                "account": "Login, profile, and account access",
                "other": "Issues that do not fit the other categories",
            },
        ),
    },
)

elapsed_time = time.perf_counter() - start_time

# Extract dictionary representation from result (or fallback to vars/dict mapping)
result_data = result.model_dump()

print(f"Elapsed Time: {elapsed_time:.3f} seconds\n")
print("Response JSON:")
print(json.dumps(result_data, indent=2, default=str, ensure_ascii=False))