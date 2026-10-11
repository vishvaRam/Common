import statistics
import time

from typesafe_sdk import Choice, TypeSafeClient

RUNPOD_PROXY_URL = "https://4v629awimus0xl-8000.proxy.runpod.net/"
MODEL_ID = "Vishva007/clef-flash-W4A16-AutoRound"


def run_single_test(client: TypeSafeClient, name: str, state: str, questions: dict):
    """Executes a single test case with high-precision timing."""
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {name}")
    print(f"{'=' * 60}")

    t_start = time.perf_counter()
    result = client.system_one(state=state, questions=questions)
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    print(f"Latency: {elapsed_ms:.2f} ms\n")
    for key, decision in result.choices.items():
        print(f"[{key.upper()}] -> Selected: {decision.choice} (Confidence: {decision.confidence:.2%})")
        print(f"  Probabilities: {decision.probabilities}")

    return elapsed_ms


def run_benchmark(client: TypeSafeClient, iterations: int = 2):
    """Runs a repeatable benchmark to calculate p50 and average latency."""
    test_cases = [
        {
            "name": "Invoice Request",
            "state": "Can you email me the VAT invoice for March 2024?",
            "instruction": "Route query to the appropriate team",
            "options": {"billing": "Invoices and receipts", "support": "General queries"},
        },
        {
            "name": "SDK Bug",
            "state": "The SDK throws ImportError: cannot import name 'TypeSafeClient' in Python 3.11.",
            "instruction": "Route query to the appropriate team",
            "options": {"developer_support": "SDK, API, and code errors", "billing": "Payments"},
        },
        {
            "name": "Bulk Enterprise Sales",
            "state": "We want to purchase 500 enterprise seats for our organization next quarter.",
            "instruction": "Route query to the appropriate team",
            "options": {"sales": "New licenses and bulk deals", "support": "Troubleshooting"},
        },
    ]

    print(f"\n{'=' * 60}")
    print(f"RUNNING BENCHMARK ({len(test_cases) * iterations} total runs)")
    print(f"{'=' * 60}")

    latencies = []
    run_idx = 1

    for _ in range(iterations):
        for tc in test_cases:
            t_start = time.perf_counter()
            res = client.system_one(
                state=tc["state"],
                questions={"route": Choice(instructions=tc["instruction"], criteria=tc["options"])},
            )
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            latencies.append(elapsed_ms)

            choice = res.choices["route"].choice
            conf = res.choices["route"].confidence
            print(f"Run {run_idx:02d} | [{tc['name'][:18]:<18}] | {elapsed_ms:6.2f} ms | {choice:<18} | Conf: {conf:.1%}")
            run_idx += 1

    print("\n--- Latency Summary ---")
    print(f"Mean Latency:   {statistics.mean(latencies):.2f} ms")
    print(f"Median (p50):   {statistics.median(latencies):.2f} ms")
    print(f"Min / Max:      {min(latencies):.2f} ms / {max(latencies):.2f} ms")


def main():
    with TypeSafeClient(
        api_key="EMPTY",
        base_url=RUNPOD_PROXY_URL,
        model=MODEL_ID,
    ) as client:

        # 0. Warmup run to prime HTTP session and server KV cache
        print("Warming up connection...")
        client.system_one(
            state="Ping",
            questions={"ping": Choice(instructions="Ping", criteria={"ok": "ok", "err": "err"})},
        )
        print("Warmup complete.")

        # Test Case 1: Multi-Attribute Escalation & Risk
        state_1 = """
        Ticket #8492:
        'Our production cluster went down 15 minutes ago right after we upgraded our enterprise plan. 
        None of our internal apps can connect to the database. We are losing transactions every minute, 
        and your status page shows all green! If this isn't resolved within an hour, our legal team 
        will review our SLA breach clauses.'
        """
        questions_1 = {
            "tier": Choice(
                instructions="Classify incident severity.",
                criteria={
                    "sev1_critical": "Production down, data loss, or total outage affecting business operations",
                    "sev2_major": "Core feature degraded with no immediate workaround",
                    "sev3_minor": "General issue, non-blocking bug, or standard configuration request",
                },
            ),
            "department": Choice(
                instructions="Assign primary responder team.",
                criteria={
                    "sre_infrastructure": "Host outages, hardware failures, cluster crashes, network downtime",
                    "billing_accounts": "Subscription tiers, renewals, invoice questions, and payments",
                    "legal_compliance": "Contractual dispute, terms of service breach, regulatory inquiries",
                },
            ),
            "churn_risk": Choice(
                instructions="Estimate customer churn/retention risk.",
                criteria={
                    "high": "Threatening legal action, cancellation, or expressing severe frustration",
                    "low": "Constructive, patient, or neutral communication",
                },
            ),
        }
        run_single_test(client, "Multi-Attribute Critical SLA Incident", state_1, questions_1)

        # Test Case 2: Mixed Intent (Bug causing double billing)
        state_2 = """
        'I was charged twice on my invoice because your checkout button froze and threw a 504 Gateway error.
        I clicked it again, and now my credit card statement has two $499 debits. 
        Please fix the bug on the checkout page so it doesn't happen to others, and issue a credit.'
        """
        questions_2 = {
            "primary_intent": Choice(
                instructions="Identify the primary action required for immediate resolution.",
                criteria={
                    "financial_remediation": "Priority is issuing a refund, reversing double charges, or fixing balance",
                    "engineering_triage": "Priority is filing a bug report, reviewing logs, and fixing code defects",
                },
            ),
            "requires_bug_report": Choice(
                instructions="Does this ticket warrant an engineering bug report?",
                criteria={
                    "yes": "Explicit technical bug or gateway error documented",
                    "no": "Routine billing discrepancy without software defect",
                },
            ),
        }
        run_single_test(client, "Mixed-Intent Boundary Test", state_2, questions_2)

        # Benchmark: Multi-iteration latency test
        run_benchmark(client, iterations=2)


if __name__ == "__main__":
    main()