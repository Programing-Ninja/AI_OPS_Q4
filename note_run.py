import argparse
import json
import platform

import torch
from mlflow.tracking import MlflowClient

TOLERANCE = 1e-4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True, help="Partner A's run id")
    p.add_argument("--your-accuracy", type=float, required=True)
    p.add_argument("--comment", default="", help="explanation if it did not match")
    args = p.parse_args()

    with open("metrics.json") as fh:
        ref = json.load(fh)

    delta = abs(args.your_accuracy - ref["test_accuracy"])
    matched = delta <= TOLERANCE

    verdict = (
        f"REPRODUCTION {'MATCHED' if matched else 'DID NOT MATCH'}\n"
        f"reference test_accuracy: {ref['test_accuracy']:.6f}\n"
        f"reproduced test_accuracy: {args.your_accuracy:.6f}\n"
        f"absolute difference: {delta:.8f}\n"
        f"stated tolerance: {TOLERANCE}\n"
        f"reproducer platform: {platform.platform()}\n"
        f"reproducer torch: {torch.__version__}\n"
    )
    if args.comment:
        verdict += f"\nnotes: {args.comment}\n"

    client = MlflowClient()
    client.set_tag(args.run_id, "reproduction_status", "matched" if matched else "mismatch")
    client.set_tag(args.run_id, "reproduced_accuracy", f"{args.your_accuracy:.6f}")
    client.set_tag(args.run_id, "reproduction_delta", f"{delta:.8f}")
    client.update_run(args.run_id, description=verdict)

    print(verdict)


if __name__ == "__main__":
    main()
