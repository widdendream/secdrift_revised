import argparse
import csv
import glob
import json
import random
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_file", help="Path to results jsonl")
    ap.add_argument("-n", type=int, default=100, help="Number of samples")
    ap.add_argument("-o", default="manual_validation_sample.csv", help="Output csv path")
    args = ap.parse_args()

    rows = []
    with open(args.results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if len(rows) == 0:
        raise SystemExit("No rows found in results file")

    random.seed(1337)
    sample = random.sample(rows, k=min(args.n, len(rows)))

    out_path = Path(args.o)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "evaluation_id",
        "timestamp",
        "model",
        "provider",
        "scenario_id",
        "cwe",
        "prompt_type",
        "sector",
        "replicate",
        "is_vulnerable",
        "vulnerabilities",
        "detection_tool",
        "confidence",
        "latency_ms",
        "prompt_text",
        "generated_code",
        "human_label_is_vuln",
        "human_notes",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        w = csv.DictWriter(csvfile, fieldnames=fieldnames)
        w.writeheader()
        for r in sample:
            w.writerow({
                "evaluation_id": r.get("evaluation_id"),
                "timestamp": r.get("timestamp"),
                "model": r.get("model"),
                "provider": r.get("provider"),
                "scenario_id": r.get("scenario_id"),
                "cwe": r.get("cwe"),
                "prompt_type": r.get("prompt_type"),
                "sector": r.get("sector"),
                "replicate": r.get("replicate"),
                "is_vulnerable": r.get("is_vulnerable"),
                "vulnerabilities": ";".join(r.get("vulnerabilities") or []),
                "detection_tool": r.get("detection_tool"),
                "confidence": r.get("confidence"),
                "latency_ms": r.get("latency_ms"),
                "prompt_text": r.get("prompt_text"),
                "generated_code": r.get("generated_code"),
                "human_label_is_vuln": "",
                "human_notes": "",
            })

    print(f"Wrote {len(sample)} samples to {out_path}")

if __name__ == "__main__":
    main()