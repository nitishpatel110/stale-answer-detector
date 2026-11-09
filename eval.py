"""Benchmark: python eval.py  -> precision/recall/accuracy/latency on data/test_cases.json.
IMPORTANT: facts change. Re-verify every label before you present numbers, and extend to 50-100 cases."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

from detector import check_answer


def run_case(case):
    try:
        res = check_answer(case["question"], case["answer"])
        pred = "stale" if res["verdict"] == "stale" else "not_stale"
        return case, pred, res["latency_s"], res["verdict"]
    except Exception as e:
        return case, "error", 0, str(e)


def main():
    cases = json.load(open("data/test_cases.json"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(run_case, cases))

    tp = fp = fn = tn = 0
    lat = []
    for case, pred, latency, raw in rows:
        truth = "stale" if case["label"] == "stale" else "not_stale"
        if pred == "error":
            print(f"#{case['id']} ERROR {raw}")
            continue
        lat.append(latency)
        tp += truth == "stale" and pred == "stale"
        fn += truth == "stale" and pred != "stale"
        fp += truth != "stale" and pred == "stale"
        tn += truth != "stale" and pred != "stale"
        mark = "OK " if truth == pred else "BAD"
        print(f"{mark} #{case['id']:>3} truth={case['label']:<6} got={raw:<9} {latency}s  {case['question']}")

    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    print("\n=== RESULTS ===")
    print(f"Cases: {total} | Accuracy: {(tp + tn) / total:.0%}" if total else "No results")
    print(f"Stale precision: {precision:.0%} | Stale recall: {recall:.0%} | False-positive rate: {fp / (fp + tn):.0%}" if fp + tn else "")
    if lat:
        print(f"Avg latency: {sum(lat) / len(lat):.1f}s")


if __name__ == "__main__":
    main()
