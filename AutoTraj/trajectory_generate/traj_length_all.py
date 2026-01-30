import json

# ======================
# Config
# ======================
DATASETS = [
    {
        "name": "GSM8K",
        "input": "/to/your/path/gsm8k_test_autotraj.json",
    },
    {
        "name": "AIME24",
        "input": "/to/your/path/aime24_autotraj.json",
    },
    {
        "name": "AIME25",
        "input": "/to/your/path/aime25_autotraj.json",
    },
    {
        "name": "math",
        "input": "/to/your/path/math_autotraj.json",
    },
    {
        "name": "amc23",
        "input": "/to/your/path/amc23_autotraj.json",
    },
    {
        "name": "2wiki",
        "input": "/to/your/path/2wiki_autotraj.json",
    },
    {
        "name": "hotpotqa",
        "input": "/to/your/path/hotpotqa_autotraj.json",
    },
    {
        "name": "musique",
        "input": "/to/your/path/musique_autotraj.json",
    },
    {
        "name": "hle",
        "input": "/to/your/path/hle_autotraj.json",
    },
]

# ======================
# Length Statistics
# ======================
def process_dataset(cfg):
    with open(cfg["input"], "r", encoding="utf-8") as f:
        data = json.load(f)

    total_length = 0
    total_count = 0

    for item in data:
        trajs = item["trajectories"]
        for traj in trajs:
            length = len(traj["react_trajectory"].split())
            total_length += length
            total_count += 1

    avg_length = total_length / total_count if total_count > 0 else 0.0

    return {
        "dataset": cfg["name"],
        "avg_length": avg_length,
        "count": total_count,
        "total_length": total_length,
    }

# ======================
# Entry
# ======================
if __name__ == "__main__":
    print("\n📏 Trajectory Length Statistics")
    print("-" * 70)

    all_total_length = 0
    all_total_count = 0

    results = []

    for cfg in DATASETS:
        stats = process_dataset(cfg)
        results.append(stats)

        all_total_length += stats["total_length"]
        all_total_count += stats["count"]

        print(
            f"{stats['dataset']:<12} | "
            f"Trajectories: {stats['count']:<6} | "
            f"Avg length: {stats['avg_length']:.2f}"
        )

    print("-" * 70)
    overall_avg = all_total_length / all_total_count if all_total_count > 0 else 0.0

    print(
        f"ALL DATASETS | "
        f"Trajectories: {all_total_count} | "
        f"Avg length: {overall_avg:.2f}"
    )
