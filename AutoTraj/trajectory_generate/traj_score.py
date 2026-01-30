import json
import re
import math
from collections import Counter
from copy import deepcopy

# ======================
# Config
# ======================
HIGH_QUALITY_THRESHOLD = 0.86
ALL_HIGH_THRESHOLD = 0.9

UNCERTAINTY_WORDS = [
    "maybe", "unsure", "guess", "seems", "perhaps",
    "probably", "might", "likely", "recheck"
]
UNCERTAINTY_REGEX = re.compile(
    r"\b(" + "|".join(UNCERTAINTY_WORDS) + r")\b", re.I
)

DATASETS = [
    {
        "name": "GSM8K",
        "input": "/to/your/path/gsm8k_test_ac_autotraj.json",
        "high_out": "/to/your/path/gsm8k_test_ac_hq_autotraj.json",
        "low_out": "/to/your/path//gsm8k_test_ac_lq_autotraj.json",
    },
    {
        "name": "AIME24",
        "input": "/to/your/path/aime24_ac_autotraj.json",
        "high_out": "/to/your/path/aime24_ac_hq_autotraj.json",
        "low_out": "/to/your/path/aime24_ac_lq_autotraj.json",
    },
    {
        "name": "AIME25",
        "input": "/to/your/path/aime25_ac_autotraj.json",
        "high_out": "/to/your/path/aime25_ac_hq_autotraj.json",
        "low_out": "/to/your/path/aime25_ac_lq_autotraj.json",
    },
    {
        "name": "math",
        "input": "/to/your/path/math_ac_autotraj.json",
        "high_out": "/to/your/path/math_ac_hq_autotraj.json",
        "low_out": "/to/your/path/math_ac_lq_autotraj.json",
    },
    {
        "name": "amc23",
        "input": "/to/your/path/amc23_ac_autotraj.json",
        "high_out": "/to/your/path/amc23_ac_hq_autotraj.json",
        "low_out": "/to/your/path/amc23_ac_lq_autotraj.json",
    },
    {
        "name": "2wiki",
        "input": "/to/your/path/2wiki_ac_autotraj.json",
        "high_out": "/to/your/path/2wiki_ac_hq_autotraj.json",
        "low_out": "/to/your/path/2wiki_ac_lq_autotraj.json",
    },
    {
        "name": "hotpotqa",
        "input": "/to/your/path/hotpotqa_ac_autotraj.json",
        "high_out": "/to/your/path/hotpotqa_ac_hq_autotraj.json",
        "low_out": "/to/your/path/hotpotqa_ac_lq_autotraj.json",
    },
    {
        "name": "musique",
        "input": "/to/your/path/musique_ac_autotraj.json",
        "high_out": "/to/your/path/musique_ac_hq_autotraj.json",
        "low_out": "/to/your/path/musique_ac_lq_autotraj.json",
    },
    {
        "name": "hle",
        "input": "/to/your/path/hle_ac_autotraj.json",
        "high_out": "/to/your/path/hle_ac_hq_autotraj.json",
        "low_out": "/to/your/path/hle_ac_lq_autotraj.json",
    },
    # 👉 后续数据集只需继续加
]

# ======================
# Scoring Functions
# ======================
def confidence_score(traj_text: str) -> float:
    thinks = re.findall(r"<think>(.*?)</think>", traj_text, re.S | re.I)
    for think in thinks:
        if UNCERTAINTY_REGEX.search(think):
            return 0.0
    return 1.0


def length_score(traj_text: str, ideal_len: int) -> float:
    length = len(traj_text.split())
    if ideal_len <= 0:
        return 1.0
    sigma = ideal_len * 0.5
    return math.exp(-((length - ideal_len) ** 2) / (2 * sigma ** 2))


def entropy_score(traj_text: str) -> float:
    words = traj_text.split()
    if len(words) < 6:
        return 1.0

    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    counter = Counter(trigrams)

    repeated = sum(c for c in counter.values() if c > 1)
    total = len(trigrams)

    repetition_rate = repeated / total
    return max(0.0, 1.0 - repetition_rate)


def trajectory_score(traj_text: str, ideal_len: int) -> float:
    c = confidence_score(traj_text)
    l = length_score(traj_text, ideal_len)
    e = entropy_score(traj_text)
    return 0.4 * c + 0.3 * l + 0.3 * e

# ======================
# Main Processing (Single Dataset)
# ======================
def process_dataset(cfg):
    with open(cfg["input"], "r", encoding="utf-8") as f:
        data = json.load(f)

    high_quality = []
    correct_low_quality = []

    # ⭐ 统计用
    sum_high, cnt_high = 0.0, 0
    sum_low, cnt_low = 0.0, 0
    sum_all, cnt_all = 0.0, 0

    for item in data:
        question = item["question"]
        gold = item["gold_answer"]
        trajs = item["trajectories"]

        lengths = [len(t["react_trajectory"].split()) for t in trajs]
        ideal_len = min(lengths)

        scored_trajs = []
        for traj in trajs:
            score = trajectory_score(traj["react_trajectory"], ideal_len)
            new_traj = deepcopy(traj)
            new_traj["trajectory_score"] = round(score, 4)
            scored_trajs.append(new_traj)

            sum_all += score
            cnt_all += 1

        # ⭐ 规则 0：全部高质量
        all_scores = [t["trajectory_score"] for t in scored_trajs]
        if min(all_scores) >= ALL_HIGH_THRESHOLD:
            for traj in scored_trajs:
                high_quality.append({
                    "question": question,
                    "gold_answer": gold,
                    "trajectory": traj
                })
                sum_high += traj["trajectory_score"]
                cnt_high += 1
            continue

        # 原有逻辑
        scored_trajs.sort(key=lambda x: x["trajectory_score"], reverse=True)
        best = scored_trajs[0]

        if best["trajectory_score"] >= HIGH_QUALITY_THRESHOLD:
            high_quality.append({
                "question": question,
                "gold_answer": gold,
                "trajectory": best
            })
            sum_high += best["trajectory_score"]
            cnt_high += 1

            for traj in scored_trajs[1:]:
                correct_low_quality.append({
                    "question": question,
                    "gold_answer": gold,
                    "trajectories": [traj]
                })
                sum_low += traj["trajectory_score"]
                cnt_low += 1
        else:
            for traj in scored_trajs:
                correct_low_quality.append({
                    "question": question,
                    "gold_answer": gold,
                    "trajectories": [traj]
                })
                sum_low += traj["trajectory_score"]
                cnt_low += 1

    # 保存结果
    with open(cfg["high_out"], "w", encoding="utf-8") as f:
        json.dump(high_quality, f, indent=2, ensure_ascii=False)

    with open(cfg["low_out"], "w", encoding="utf-8") as f:
        json.dump(correct_low_quality, f, indent=2, ensure_ascii=False)

    return {
        "high_avg": sum_high / cnt_high if cnt_high else 0.0,
        "low_avg": sum_low / cnt_low if cnt_low else 0.0,
        "all_avg": sum_all / cnt_all if cnt_all else 0.0,
        "high_cnt": cnt_high,
        "low_cnt": cnt_low,
        "all_cnt": cnt_all,
    }

# ======================
# Entry
# ======================
if __name__ == "__main__":
    print("\n📊 Stage-2 Trajectory Quality Statistics")
    print("-" * 70)

    for cfg in DATASETS:
        print(f"\n🚀 Dataset: {cfg['name']}")
        stats = process_dataset(cfg)

        print(
            f"High-quality avg score: {stats['high_avg']:.4f} "
            f"(count={stats['high_cnt']})"
        )
        print(
            f"Low-quality  avg score: {stats['low_avg']:.4f} "
            f"(count={stats['low_cnt']})"
        )
        print(
            f"All trajectories avg:   {stats['all_avg']:.4f} "
            f"(count={stats['all_cnt']})"
        )
