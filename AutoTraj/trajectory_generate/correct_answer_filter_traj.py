import json
import re
import math
from collections import Counter
from copy import deepcopy

# ======================
# Config
# ======================
INPUT_PATH = "/to/your/path/answer_correct_trajectory.json"
HIGH_QUALITY_PATH = "/to/your/path/ac_high_quality_trajectory.json"
ANSWER_CORRECT_LOW_PATH = "/to/your/path/ac_low_quality_trajectory.json"

HIGH_QUALITY_THRESHOLD = 0.86
ALL_HIGH_THRESHOLD = 0.9   # ⭐ 全部高质量阈值

UNCERTAINTY_WORDS = [
    "maybe", "unsure", "guess", "seems", "perhaps",
    "probably", "might", "likely", "recheck"
]
UNCERTAINTY_REGEX = re.compile(
    r"\b(" + "|".join(UNCERTAINTY_WORDS) + r")\b", re.I
)

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

    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
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
# Main
# ======================
def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    #data = data[:20]   # ✅ 只测试前 20 条问答对

    high_quality = []
    correct_low_quality = []

    high_cnt = 0
    low_cnt = 0

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

        # ======================
        # ⭐ 新规则 0：全部高质量
        # ======================
        all_scores = [t["trajectory_score"] for t in scored_trajs]
        if min(all_scores) >= ALL_HIGH_THRESHOLD:
            for traj in scored_trajs:
                high_quality.append({
                    "question": question,
                    "gold_answer": gold,
                    "trajectory": traj
                })
                high_cnt += 1
            continue  # ⭐ 不再执行后续逻辑

        # ======================
        # 原有逻辑
        # ======================
        scored_trajs.sort(key=lambda x: x["trajectory_score"], reverse=True)
        best = scored_trajs[0]

        if best["trajectory_score"] >= HIGH_QUALITY_THRESHOLD:
            high_quality.append({
                "question": question,
                "gold_answer": gold,
                "trajectory": best
            })
            high_cnt += 1

            if len(scored_trajs) > 1:
                correct_low_quality.append({
                    "question": question,
                    "gold_answer": gold,
                    "trajectories": scored_trajs[1:]
                })
                low_cnt += len(scored_trajs) - 1
        else:
            correct_low_quality.append({
                "question": question,
                "gold_answer": gold,
                "trajectories": scored_trajs
            })
            low_cnt += len(scored_trajs)

    print(f"[Stage 2] High quality trajectories: {high_cnt}")
    print(f"[Stage 2] Answer-correct low quality trajectories: {low_cnt}")

    with open(HIGH_QUALITY_PATH, "w") as f:
        json.dump(high_quality, f, indent=2, ensure_ascii=False)

    with open(ANSWER_CORRECT_LOW_PATH, "w") as f:
        json.dump(correct_low_quality, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
