import json
import re
import string
from collections import Counter

# ======================================================
# Answer Extraction (shared)
# ======================================================
def extract_last_answer_block(react_trajectory: str) -> str:
    """提取最后一个 <answer>...</answer> 中的内容"""
    blocks = re.findall(r"<answer>(.*?)</answer>", react_trajectory, re.DOTALL)
    return blocks[-1] if blocks else ""


def extract_boxed_answer(answer_block: str) -> str:
    """提取 \boxed{} 或 ed{} 中的内容"""
    match = re.findall(r"(?:\\?boxed|ed)\{(.*?)\}", answer_block, re.DOTALL)
    return match[-1].strip() if match else answer_block.strip()


def extract_core_answer(s: str) -> str:
    """提取 {} 或 \boxed{} / ed{} 中的核心内容"""
    match = re.search(r"(?:\\?boxed|ed)?\{(.*?)\}", s, re.DOTALL)
    return match.group(1).strip() if match else s.strip()


# ======================================================
# Math: Enhanced Exact & Numeric Match
# ======================================================
def clean_latex(s: str) -> str:
    """去掉LaTeX命令和多余符号"""
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "").replace(" ", "")
    return s


def extract_numbers(s: str) -> list:
    """提取字符串中的所有数字（整数或浮点）"""
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", s)]


def is_math_correct(gold_answer: str, react_trajectory: str, tol: float = 1e-6) -> bool:
    """
    数学题判断逻辑：
    1. 提取 gold_answer {} 中内容
    2. 提取 pred \boxed{} 中内容
    3. 去掉 LaTeX 符号先精确匹配
    4. 如果精确匹配失败，则提取数字进行匹配（允许浮点误差）
    5. 支持多选答案（顺序无关）
    """
    pred_block = extract_last_answer_block(react_trajectory)
    pred = extract_boxed_answer(pred_block)
    gold_core = extract_core_answer(gold_answer)

    # 1️⃣ 去 LaTeX 清理精确匹配
    if clean_latex(pred) == clean_latex(gold_core):
        return True

    # 2️⃣ 提取数字
    gold_nums = extract_numbers(gold_core)
    pred_nums = extract_numbers(pred)

    if len(gold_nums) == len(pred_nums):
        if all(abs(a - b) <= tol for a, b in zip(sorted(gold_nums), sorted(pred_nums))):
            return True

    # 3️⃣ 多选答案处理（字母或数字，顺序无关）
    gold_set = set(x.strip().lower() for x in re.split(r"[,\s]+", gold_core) if x)
    pred_set = set(x.strip().lower() for x in re.split(r"[,\s]+", pred) if x)
    return gold_set == pred_set


# ======================================================
# QA: Normalization & F1 Match
# ======================================================
def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def is_qa_correct(
    gold_answer: str,
    react_trajectory: str,
    f1_threshold: float = 0.5
) -> bool:
    """
    QA 判断：
    1. 精确匹配（忽略大小写和空格）
    2. F1 匹配（文本相似度 >= f1_threshold）
    """
    pred_block = extract_last_answer_block(react_trajectory)
    pred = extract_boxed_answer(pred_block)
    gold_core = extract_core_answer(gold_answer)
    pred_core = extract_core_answer(pred)

    if pred_core.strip().lower() == gold_core.strip().lower():
        return True

    return f1_score(pred_core, gold_core) >= f1_threshold


# ======================================================
# Unified Trajectory Classifier
# ======================================================
def classify_trajectories(
    input_file,
    correct_file,
    incorrect_file,
    task_type: str,
    f1_threshold: float = 0.5
):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    correct_data, incorrect_data = [], []
    correct_cnt, incorrect_cnt = 0, 0

    for item in data:
        gold_answer = item["gold_answer"]
        trajectories = item.get("trajectories", [])
        results = []

        for traj in trajectories:
            if task_type == "math":
                ok = is_math_correct(gold_answer, traj["react_trajectory"])
            elif task_type == "qa":
                ok = is_qa_correct(gold_answer, traj["react_trajectory"], f1_threshold)
            else:
                raise ValueError(f"Unknown task_type: {task_type}")
            results.append(ok)

        correct_cnt += sum(results)
        incorrect_cnt += len(results) - sum(results)

        if all(results):
            correct_data.append(item)
        elif not any(results):
            incorrect_data.append(item)
        else:
            # 部分正确的拆分
            for i, ok in enumerate(results):
                single = {
                    "question": item["question"],
                    "gold_answer": gold_answer,
                    "trajectories": [trajectories[i]]
                }
                (correct_data if ok else incorrect_data).append(single)

    with open(correct_file, "w", encoding="utf-8") as f:
        json.dump(correct_data, f, ensure_ascii=False, indent=2)

    with open(incorrect_file, "w", encoding="utf-8") as f:
        json.dump(incorrect_data, f, ensure_ascii=False, indent=2)

    total = correct_cnt + incorrect_cnt
    acc = correct_cnt / total if total > 0 else 0.0

    return {
        "total_traj": total,
        "correct_traj": correct_cnt,
        "incorrect_traj": incorrect_cnt,
        "accuracy": acc
    }


# ======================================================
# Batch Processing Entry
# ======================================================
if __name__ == "__main__":

    DATASETS = [
        # ---------- Math ----------
        {
            "name": "GSM8K",
            "task_type": "math",
            "input": "/to/your/path/gsm8k_test_autotraj.json",
            "correct": "/to/your/path/gsm8k_test_ac_autotraj.json",
            "incorrect": "/to/your/path/gsm8k_test_aw_autotraj.json",
        },
        {
            "name": "AIME24",
            "task_type": "math",
            "input": "/to/your/path/aime24_autotraj.json",
            "correct": "/to/your/path/aime24_ac_autotraj.json",
            "incorrect": "/to/your/path/aime24_aw_autotraj.json",
        },
        {
            "name": "AIME25",
            "task_type": "math",
            "input": "/to/your/path/aime25_autotraj.json",
            "correct": "/to/your/path/aime25_ac_autotraj.json",
            "incorrect": "/to/your/path/aime25_aw_autotraj.json",
        },
        {
            "name": "math",
            "task_type": "math",
            "input": "/to/your/path/math_autotraj.json",
            "correct": "/to/your/path/math_ac_autotraj.json",
            "incorrect": "/to/your/path/math_aw_autotraj.json",
        },
        {
            "name": "amc23",
            "task_type": "math",
            "input": "/to/your/path/amc23_autotraj.json",
            "correct": "/to/your/path/amc23_ac_autotraj.json",
            "incorrect": "/to/your/path/amc23_aw_autotraj.json",
        },

        # ---------- QA ----------
        {
            "name": "2wiki",
            "task_type": "qa",
            "input": "/to/your/path/2wiki_autotraj.json",
            "correct": "/to/your/path/2wiki_ac_autotraj.json",
            "incorrect": "/to/your/path/2wiki_aw_autotraj.json",
        },
        {
            "name": "hotpotqa",
            "task_type": "qa",
            "input": "/to/your/path/hotpotqa_autotraj.json",
            "correct": "/to/your/path/hotpotqa_ac_autotraj.json",
            "incorrect": "/to/your/path/hotpotqa_aw_autotraj.json",
        },
        {
            "name": "musique",
            "task_type": "qa",
            "input": "/to/your/path/musique_autotraj.json",
            "correct": "/to/your/path/musique_ac_autotraj.json",
            "incorrect": "/to/your/path/musique_aw_autotraj.json",
        },
        {
            "name": "hle",
            "task_type": "qa",
            "input": "/to/your/path/hle_autotraj.json",
            "correct": "/to/your/path/hle_ac_autotraj.json",
            "incorrect": "/to/your/path/hle_aw_autotraj.json",
        },
    ]

    print("\n📊 Trajectory Classification Summary\n" + "-" * 60)

    for cfg in DATASETS:
        print(f"\n🚀 Processing {cfg['name']} ({cfg['task_type']})")

        stats = classify_trajectories(
            input_file=cfg["input"],
            correct_file=cfg["correct"],
            incorrect_file=cfg["incorrect"],
            task_type=cfg["task_type"],
            f1_threshold=0.5
        )

        print(
            f"✅ {cfg['name']} | "
            f"Total: {stats['total_traj']} | "
            f"Correct: {stats['correct_traj']} | "
            f"Wrong: {stats['incorrect_traj']} | "
            f"Accuracy: {stats['accuracy']:.4f}"
        )
