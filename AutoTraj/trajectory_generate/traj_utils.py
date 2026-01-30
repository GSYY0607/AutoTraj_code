import re
import math
from collections import Counter

# =====================================================
# Answer Matching (严格等价于你 Stage 1 中的实现)
# =====================================================

def is_answer_correct(gold_answer: str, traj_answer: str) -> bool:
    """
    判断轨迹答案是否正确：
    1. 从标准答案和轨迹答案中提取 {} 内内容（取最后一个）
    2. 纯数字答案进行浮点数比较
    3. 非数字答案做字符串精确匹配或子串匹配
    """

    def extract_brace_content(text: str) -> str:
        matches = re.findall(r'\{(.*?)\}', text)
        if matches:
            return matches[-1].strip()
        return text.strip()

    gold_ans = extract_brace_content(gold_answer)
    traj_ans = extract_brace_content(traj_answer)

    # ---------- 数字答案 ----------
    try:
        gold_num = float(re.sub(r'[^\d.-]', '', gold_ans))
        traj_num = float(re.sub(r'[^\d.-]', '', traj_ans))
        return abs(gold_num - traj_num) < 1e-6
    except Exception:
        # ---------- 非数字 ----------
        if gold_ans == traj_ans:
            return True
        if traj_ans in gold_ans:
            return True

    return False


# =====================================================
# Trajectory Scoring (严格等价于你 Stage 2 中的实现)
# =====================================================

UNCERTAINTY_WORDS = [
    "maybe", "unsure", "guess", "seems", "perhaps",
    "probably", "might", "likely", "recheck"
]

UNCERTAINTY_REGEX = re.compile(
    r"\b(" + "|".join(UNCERTAINTY_WORDS) + r")\b",
    re.I
)


def confidence_score(traj_text: str) -> float:
    """
    <think> 中出现任意不确定词 → 置信度 0
    否则 → 1
    """
    thinks = re.findall(r"<think>(.*?)</think>", traj_text, re.S | re.I)
    for think in thinks:
        if UNCERTAINTY_REGEX.search(think):
            return 0.0
    return 1.0


def length_score(traj_text: str, ideal_len: int) -> float:
    """
    钟形函数，ideal_len = 同一问题下最短轨迹长度
    """
    length = len(traj_text.split())
    if ideal_len <= 0:
        return 1.0

    sigma = ideal_len * 0.5
    return math.exp(-((length - ideal_len) ** 2) / (2 * sigma ** 2))


def entropy_score(traj_text: str) -> float:
    """
    3-gram 重复率
    """
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
    """
    总分：
    0.4 * 置信度
    0.3 * 长度
    0.3 * 重复率
    """
    c = confidence_score(traj_text)
    l = length_score(traj_text, ideal_len)
    e = entropy_score(traj_text)

    return 0.4 * c + 0.3 * l + 0.3 * e

if __name__ == "__main__":
    print("====== traj_utils self-test ======")

    # ---------- Test 1: Answer Matching ----------
    gold = "The answer is \\(\\boxed{175}\\)."
    traj_ok = "<answer> final is \\boxed{175} </answer>"
    traj_bad = "<answer> final is \\boxed{170} </answer>"

    print("\n[Test Answer Matching]")
    print("Correct case:", is_answer_correct(gold, traj_ok))     # True
    print("Wrong case  :", is_answer_correct(gold, traj_bad))    # False

    # ---------- Test 2: Confidence ----------
    traj_confident = "<think>We compute carefully.</think>"
    traj_uncertain = "<think>It seems the answer might be 55.</think>"

    print("\n[Test Confidence Score]")
    print("Confident:", confidence_score(traj_confident))   # 1.0
    print("Uncertain:", confidence_score(traj_uncertain))   # 0.0

    # ---------- Test 3: Length + Entropy + Total ----------
    traj_text = (
        "<think>Compute derivative step by step.</think> "
        "<answer>The result is \\boxed{55}</answer>"
    )

    ideal_len = len(traj_text.split())

    print("\n[Test Trajectory Score]")
    print("Confidence score:", round(confidence_score(traj_text), 4))
    print("Length score :", round(length_score(traj_text, ideal_len), 4))
    print("Entropy score:", round(entropy_score(traj_text), 4))
    print("Total score  :", round(trajectory_score(traj_text, ideal_len), 4))

    print("\n====== self-test finished ======")
