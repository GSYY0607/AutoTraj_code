import json
import os
import re
import multiprocessing as mp
from copy import deepcopy
from tqdm import tqdm

from tool_wrapper2 import ToolWrapper
from traj_utils import (
    is_answer_correct,
    trajectory_score,
)

# ======================
# Config
# ======================
INPUT_PATH = "/to/your/path/ac_low_quality_trajectory.json"

HIGH_OUT = "/to/your/path/ac_repaired_high_quality_trajectory_test.json"
COMPARE_OUT = "/to/your/path/ac_repair_compare_pairs_test.json"
HARD_OUT = "/to/your/path/ac_repair_hard_cases_test.json"

MODEL_PATH = "/to/your/path/model"

SCORE_THRESHOLD = 0.86
MAX_TOOL_CALLS = 3

# ======================
# Prompt
# ======================
SYSTEM_PROMPT = """You are a helpful assistant that solves questions step by step using tools.
You are given:
- A question
- A low-quality reasoning trajectory that reaches the correct answer

Before generating, silently analyze whether the old trajectory contains:
- redundant reasoning
- unclear logic
- vague or uncertain words (e.g., seems, maybe, perhaps, probably)
Do NOT write this analysis in your output.

Then regenerate a high-quality reasoning trajectory from scratch.

REQUIREMENTS:
1. The FIRST non-whitespace token MUST be exactly <think>
2. ALL reasoning MUST appear ONLY inside <think>...</think>
3. You MUST NOT reveal reasoning outside <think> tags
4. You MAY use tools ONLY if necessary, and AT MOST 3 tool calls
5. Use tools ONLY when needed. If Wikipedia search is used, the search query and
result are enclosed in <search> </search> and <result> </result> tags respectively. If Python
code execution is needed, the code and results are enclosed within <code> </code> and <result>
</result> tags respectively.
6. You MUST reason again inside <think> after each <result>
7. You MUST produce EXACTLY ONE <answer>...</answer> block
8. The <answer> block MUST be the FINAL output. After </answer>, output NOTHING
9. You MUST NOT output multiple <answer> blocks
10. Do NOT use uncertain words like seems, maybe, perhaps, probably, might, likely
11. Keep reasoning concise, logical, and clear
12.Inside <answer>, output ONLY the final answer formatted exactly as:\boxed{final_answer}
"""

# ======================
# ReAct Generator
# ======================
class RepairGenerator:
    TOOL_PATTERN = re.compile(r"<(code|search)>(.*?)</\1>", re.S)

    def __init__(self, model_path):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=4,
            dtype="bfloat16",
            trust_remote_code=True,
            disable_custom_all_reduce=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
            stop=["</answer>"]
        )
        self.tool_wrapper = ToolWrapper()

    def _build_prompt(self, messages):
        prompt = ""
        for m in messages:
            if m["role"] == "system":
                prompt += m["content"] + "\n\n"
            elif m["role"] == "user":
                prompt += "Question:\n" + m["content"] + "\n\n"
            else:
                prompt += m["content"] + "\n"
        return prompt

    def generate(self, question):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        trajectory = ""
        tool_calls = 0

        for _ in range(10):
            prompt = self._build_prompt(messages)
            out = self.llm.generate([prompt], self.sampling_params)[0]
            text = out.outputs[0].text

            trajectory += text
            messages.append({"role": "assistant", "content": text})

            if "</answer>" in text:
                break

            if tool_calls >= MAX_TOOL_CALLS:
                break

            match = self.TOOL_PATTERN.search(text)
            if match:
                tool_calls += 1
                tool, content = match.groups()
                result = self.tool_wrapper.call_tool(tool, content.strip())
                result_block = f"<result>\n{result}\n</result>\n"
                trajectory += result_block
                messages.append({"role": "assistant", "content": result_block})

        if "<answer>" not in trajectory:
            boxed = re.findall(r"\\boxed\{.*?\}", trajectory)
            if boxed:
                trajectory += f"\n<answer>{boxed[-1]}</answer>"
            else:
                trajectory += "\n<answer>Unable to determine.</answer>"
        elif "</answer>" not in trajectory:
            trajectory += "</answer>"

        return trajectory


# ======================
# Main
# ======================
def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 🧪 测试：只跑前 5 条问题
    data = data[:5]

    generator = RepairGenerator(MODEL_PATH)

    high_out = []
    compare_out = []
    hard_out = []

    # 断点续跑
    if os.path.exists(HIGH_OUT):
        with open(HIGH_OUT, "r", encoding="utf-8") as f:
            high_out = json.load(f)
    if os.path.exists(COMPARE_OUT):
        with open(COMPARE_OUT, "r", encoding="utf-8") as f:
            compare_out = json.load(f)
    if os.path.exists(HARD_OUT):
        with open(HARD_OUT, "r", encoding="utf-8") as f:
            hard_out = json.load(f)

    # 使用 react_trajectory 作为唯一标识，避免 unhashable dict 错误
    processed = {
        (x["question"], x["trajectory"]["react_trajectory"])
        for x in hard_out
    }

    for item in tqdm(data, desc="Repairing trajectories"):
        question = item["question"]
        gold = item["gold_answer"]

        for traj in item["trajectories"]:
            traj_text = traj["react_trajectory"]
            orig_score = traj.get("trajectory_score", 0.0)  # 获取原始轨迹分数，默认0.0

            if (question, traj_text) in processed:
                continue

            repaired = generator.generate(question)

            correct = is_answer_correct(gold, repaired)
            #ideal_len = len(traj_text.split())
            ideal_len = min(len(traj_text.split()), len(repaired.split()))
            score = trajectory_score(repaired, ideal_len)

            if correct and score > SCORE_THRESHOLD and score > orig_score:
                high_out.append({
                    "question": question,
                    "gold_answer": gold,
                    "trajectory": repaired,
                    "score": round(score, 4)
                })

                compare_out.append({
                    "question": question,
                    "gold_answer": gold,
                    "bad_trajectory": traj_text,
                    "repaired_trajectory": repaired
                })
            else:
                hard_out.append({
                    "question": question,
                    "gold_answer": gold,
                    "trajectory": repaired,
                    "score": round(score, 4)
                })

            # ✅ 实时写盘
            with open(HIGH_OUT, "w", encoding="utf-8") as f:
                json.dump(high_out, f, indent=2, ensure_ascii=False)
            with open(COMPARE_OUT, "w", encoding="utf-8") as f:
                json.dump(compare_out, f, indent=2, ensure_ascii=False)
            with open(HARD_OUT, "w", encoding="utf-8") as f:
                json.dump(hard_out, f, indent=2, ensure_ascii=False)

    # 打印各个数据集轨迹数量
    print("✅ Repair finished")
    print(f"High-quality repaired trajectories: {len(high_out)}")
    print(f"Compare pairs: {len(compare_out)}")
    print(f"Hard cases: {len(hard_out)}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
