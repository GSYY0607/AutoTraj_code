import json
import os
import re
import multiprocessing as mp
from tqdm import tqdm

from tool_wrapper2 import ToolWrapper
from traj_utils import (
    is_answer_correct,
    trajectory_score,
)

# ======================
# Config
# ======================
INPUT_PATH = "/to/your/path/answer_wrong_low_quality_trajectory.json"

HIGH_OUT = "/to/your/path/aw_repair_high_quality_trajectory_test.json"
COMPARE_OUT = "/to/your/path/aw_repair_compare_pairs_test.json"
HARD_OUT = "/to/your/path/aw_repair_hard_cases_test.json"

MODEL_PATH = "/to/your/path/model"

SCORE_THRESHOLD = 0.8
MAX_TOOL_CALLS = 3

# ======================
# Prompt (ANSWER-WRONG REPAIR)
# ======================
SYSTEM_PROMPT = """You are a helpful assistant that solves questions step by step using tools.

You are given:
- A question
- A low-quality reasoning trajectory that leads to an INCORRECT answer

Before generating, silently analyze step by step:
- where the reasoning goes wrong
- whether the error is caused by misunderstanding, logic flaw, calculation mistake, or tool misuse
- how the mistake propagates to the final incorrect answer

Do NOT write this analysis in your output.

Then regenerate a CORRECT and high-quality reasoning trajectory from scratch.

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
11. Keep reasoning concise, logical, and correct
12. Inside <answer>, output ONLY the final answer formatted exactly as:
\boxed{final_answer}
"""

# ======================
# ReAct Generator
# ======================
class RepairGenerator:
    TOOL_PATTERN = re.compile(r"<(code|search)>(.*?)</\\1>", re.S)

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

        # 兜底
        if "<answer>" not in trajectory:
            boxed = re.findall(r"\\boxed\\{.*?\\}", trajectory)
            if boxed:
                trajectory += f"\n<answer>{boxed[-1]}</answer>"
            else:
                trajectory += "\n<answer>\\boxed{UNKNOWN}</answer>"
        elif "</answer>" not in trajectory:
            trajectory += "</answer>"

        return trajectory


# ======================
# Main
# ======================
def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 🧪 测试阶段
    data = data[:5]

    generator = RepairGenerator(MODEL_PATH)

    high_out, compare_out, hard_out = [], [], []

    # 断点续跑
    if os.path.exists(HIGH_OUT):
        high_out = json.load(open(HIGH_OUT, "r", encoding="utf-8"))
    if os.path.exists(COMPARE_OUT):
        compare_out = json.load(open(COMPARE_OUT, "r", encoding="utf-8"))
    if os.path.exists(HARD_OUT):
        hard_out = json.load(open(HARD_OUT, "r", encoding="utf-8"))

    processed = {
        (x["question"], x["trajectory"])
        for x in hard_out
    }

    for item in tqdm(data, desc="Repairing WRONG trajectories"):
        question = item["question"]
        gold = item["gold_answer"]

        for traj in item["trajectories"]:
            bad_traj = traj["react_trajectory"]

            if (question, bad_traj) in processed:
                continue

            repaired = generator.generate(question)

            correct = is_answer_correct(gold, repaired)
            #ideal_len = min(len(bad_traj.split()), len(repaired.split()))
            ideal_len = len(repaired.split())
            score = trajectory_score(repaired, ideal_len)

            if correct and score > SCORE_THRESHOLD:
                high_out.append({
                    "question": question,
                    "gold_answer": gold,
                    "trajectory": repaired,
                    "score": round(score, 4)
                })

                compare_out.append({
                    "question": question,
                    "gold_answer": gold,
                    "bad_trajectory": bad_traj,
                    "repaired_trajectory": repaired
                })
            else:
                hard_out.append({
                    "question": question,
                    "gold_answer": gold,
                    "trajectory": repaired,
                    "score": round(score, 4)
                })

            # 实时写盘
            json.dump(high_out, open(HIGH_OUT, "w", encoding="utf-8"),
                      indent=2, ensure_ascii=False)
            json.dump(compare_out, open(COMPARE_OUT, "w", encoding="utf-8"),
                      indent=2, ensure_ascii=False)
            json.dump(hard_out, open(HARD_OUT, "w", encoding="utf-8"),
                      indent=2, ensure_ascii=False)

    print("✅ Wrong-answer repair finished")
    print(f"High-quality repaired trajectories: {len(high_out)}")
    print(f"Compare pairs: {len(compare_out)}")
    print(f"Hard cases: {len(hard_out)}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
