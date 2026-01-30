import json
from pathlib import Path
from typing import List, Dict

# =========================
# File paths
# =========================
FILE_AC_REPAIR = Path("/to/your/path/ac_repair_high_quality_trajectory.json")
FILE_AW_REPAIR = Path("/to/your/path/aw_repair_high_quality_trajectory.json")
FILE_AC       = Path("/to/your/path/ac_high_quality_trajectory.json")

OUTPUT_FILE = Path("/to/your/path/sft_high_quality_trajectory.json")

# =========================
# Instruction
# =========================
INSTRUCTION = """You are a helpful assistant that can solve the given question step by step with the help of tools
like Wikipedia search and Python code execution. Given a question, you need to first think about
the reasoning process in the mind and then provide the answer. During thinking, You may invoke
the Wikipedia search tool for factual information or use Python code execution for calculation
when needed. The reasoning process is enclosed within <think> </think>, and the answer is
enclosed within <answer> </answer> tags. If Wikipedia search is used, the search query and
result are enclosed in <search> </search> and <result> </result> tags respectively. If Python
code execution is needed, the code and results are enclosed within <code> </code> and <result>
</result> tags respectively.
Example: <think> This is the reasoning process. </think> <search> search query here </search>
<result> search result here </result> <think> This is the reasoning process based on search result.
</think> <answer> The final answer is \boxed{answer here} </answer>. Or with Python code
execution: <think> This is the reasoning process. </think> <code> python code here </code>
<result> code result here </result> <think> This is the reasoning process based on code result.
</think> <answer> The final answer is \boxed{answer here} </answer>. If no tools are needed:
<think> This is the reasoning process. </think> <answer> The final answer is \boxed{answer
here} </answer>. In the last part of the answer, the final exact answer is enclosed within \boxed{}
with latex format."""

# =========================
# Utils
# =========================
def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def to_alpaca(question: str, output: str) -> Dict:
    return {
        "instruction": INSTRUCTION,
        "input": question.strip(),
        "output": output.strip(),
    }

# =========================
# Main
# =========================
def main():
    sft_data: List[Dict] = []

    # -------- ac_repair & aw_repair --------
    for path in [FILE_AC_REPAIR, FILE_AW_REPAIR]:
        for item in load_json(path):
            q = item.get("question")
            traj = item.get("trajectory")

            if q and traj:
                sft_data.append(to_alpaca(q, traj))

    # -------- ac_high_quality --------
    for item in load_json(FILE_AC):
        q = item.get("question")
        traj_obj = item.get("trajectory", {})
        react_traj = traj_obj.get("react_trajectory")

        if q and react_traj:
            sft_data.append(to_alpaca(q, react_traj))

    # -------- save --------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)

    print(f"[✓] Total SFT samples: {len(sft_data)}")
    print(f"[✓] Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
