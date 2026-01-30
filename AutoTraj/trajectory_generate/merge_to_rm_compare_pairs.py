import json
from pathlib import Path

# ========= 路径配置 =========
input_files = [
    "/to/your/path/ac_repair_compare_pairs.json",
    "/to/your/path/aw_repair_compare_pairs.json",
]

output_file = "/to/your/path/rm_compare_pairs.json"

# ========= 固定 instruction =========
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

# ========= 合并逻辑 =========
merged_data = []

for file_path in input_files:
    file_path = Path(file_path)
    assert file_path.exists(), f"File not found: {file_path}"

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for idx, sample in enumerate(data):
        # 基本字段校验
        if not all(k in sample for k in ["question", "bad_trajectory", "repaired_trajectory"]):
            raise ValueError(
                f"Missing keys in {file_path} at index {idx}: {sample.keys()}"
            )

        rm_item = {
            "instruction": INSTRUCTION,
            "input": sample["question"].strip(),
            "chosen": sample["repaired_trajectory"].strip(),
            "rejected": sample["bad_trajectory"].strip(),
        }

        merged_data.append(rm_item)

# ========= 保存 =========
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=2)

print(f"✅ Merged {len(merged_data)} samples")
print(f"📄 Saved to: {output_file}")
