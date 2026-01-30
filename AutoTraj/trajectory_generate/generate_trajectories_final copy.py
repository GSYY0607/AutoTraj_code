import json
import os
import re
from typing import List, Dict
import multiprocessing as mp
from tqdm import tqdm
from tool_wrapper2 import ToolWrapper

SYSTEM_PROMPT = """You are a helpful assistant that can solve the given question step by step with the help of tools
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

class ReActTrajectoryGenerator:
    TOOL_PATTERN = re.compile(r"<(code|search)>(.*?)</\1>", re.S)

    def __init__(self, model_path: str):
        from vllm import LLM, SamplingParams
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=4,
            dtype="bfloat16",
            trust_remote_code=True,
            disable_custom_all_reduce=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0.3,
            top_p=0.9,
            max_tokens=1024,
            stop=["</answer>"]
        )
        self.tool_wrapper = ToolWrapper()

    def _build_prompt(self, messages: List[Dict]) -> str:
        prompt = ""
        for m in messages:
            if m["role"] == "system":
                prompt += f"{m['content']}\n\n"
            elif m["role"] == "user":
                prompt += f"Question:\n{m['content']}\n\n"
            else:
                prompt += f"{m['content']}\n"
        return prompt

    def generate_one(self, question: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trajectory = ""
        tool_call_count = 0
        max_tool_calls = 3

        for _ in range(10):
            prompt = self._build_prompt(messages)
            out = self.llm.generate([prompt], self.sampling_params)[0]
            text = out.outputs[0].text
            trajectory += text
            messages.append({"role": "assistant", "content": text})

            if "</answer>" in text:
                break
            if tool_call_count >= max_tool_calls:
                break

            match = self.TOOL_PATTERN.search(text)
            if match:
                tool_call_count += 1
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
        else:
            if "</answer>" not in trajectory:
                trajectory += "</answer>"
        return trajectory

    def generate_dataset(self, data: List[Dict], num_traj=1, output_path=None):
        results = []

        # 断点续跑：读取已有输出
        if output_path and os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed_questions = {item["question"] for item in results}
            print(f"🟢 Resuming from existing file, {len(processed_questions)} questions already processed.")
        else:
            processed_questions = set()

        for item in tqdm(data, desc="Generating dataset", unit="question"):
            question = item["question"]
            gold = item.get("gold_answer") or item.get("answer")

            if question in processed_questions:
                continue  # 跳过已生成

            trajectories = []
            for tid in range(num_traj):
                traj = self.generate_one(question)
                trajectories.append({
                    "trajectory_id": tid,
                    "react_trajectory": traj
                })

            results.append({
                "question": question,
                "gold_answer": gold,
                "trajectories": trajectories
            })

            # 实时写入 OUTPUT
            if output_path:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

        return results


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    INPUT = "/to/your/path/input_qa_pairs.json"
    OUTPUT = "/to/your/path/generated_trajectory.json"
    MODEL = "/to/your/path/model"

    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    gen = ReActTrajectoryGenerator(MODEL)

    # 开始生成
    results = gen.generate_dataset(data, num_traj=2, output_path=OUTPUT)

    print(f"✅ Finished generating {len(results)} questions to {OUTPUT}")
