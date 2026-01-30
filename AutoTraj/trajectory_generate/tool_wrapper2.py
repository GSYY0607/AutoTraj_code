import sys
import os
import requests
import json

# 添加工具所在目录到Python路径
sys.path.append('/to/your/path/tools')

# 导入工具类
from calculator import CalculatorTool
from python_interpreter import PythonExecutor

class ToolWrapper:
    def __init__(self):
        # 初始化工具实例
        self.calculator = CalculatorTool()
        self.python_executor = PythonExecutor(get_answer_from_stdout=True)
        self.flashrag_url = "your url/search"
    
    def calculate(self, expression: str) -> str:
        """调用计算器工具执行数学表达式"""
        try:
            value, result_str = self.calculator.evaluate(expression)
            return f"{result_str}"
        except Exception as e:
            return f"Error: {e}"
    
    def python_execute(self, code: str) -> str:
        """调用Python解释器执行代码"""
        try:
            result, report = self.python_executor.apply(code)
            if report == "Done":
                return result
            else:
                return f"Error: {report}"
        except Exception as e:
            return f"Error: {e}"
    
    def websearch(self, query: str) -> str:
        """调用FlashRAG服务进行检索"""
        try:
            payload = {
                "query": query,
                "top_n": 1,
                "return_score": True
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(self.flashrag_url, json=payload, headers=headers, timeout=30)

            if response.status_code != 200:
                return f"Error: HTTP {response.status_code}"

            data = response.json()

            # FlashRAG DPR mode: [docs_list, scores_list]
            if (
                isinstance(data, list)
                and len(data) == 2
                and isinstance(data[0], list)
                and isinstance(data[1], list)
            ):
                docs = data[0]
                scores = data[1]

                formatted = []
                for i, (doc, score) in enumerate(zip(docs, scores)):
                    contents = doc.get("contents", "")[:200].replace("\n", " ")
                    doc_id = doc.get("id", "unknown")
                    formatted.append(f"[{i+1}] (ID: {doc_id}) {contents}... (score: {score:.4f})")

                return "\n".join(formatted)

            return f"Unrecognized FlashRAG return format: {data}"

        except Exception as e:
            return f"Error: {e}"
    
    def call_tool(self, tool_type: str, tool_input: str) -> str:
        """统一的工具调用接口"""
        if tool_type == "calculator":
            return self.calculate(tool_input)
        elif tool_type == "code":
            return self.python_execute(tool_input)
        elif tool_type == "search":
            return self.websearch(tool_input)
        else:
            return f"Error: Unsupported tool type '{tool_type}'"

# 测试代码
if __name__ == "__main__":
    tool_wrapper = ToolWrapper()
    
    # 测试计算器
    print("Calculator test:")
    print(tool_wrapper.calculate("2+3*4"))
    
    # 测试Python解释器
    print("\nPython test:")
    test_code = "print(2+3*4)"
    print(tool_wrapper.python_execute(test_code))
    
    # 测试WebSearch
    print("\nWebSearch test:")
    print(tool_wrapper.websearch("What is machine learning?"))
