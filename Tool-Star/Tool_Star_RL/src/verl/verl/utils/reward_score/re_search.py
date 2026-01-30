import re
import sys
import string
import requests
from typing import Union, List
from collections import Counter
from transformers import AutoTokenizer


def validate_format(text: str) -> tuple[bool, str]:
    # check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"
    
    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"
    
    #if text.count('<answer>') != 1 or text.count('</answer>') != 1:
    if text.count('<answer>') == 0 or text.count('</answer>') == 0:
        return False, "<answer> or </answer> not found"        
    
    # check the order of search/result
    current_pos = 0
    while True:
        search_pos = text.find('<search>', current_pos)
        if search_pos == -1:
            break
            
        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</search>', search_pos)
        result_end_pos = text.find('</result>', result_pos)
        
        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "search/result tags are incomplete"
            
        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "search/result tags are nested in the wrong order"
            
        current_pos = result_end_pos
    
    # check the order of python/result
    current_pos = 0
    while True:
        python_pos = text.find('<code>', current_pos)
        if python_pos == -1:
            break
            
        result_pos = text.find('<result>', python_pos)
        python_end_pos = text.find('</code>', python_pos)
        result_end_pos = text.find('</result>', result_pos)
        
        if -1 in (result_pos, python_end_pos, result_end_pos):
            return False, "python/result tags are incomplete"
            
        if not (python_pos < python_end_pos < result_pos < result_end_pos):
            return False, "python/result tags are nested in the wrong order"
            
        current_pos = result_end_pos
    
    # check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"
    
    return True, "format is correct"



def validate_format_python(text: str) -> tuple[bool, str]:
    # check if <think></think>, <answer></answer> is paired
    if text.count('<think>') != text.count('</think>'):
        return False, "<think> </think> not paired"
    
    if text.count('<think>') == 0 or text.count('</think>') == 0:
        return False, "<think> or </think> not found"
    
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return False, "<answer> or </answer> not found"        
    
    # check the order of search/result
    current_pos = 0
    while True:
        search_pos = text.find('<code>', current_pos)
        if search_pos == -1:
            break
            
        result_pos = text.find('<result>', search_pos)
        search_end_pos = text.find('</code>', search_pos)
        result_end_pos = text.find('</result>', result_pos)
        
        if -1 in (result_pos, search_end_pos, result_end_pos):
            return False, "python/result tags are incomplete"
            
        if not (search_pos < search_end_pos < result_pos < result_end_pos):
            return False, "python/result tags are nested in the wrong order"
            
        current_pos = result_end_pos
    
    # check if \boxed{} is in the answer
    answer_start = text.find('<answer>')
    answer_end = text.find('</answer>')
    if answer_start > answer_end:
        return False, "<answer> must be before </answer>"
    answer_content = text[answer_start:answer_end]
    if '\\boxed{' not in answer_content or '}' not in answer_content:
        return False, "answer is missing \\boxed{} format"
    
    return True, "format is correct"

def extract_answer(text: str):
    text = text.strip()

    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    
    return match.group(1)

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[:len(left)] == left
        return s[len(left):]

    left = "\\boxed{"

    assert s[:len(left)] == left
    assert s[-1] == "}"

    return s[len(left):-1]

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_f1_score(prediction: str, ground_truths: Union[str, List[str]]):
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    
    final_metric = {"f1": 0, "precision": 0, "recall": 0}

    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)

        if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue
        
        if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
            continue

        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        final_metric["precision"] = max(precision, final_metric["precision"])
        final_metric["recall"] = max(recall, final_metric["recall"])
        final_metric["f1"] = max(f1, final_metric["f1"])
    
    return final_metric['f1']

def get_rm_score(trajectory: str) -> float:
    try:
        resp = requests.post(
            "your url",
            json={"trajectory": trajectory},
            timeout=10
        )
        print("RM raw response:", repr(resp.text)) 
        resp.raise_for_status()
        data = resp.json()
        return float(data["score"])
    except Exception as e:
        print("RM error:", e)
        print("RM raw response:", getattr(resp, "text", None))
        return 0.0   # 或一个惩罚值

def compute_score(tokenizer, solution_str, ground_truth) -> float:
    # handling both the base model and the instruction-tuned model
    if "<|im_start|>assistant\n" in solution_str:
        solution_str_split = solution_str.split("<|im_start|>assistant\n")
    else:
        solution_str_split = solution_str.split("Assistant:")
    
    response = solution_str_split[1]
    valid_template, reason = validate_format(response)
    if not valid_template:
        print(f"--------------------------------search bad format--------------------------------")
        return -1, f'search bad format: {reason}'

    if response.endswith(tokenizer.eos_token):
        response = response[:-len(tokenizer.eos_token)]
    


    answer_part = extract_answer(response)
    if answer_part is not None:
        try:
            answer = remove_boxed(last_boxed_only_string(answer_part))
        except Exception as e:
            print(f"--------------------------------find box error--------------------------------")
            return -1, f'find box error: {e}'
    else:
        print(f"--------------------------------cannot extract answer--------------------------------")
        return -1, f'cannot extract answer'

    f1_score = get_f1_score(answer, ground_truth)
    print(f"f1_score: {f1_score},answer: {answer},ground_truth: {ground_truth}")
    
    # 3. 轨迹奖励（RM）
    # 在调用 RM 之前
    trajectory_for_rm = response
    # 去掉 eos / chat token
    trajectory_for_rm = trajectory_for_rm.replace(tokenizer.eos_token, "")
    trajectory_for_rm = trajectory_for_rm.replace("<|im_end|>", "").strip()

    rm_score = get_rm_score(trajectory_for_rm)
    print(f"rm_score: {rm_score}")

    final_reward = f1_score
    if f1_score > 0:
        final_reward += 0.2 * rm_score

    final_reward = max(min(final_reward, 3.0), -3.0)
    final_reward = max(f1_score, final_reward)

    return final_reward, f'f1 score: {f1_score}, rm score: {rm_score}, final reward: {final_reward}'
    return final_reward, f'f1 score: {f1_score}, final reward: {final_reward}'

    #if f1_score > 0:
    #elif f1_score > 0:
        #print(f"--------------------------------correct answer--------------------------------")
        #return f1_score, f'correct answer, get f1 score: {f1_score}'
    #else:
        #print(f"--------------------------------wrong answer--------------------------------")
        #return 0, f'wrong answer but good format: {answer}'


if __name__ == "__main__":
    tokenizer=AutoTokenizer.from_pretrained("/to/your/path/model")
    response="<|im_start|>system\nYou are a helpful assistant that can solve the given question step by step with the help of tools like Wikipedia search and Python code execution. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, You may invoke the Wikipedia search tool for factual information or use Python code execution for calculation when needed. The reasoning process is enclosed within <think> </think>, and the answer is enclosed within <answer> </answer> tags. If Wikipedia search is used, the search query and result are enclosed in <search> </search> and <result> </result> tags respectively. If Python code execution is needed, the code and results are enclosed within <code> </code> and <result> </result> tags respectively. Example: <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process based on search result. </think> <answer> The final answer is \\boxed{answer here} </answer>. Or with Python code execution: <think> This is the reasoning process. </think> <code> python code here </code> <result> code result here </result> <think> This is the reasoning process based on code result. </think> <answer> The final answer is \\boxed{answer here} </answer>. If no tools are needed: <think> This is the reasoning process. </think> <answer> The final answer is \\boxed{answer here} </answer>. In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.<|im_end|>\n<|im_start|>user\nIn each cell, a strip of length $100$ is worth a chip. You can change any $2$ neighboring chips and pay $1$ rouble, and you can also swap any $2$ chips for free, between which there are exactly $4$ chips. What is the smallest amount of rubles you can spend to rearrange the chips in reverse order?<|im_end|>\n<|im_start|>assistant\n<think> To solve the given problem, we will use both natural language reasoning and Python code. Let's break down the problem step-by-step.\n\n1. **Given Equations:**\n   \\[\n   \\log_{10}(\\sin x) + \\log_{10}(\\cos x) = -1\n   \\]\n   Using the properties of logarithms, we can combine the logs:\n   \\[\n   \\log_{10}(\\sin x \\cos x) = -1 \\implies \\sin x \\cos x = 10^{-1} = \\frac{1}{10}\n   \\]\n   Recall the double-angle identity for sine: \\(\\sin(2x) = 2 \\sin x \\cos x\\). So,\n   \\[\n   \\sin(2x) = 2 \\left(\\frac{1}{10}\\right) = \\frac{1}{5}\n   \\]\n\n2. **Second Given Equation:**\n   \\[\n   \\log_{10}(\\sin x + \\cos x) = \\frac{1}{2}(\\log_{10}(n) - 1)\n   \\]\n   We can rewrite the right-hand side using properties of logarithms:\n   \\[\n   \\log_{10}(\\sin x + \\cos x) = \\frac{1}{2}\\log_{10}(n) - \\frac{1}{2} = \\log_{10}(n^{1\/2}) - \\log_{10}(10^{1\/2}) = \\log_{10}\\left(\\frac{\\sqrt{n}}{\\sqrt{10}}\\right)\n   \\]\n   This implies:\n   \\[\n   \\sin x + \\cos x = \\frac{\\sqrt{n}}{\\sqrt{10}}\n   \\]\n   Squaring both sides, we get:\n   \\[\n   (\\sin x + \\cos x)^2 = \\frac{n}{10}\n   \\]\n   Expanding the left side:\n   \\[\n   \\sin^2 x + \\cos^2 x + 2 \\sin x \\cos x = \\frac{n}{10}\n   \\]\n   Using the Pythagorean identity \\(\\sin^2 x + \\cos^2 x = 1\\) and the earlier result \\(\\sin x \\cos x = \\frac{1}{10}\\):\n   \\[\n   1 + 2 \\left(\\frac{1}{10}\\right) = \\frac{n}{10} \\implies 1 + \\frac{1}{5} = \\frac{n}{10} \\implies \\frac{6}{5} = \\frac{n}{10} \\implies n = 12\n   \\]\n\nNow, let's verify this with Python code to ensure the result is accurate.\n </think><code>\nimport math\r\nfrom sympy import symbols, Eq, solve, log, sin, cos\r\n\r\n# Define the variable\r\nx = symbols('x')\r\n\r\n# First equation: log10(sin(x)) + log10(cos(x)) = -1\r\neq1 = Eq(log(sin(x), 10) + log(cos(x), 10), -1)\r\n\r\n# Solve for sin(x) * cos(x)\r\nsin_x_cos_x = solve(eq1, sin(x) * cos(x))[0]\r\nsin_x_cos_x = sin_x_cos_x.simplify()\r\n\r\n# We know sin(x) * cos(x) = 1\/10\r\nsin_x_cos_x_value = 1 \/ 10\r\n\r\n# Second equation: log10(sin(x) + cos(x)) = 1\/2 * (log10(n) - 1)\r\nn = symbols('n')\r\neq2 = Eq(log(sin(x) + cos(x), 10), (1\/2) * (log(n, 10) - 1))\r\n\r\n# We know (sin(x) + cos(x))^2 = 1 + 2 * sin(x) * cos(x) = 1 + 2 * (1\/10) = 6\/5\r\nsin_x_plus_cos_x_squared = 6 \/ 5\r\nsin_x_plus_cos_x = math.sqrt(sin_x_plus_cos_x_squared)\r\n\r\n# Substitute sin(x) + cos(x) into the second equation\r\nlhs = log(sin_x_plus_cos_x, 10)\r\nrhs = (1\/2) * (log(n, 10) - 1)\r\n\r\n# Solve for n\r\nn_value = solve(Eq(lhs, rhs), n)[0]\r\nprint(n_value)\n</code><result>\n\n12.0000000000000\n</result><search> query </search> query <result> result </result> <answer>The value of \\( n \\) is \\(\\boxed{12}\\). The calculations and the verification using Python both confirm that the final answer is indeed \\( n = 12 \\).</answer><|im_end|>"
    #response="<|im_start|>system\nYou are a helpful assistant that can solve the given question step by step with the help of tools like Wikipedia search and Python code execution. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, You may invoke the Wikipedia search tool for factual information or use Python code execution for calculation when needed. The reasoning process is enclosed within <think> </think>, and the answer is enclosed within <answer> </answer> tags. If Wikipedia search is used, the search query and result are enclosed in <search> </search> and <result> </result> tags respectively. If Python code execution is needed, the code and results are enclosed within <code> </code> and <result> </result> tags respectively. Example: <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process based on search result. </think> <answer> The final answer is \\boxed{answer here} </answer>. Or with Python code execution: <think> This is the reasoning process. </think> <code> python code here </code> <result> code result here </result> <think> This is the reasoning process based on code result. </think> <answer> The final answer is \\boxed{answer here} </answer>. If no tools are needed: <think> This is the reasoning process. </think> <answer> The final answer is \\boxed{answer here} </answer>. In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.<|im_end|>\n<|im_start|>user\nDerek was 6 years old when he had three times as many dogs as cars. Ten years later, after selling some of his dogs and buying 210 more cars, the number of cars became twice the number of dogs. How many dogs does Derek have now if he had 90 dogs when he was six years old?<|im_end|>\n<|im_start|>assistant\n<think>\nTo solve this problem, we need to determine the number of dogs Derek has now. We know that when Derek was 6 years old, he had three times as many dogs as cars. We also know that 10 years later, the number of cars became twice the number of dogs after he sold some of his dogs and bought 210 more cars. We are given that Derek had 90 dogs when he was 6 years old.\n\nFirst, let's establish the initial number of cars and dogs when Derek was 6 years old. If Derek had 90 dogs, and he had three times as many dogs as cars, then the number of cars is:\n\\[ \\text{Number of cars} = \\frac{90}{3} = 30 \\]\n\nNext, we need to determine the number of dogs and cars 10 years later. Let \\( x \\) be the number of dogs Derek sold. The number of dogs after selling \\( x \\) dogs and buying 210 more cars is:\n\\[ 90 - x + 210 \\]\n\nThe number of cars after buying 210 more cars is:\n\\[ 30 + 210 = 240 \\]\n\nAccording to the problem, the number of cars is twice the number of dogs 10 years later:\n\\[ 240 = 2(90 - x + 210) \\]\n\nSolving for \\( x \\):\n\\[ 240 = 2(300 - x) \\]\n\\[ 240 = 600 - 2x \\]\n\\[ 2x = 600 - 240 \\]\n\\[ 2x = 360 \\]\n\\[ x = 180 \\]\n\nSo, Derek sold 180 dogs and bought 210 more cars. The number of dogs now is:\n\\[ 90 - 180 + 210 = 120 \\]\n\nTherefore, the number of dogs Derek has now is 120.</think>\n<answer>\\boxed{120}</answer><|im_end|>"
    #response="<|im_start|>system\nYou are a helpful assistant that can solve the given question step by step with the help of tools like Wikipedia search and Python code execution. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, You may invoke the Wikipedia search tool for factual information or use Python code execution for calculation when needed. The reasoning process is enclosed within <think> </think>, and the answer is enclosed within <answer> </answer> tags. If Wikipedia search is used, the search query and result are enclosed in <search> </search> and <result> </result> tags respectively. If Python code execution is needed, the code and results are enclosed within <code> </code> and <result> </result> tags respectively. Example: <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> <think> This is the reasoning process based on search result. </think> <answer> The final answer is \\boxed{answer here} </answer>. Or with Python code execution: <think> This is the reasoning process. </think> <code> python code here </code> <result> code result here </result> <think> This is the reasoning process based on code result. </think> <answer> The final answer is \\boxed{answer here} </answer>. If no tools are needed: <think> This is the reasoning process. </think> <answer> The final answer is \\boxed{answer here} </answer>. In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format.<|im_end|>\n<|im_start|>user\nBella is making bracelets for her 6 friends. She needs 8 beads per bracelet. She has 36 beads. How many more beads does she need to make all six bracelets?<|im_end|>\n<|im_start|>assistant\n<think>\nTo solve this problem, we need to calculate the total number of beads required for 6 bracelets and then subtract the number of beads Bella already has. \n\nFirst, we calculate the total number of beads needed for 6 bracelets:\n- Each bracelet requires 8 beads.\n- Therefore, the total number of beads needed is \\( 6 \\times 8 = 48 \\).\n\nNext, we subtract the number of beads Bella already has from the total number of beads needed:\n- Bella has 36 beads.\n- The number of additional beads needed is \\( 48 - 36 = 12 \\).\n\nThus, Bella needs 12 more beads to make all six bracelets.\n</think>\n<answer>\\boxed{12}</answer><|im_end|>"
    
    answer="12"
    #answer="120"
    #answer="12"
    res=compute_score(tokenizer,response, answer)
    print(res)
