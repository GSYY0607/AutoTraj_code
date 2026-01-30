import json
import re

# --------- 答案判定函数 ---------
def is_answer_correct(gold_answer, traj_answer):
    """
    判断轨迹答案是否正确：
    1. 从标准答案和轨迹答案中提取 {} 内内容
    2. 纯数字答案进行浮点数比较
    3. 非数字答案做字符串精确匹配或子串匹配
    """
    def extract_brace_content(text):
        matches = re.findall(r'\{(.*?)\}', text)
        if matches:
            # 取最后一个 {} 内的内容作为答案
            return matches[-1].strip()
        return text.strip()

    gold_ans = extract_brace_content(gold_answer)
    traj_ans = extract_brace_content(traj_answer)

    # 尝试数字比较
    try:
        gold_num = float(re.sub(r'[^\d.-]', '', gold_ans))
        traj_num = float(re.sub(r'[^\d.-]', '', traj_ans))
        return abs(gold_num - traj_num) < 1e-6
    except:
        # 非数字，先尝试精确匹配
        if gold_ans == traj_ans:
            return True
        # 再尝试子串匹配
        if traj_ans in gold_ans:
            return True
    return False

# --------- 主处理函数 ---------
def classify_trajectories(input_file, correct_file, incorrect_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    #data = data[:10]   # ✅ 只测试前 10 条问答对
    
    correct_data = []
    incorrect_data = []

    correct_traj_count = 0
    incorrect_traj_count = 0

    for item in data:
        question = item['question']
        gold_answer = item['gold_answer']
        trajectories = item.get('trajectories', [])

        # 对每条轨迹判断答案是否正确
        traj_classes = []
        for traj in trajectories:
            traj_ans_match = re.search(r'<answer>(.*?)</answer>', traj['react_trajectory'], re.DOTALL)
            traj_ans_text = traj_ans_match.group(1) if traj_ans_match else ''
            correct = is_answer_correct(gold_answer, traj_ans_text)
            traj_classes.append(correct)

        # 统计轨迹数量
        correct_traj_count += sum(traj_classes)
        incorrect_traj_count += len(traj_classes) - sum(traj_classes)

        # 判断同一问题下轨迹分类情况
        if all(traj_classes):
            # 都正确
            correct_data.append({
                'question': question,
                'gold_answer': gold_answer,
                'trajectories': trajectories
            })
        elif not any(traj_classes):
            # 都错误
            incorrect_data.append({
                'question': question,
                'gold_answer': gold_answer,
                'trajectories': trajectories
            })
        else:
            # 一正一负
            for idx, correct in enumerate(traj_classes):
                single_traj = {
                    'question': question,
                    'gold_answer': gold_answer,
                    'trajectories': [trajectories[idx]]
                }
                if correct:
                    correct_data.append(single_traj)
                else:
                    incorrect_data.append(single_traj)

    # 保存结果
    with open(correct_file, 'w', encoding='utf-8') as f:
        json.dump(correct_data, f, ensure_ascii=False, indent=2)

    with open(incorrect_file, 'w', encoding='utf-8') as f:
        json.dump(incorrect_data, f, ensure_ascii=False, indent=2)

    print(f"分类完成，问答对数量：答案正确 {len(correct_data)}，答案错误 {len(incorrect_data)}")
    print(f"轨迹数量：答案正确 {correct_traj_count}，答案错误 {incorrect_traj_count}")

# --------- 使用示例 ---------
if __name__ == "__main__":
    input_file = '/to/your/path/generated_trajectory.json'
    correct_file = '/to/your/path/answer_correct_trajectory.json'
    incorrect_file = '/to/your/path/answer_wrong_low_quality_trajectory.json'

    classify_trajectories(input_file, correct_file, incorrect_file)
