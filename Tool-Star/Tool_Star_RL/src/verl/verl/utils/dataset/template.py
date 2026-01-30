re_search_template = """A conversation between User and Assistant. \
The user asks a question, and the assistant solves it. \
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. \
During thinking, the assistant can invoke the wikipedia search tool to search for fact information about specific topics if needed. \
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{{answer here}} \\] </answer>. \
In the last part of the answer, the final exact answer is enclosed within \\boxed{{}} with latex format. \
User: {prompt}. Assistant:"""

# re_search_template_sys = """You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool. \
# Given a question, you need to first think about the reasoning process in the mind and then provide the answer. \
# During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed. \
# The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, \
# and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. \
# For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> \
# <think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. \
# In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."""

re_search_template_sys = """You are a helpful assistant that can solve the given question step by step with the help of tools
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
</think> <answer> The final answer is \\boxed{answer here} </answer>. Or with Python code
execution: <think> This is the reasoning process. </think> <code> python code here </code>
<result> code result here </result> <think> This is the reasoning process based on code result.
</think> <answer> The final answer is \\boxed{answer here} </answer>. If no tools are needed:
<think> This is the reasoning process. </think> <answer> The final answer is \\boxed{answer
here} </answer>. In the last part of the answer, the final exact answer is enclosed within \\boxed{}
with latex format."""

# re_search_template_sys ="""You are a helpful assistant.
# Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
# The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".

# During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<search> search query (only keywords) here </search>".
# The proposed query must search for a straightforward sub-question. Furthermore, **the query must involve ONLY a single triple**.
# Then, the search system will provide you with the retrieval information with the format of "<result> ...search results... </result>".

# During the thinking process, **you can generate python code as tool to calculate the math problem** if necessary with the format of "<python> python code here </python>".
# Then, the python code will be executed. Please note that **only variables in print() statements will have results** returned by the compiler. The result will be provided to you with the format of "<result> ...python code result... </result>".

# Here is a output example:

# <think>
# Let's break down the problem:

# Betty needs $100 for a new wallet.
# She currently has half of the money she needs, which is $100/2 = $50.
# Her parents give her $15.
# Her grandparents give her twice as much as her parents, which is $15 × 2 = $30.

# Now I need to calculate how much more money Betty needs to buy the wallet.
# Money Betty has:

# Initial amount: $50
# From parents: $15
# From grandparents: $30
# ›
# Total money Betty has = $50 + $15 + $30 = $95
# Money Betty needs = $100
# How much more money Betty needs = $100 - $95 = $5
# Let me check this calculation using Python:
# <python>
# wallet_cost = 100
# initial_amount = wallet_cost / 2  # Half of what she needs
# parents_contribution = 15
# grandparents_contribution = parents_contribution * 2  # Twice as much as parents
# total_money = initial_amount + parents_contribution + grandparents_contribution
# more_money_needed = wallet_cost - total_money
# print(more_money_needed)
# </python>
# <result>5.0</result>
# The Python calculation confirms my manual calculation. Betty needs $5 more to buy the wallet.
# </think>
# <answer>
# The final answer is \[ \\boxed{{5}} \]
# </answer>


# If your knowledge is sufficient for you to reason, please generate the final answer with <answer> The final answer is \\[ \\boxed{{answer here}} \\] </answer>. the final exact answer is enclosed within \\boxed{{}} with latex format.
# """

prompt_template_dict = {}
prompt_template_dict['re_search_template'] = re_search_template
prompt_template_dict['re_search_template_sys'] = re_search_template_sys
