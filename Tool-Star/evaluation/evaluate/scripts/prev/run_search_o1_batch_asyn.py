# run_search_o1.py
import os
import json
import time
import re
from tqdm import tqdm
import numpy as np
import torch
import string
from typing import Optional, Tuple, List, Dict
import argparse
import random
import asyncio

from openai import AsyncOpenAI

from bing_search import (
    bing_web_search, 
    extract_relevant_info, 
    fetch_page_content, 
    extract_snippet_with_context
)
from evaluate import (
    run_evaluation, 
    extract_answer_fn
)
from prompts import (
    get_gpqa_search_o1_instruction, 
    get_math_search_o1_instruction, 
    get_code_search_o1_instruction, 
    get_singleqa_search_o1_instruction, 
    get_multiqa_search_o1_instruction, 
    get_webpage_to_reasonchain_instruction,
    get_task_instruction_openqa, 
    get_task_instruction_math, 
    get_task_instruction_multi_choice, 
    get_task_instruction_code, 
    gpqa_search_o1_examples_dpsk
)

# Define special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

def parse_args():
    parser = argparse.ArgumentParser(description="Run Search-o1 for various datasets and models.")

    # Dataset and split configuration
    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True,
        help="Name of the dataset to use."
    )

    parser.add_argument(
        '--split',
        type=str,
        required=True,
        help="Dataset split to use."
    )

    parser.add_argument(
        '--subset_num',
        type=int,
        default=-1,
        help="Number of examples to process. Defaults to all if not specified."
    )

    # Search and document retrieval configuration
    parser.add_argument(
        '--max_search_limit',
        type=int,
        default=10,
        help="Maximum number of searches per question."
    )

    parser.add_argument(
        '--max_turn',
        type=int,
        default=15,
        help="Maximum number of turns."
    )

    parser.add_argument(
        '--top_k',
        type=int,
        default=10,
        help="Maximum number of search documents to return."
    )

    parser.add_argument(
        '--max_doc_len',
        type=int,
        default=3000,
        help="Maximum length of each searched document."
    )

    parser.add_argument(
        '--use_jina',
        type=bool,
        default=False,
        help="Whether to use Jina API for document fetching."
    )

    parser.add_argument(
        '--jina_api_key',
        type=str,
        default='None',
        help="Your Jina API Key to Fetch URL Content."
    )

    # Sampling parameters
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help="Sampling temperature."
    )

    parser.add_argument(
        '--top_p',
        type=float,
        default=0.8,
        help="Top-p sampling parameter."
    )

    parser.add_argument(
        '--top_k_sampling',
        type=int,
        default=20,
        help="Top-k sampling parameter."
    )

    parser.add_argument(
        '--repetition_penalty',
        type=float,
        default=None,
        help="Repetition penalty. If not set, defaults based on the model."
    )

    parser.add_argument(
        '--max_tokens',
        type=int,
        default=32768,
        help="Maximum number of tokens to generate. If not set, defaults based on the model and dataset."
    )

    # Bing API Configuration
    parser.add_argument(
        '--bing_subscription_key',
        type=str,
        required=True,
        help="Bing Search API subscription key."
    )

    parser.add_argument(
        '--bing_endpoint',
        type=str,
        default="https://api.bing.microsoft.com/v7.0/search",
        help="Bing Search API endpoint."
    )

    # Add new eval and seed arguments
    parser.add_argument(
        '--eval',
        action='store_true',
        help="Whether to run evaluation after generation."
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help="Random seed for generation. If not set, will use current timestamp as seed."
    )

    # Add new arguments to parser
    parser.add_argument(
        '--api_base_url',
        type=str,
        default="http://39.101.64.147:28706/qwq/v1",
        help="Base URL for the API endpoint"
    )

    parser.add_argument(
        '--model_name',
        type=str,
        default="QwQ-32B",
        help="Name of the model to use"
    )
    
    parser.add_argument(
        '--concurrent_limit',
        type=int,
        default=200,
        help="Maximum number of concurrent API calls"
    )

    return parser.parse_args()

async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    semaphore: asyncio.Semaphore,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    top_k: int,
    model_name: str,
    retry_limit: int = 3,
) -> str:
    """Generate a single response with retry logic"""
    for attempt in range(retry_limit):
        try:
            async with semaphore:
                messages = [{"role": "user", "content": prompt}]
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=min(max_tokens, 32768),  # Reserve 1000 tokens for prompt
                    stop=[END_SEARCH_QUERY],
                    extra_body={
                        'top_k': top_k,
                        'include_stop_str_in_output': True,
                        'repetition_penalty': repetition_penalty
                    },
                    # timeout=900,
                )
                print('---\n', response.choices[0].message.content)
                return response.choices[0].message.content
        except Exception as e:
            if attempt == retry_limit - 1:
                print(f"Failed after {retry_limit} attempts: {e}")
                return ""
            await asyncio.sleep(1 * (attempt + 1))
    return ""

async def generate_all_responses(
    client: AsyncOpenAI,
    prompts: List[str],
    concurrent_limit: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
    top_k: int,
    model_name: str,
) -> List[str]:
    """Generate all responses concurrently with a limit"""
    semaphore = asyncio.Semaphore(concurrent_limit)
    
    tasks = [
        generate_response(
            client, prompt, semaphore, temperature, top_p, max_tokens, repetition_penalty, top_k, model_name
        )
        for prompt in prompts
    ]
    
    with tqdm(total=len(tasks)) as pbar:
        # Create a progress tracking callback
        async def track_progress(task):
            result = await task
            pbar.update(1)
            return result
            
        # Wrap each task with the progress tracker
        tracked_tasks = [track_progress(task) for task in tasks]
        
        # Gather all results in order
        responses = await asyncio.gather(*tracked_tasks)
    
    return responses

async def generate_webpage_to_reasonchain_batch(
    client: AsyncOpenAI,
    original_questions: List[str],
    prev_reasonings: List[str],
    search_queries: List[str],
    documents: List[str],
    dataset_name: str,
    batch_output_records: List[Dict],
    max_tokens: int = 32768,
    temperature: float = 0.7,
    top_p: float = 0.8,
    repetition_penalty: float = 1.05,
    top_k: int = 20,
    model_name: str = "QwQ-32B",
    concurrent_limit: int = 200,
) -> List[str]:
    user_prompts = [
        get_webpage_to_reasonchain_instruction(r, sq, doc)
        for r, sq, doc in zip(prev_reasonings, search_queries, documents)
    ]

    raw_outputs = await generate_all_responses(
        client=client,
        prompts=user_prompts,
        concurrent_limit=concurrent_limit,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        model_name=model_name,
    )
    
    extracted_infos = [extract_answer_fn(raw, mode='infogen') for raw in raw_outputs]

    for p, r, e in zip(user_prompts, raw_outputs, extracted_infos):
        batch_output_records.append({
            'prompt': p,
            'raw_output': r,
            'extracted_info': e
        })

    return extracted_infos

async def main_async():
    args = parse_args()

    # Set random seed
    if args.seed is None:
        args.seed = int(time.time())
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.jina_api_key == 'None':
        jina_api_key = None

    # Set default repetition_penalty if not provided
    if args.repetition_penalty is None:
        args.repetition_penalty = 1.05 if 'qwq' in args.model_name.lower() or 'deepseek' in args.model_name.lower() or 'sky-t1' in args.model_name.lower() else 1.0

    # Data paths based on dataset
    if args.dataset_name == 'livecode':
        data_path = f'./data/LiveCodeBench/{args.split}.json'
    elif args.dataset_name == 'webwalker':
        data_path = f'./data/WebWalkerQA/{args.split}.json'
    elif args.dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'gaia', 'hle']:
        data_path = f'./data/{args.dataset_name.upper()}/{args.split}.json'
    else:
        data_path = f'./data/QA_Datasets/{args.dataset_name}.json'

    print('-----------------------')
    print(f'Using {args.dataset_name} {args.split} set.')
    print('-----------------------')

    # ---------------------- Caching Mechanism ----------------------
    # Define cache directories and file paths
    cache_dir = './cache'
    search_cache_path = os.path.join(cache_dir, 'search_cache.json')
    url_cache_path = os.path.join(cache_dir, 'url_cache.json')

    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache = json.load(f)
    else:
        search_cache = {}

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache = json.load(f)
    else:
        url_cache = {}

    # Function to save caches
    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)
        with open(url_cache_path, 'w', encoding='utf-8') as f:
            json.dump(url_cache, f, ensure_ascii=False, indent=2)

    # Define output directory based on model and dataset
    if 'qwq' in args.model_name.lower():
        model_short_name = 'qwq'
    elif 'deepseek' in args.model_name.lower():
        if 'llama-8b' in args.model_name.lower():
            model_short_name = 'dpsk-llama-8b'
        elif 'llama-70b' in args.model_name.lower():
            model_short_name = 'dpsk-llama-70b'
        elif 'qwen-1.5b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-1.5b'
        elif 'qwen-7b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-7b'
        elif 'qwen-32b' in args.model_name.lower():
            model_short_name = 'dpsk-qwen-32b'
    elif 'sky-t1' in args.model_name.lower():
        model_short_name = 'sky-t1'
    else:
        model_short_name = args.model_name.split('/')[-1].lower().replace('-instruct', '')

    if model_short_name in ['qwq', 'dpsk-llama-8b', 'dpsk-llama-70b', 'dpsk-qwen-1.5b', 'dpsk-qwen-7b', 'dpsk-qwen-32b', 'sky-t1']:
        if args.dataset_name in ['math500', 'gpqa', 'aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{args.dataset_name}.{model_short_name}.search_o1'
            if args.dataset_name == 'gpqa' and (args.max_search_limit != 5 or args.top_k != 10):
                output_dir = f'./outputs/runs.analysis/{args.dataset_name}.{model_short_name}.search_o1.{args.max_search_limit}.{args.top_k}'
        else:
            output_dir = f'./outputs/runs.qa/{args.dataset_name}.{model_short_name}.search_o1'
    else:
        output_dir = f'./outputs/runs.baselines/{args.dataset_name}.{model_short_name}.search_o1'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the OpenAI client instead of vLLM
    client = AsyncOpenAI(
        api_key="empty",  # or your actual API key if needed
        base_url=args.api_base_url,
    )

    # ---------------------- Data Loading ----------------------
    with open(data_path, 'r', encoding='utf-8') as json_file:
        filtered_data = json.load(json_file)

    # ---------------------- Preparation of Input Prompts ----------------------
    input_list = []
    active_sequences = []
    for item in filtered_data:
        question = item['Question']
        if args.dataset_name in ['nq', 'triviaqa', 'hotpotqa', 'musique', 'bamboogle', '2wiki', 'gaia', 'hle', 'webwalker']:
            if args.dataset_name in ['nq', 'triviaqa']:
                instruction = get_singleqa_search_o1_instruction(args.max_search_limit)
            elif args.dataset_name in ['hotpotqa', 'musique', 'bamboogle', '2wiki', 'gaia', 'hle', 'webwalker']:
                instruction = get_multiqa_search_o1_instruction(args.max_search_limit)
            if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='qwq')
            elif 'deepseek' in args.model_name.lower():
                user_prompt = get_task_instruction_openqa(question, model_name='dpsk')
            else:
                user_prompt = get_task_instruction_openqa(question)

        elif args.dataset_name in ['math500', 'aime', 'amc']:
            instruction = get_math_search_o1_instruction(args.max_search_limit)
            if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_math(question, model_name='qwq')
            elif 'deepseek' in args.model_name.lower():
                user_prompt = get_task_instruction_math(question, model_name='dpsk')
            else:
                user_prompt = get_task_instruction_math(question)

        elif args.dataset_name in ['gpqa']:
            instruction = get_gpqa_search_o1_instruction(args.max_search_limit)
            if 'qwq' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='qwq')
            elif 'deepseek' in args.model_name.lower():
                instruction += gpqa_search_o1_examples_dpsk
                user_prompt = get_task_instruction_multi_choice(question, model_name='dpsk')
            elif 'llama' in args.model_name.lower():
                user_prompt = get_task_instruction_multi_choice(question, model_name='llama')
            else:
                user_prompt = get_task_instruction_multi_choice(question)

        elif args.dataset_name == 'livecode':
            instruction = get_code_search_o1_instruction(args.max_search_limit)
            question_title = item.get('question_title', '')
            if 'qwq' in args.model_name.lower() or 'deepseek' in args.model_name.lower() or 'sky-t1' in args.model_name.lower():
                user_prompt = get_task_instruction_code(question, question_title=question_title, model_name='qwq')
            else:
                user_prompt = get_task_instruction_code(question)

        else:
            # Default to multi-hop question
            instruction = get_multiqa_search_o1_instruction(args.max_search_limit)
            user_prompt = get_task_instruction_openqa(question)

        prompt = instruction + user_prompt
        input_list.append(prompt)
        active_sequences.append({
            'item': item,
            'prompt': prompt,
            'output': '',
            'finished': False,
            'history': [],
            'search_count': 0,
            'executed_search_queries': set(),
        })

    # ---------------------- Batch Generation Function ----------------------
    async def run_generation(sequences: List[Dict], max_tokens: int) -> List:
        prompts = [s['prompt'] for s in sequences]
        outputs = await generate_all_responses(
            client=client,
            prompts=prompts,
            concurrent_limit=args.concurrent_limit,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=max_tokens,
            repetition_penalty=args.repetition_penalty,
            top_k=args.top_k_sampling,
            model_name=args.model_name,
        )
        return [{"outputs": [{"text": output}]} for output in outputs]

    # Function to extract text between two tags
    def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
        pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
        matches = re.findall(pattern, text, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
        return None

    def replace_recent_steps(origin_str, replace_str):
        """
        Replaces specific steps in the original reasoning steps with new steps.
        If a replacement step contains "DELETE THIS STEP", that step is removed.

        Parameters:
        - origin_str (str): The original reasoning steps.
        - replace_str (str): The steps to replace or delete.

        Returns:
        - str: The updated reasoning steps after applying replacements.
        """

        def parse_steps(text):
            """
            Parses the reasoning steps from a given text.

            Parameters:
            - text (str): The text containing reasoning steps.

            Returns:
            - dict: A dictionary mapping step numbers to their content.
            """
            step_pattern = re.compile(r"Step\s+(\d+):\s*")
            steps = {}
            current_step_num = None
            current_content = []

            for line in text.splitlines():
                step_match = step_pattern.match(line)
                if step_match:
                    # If there's an ongoing step, save its content
                    if current_step_num is not None:
                        steps[current_step_num] = "\n".join(current_content).strip()
                    current_step_num = int(step_match.group(1))
                    content = line[step_match.end():].strip()
                    current_content = [content] if content else []
                else:
                    if current_step_num is not None:
                        current_content.append(line)
            
            # Save the last step if any
            if current_step_num is not None:
                steps[current_step_num] = "\n".join(current_content).strip()
            
            return steps

        # Parse the original and replacement steps
        origin_steps = parse_steps(origin_str)
        replace_steps = parse_steps(replace_str)

        # Apply replacements
        for step_num, content in replace_steps.items():
            if "DELETE THIS STEP" in content:
                # Remove the step if it exists
                if step_num in origin_steps:
                    del origin_steps[step_num]
            else:
                # Replace or add the step
                origin_steps[step_num] = content

        # Sort the steps by step number
        sorted_steps = sorted(origin_steps.items())

        # Reconstruct the reasoning steps as a single string
        new_reasoning_steps = "\n\n".join([f"{content}" for num, content in sorted_steps])

        return new_reasoning_steps

    # ---------------------- Initialize Collection Structure ----------------------
    # Initialize a list to collect batch outputs
    batch_output_records = []

    start_time = time.time()
    turn = 0

    # Main loop until all sequences are finished or maximum turns reached
    while True:
        sequences_needing_generation = [seq for seq in active_sequences if not seq['finished']]

        if sequences_needing_generation:
            turn += 1
            print(f'\n-------------- Turn {turn} --------------')
            print(f"We have {len(sequences_needing_generation)} sequences needing generation...")
            outputs = await run_generation(sequences_needing_generation, args.max_tokens)
            print("Generation completed, processing outputs...")

            # Initialize batch variables
            batch_relevant_info = []
            batch_original_questions = []
            batch_prev_reasonings = []
            batch_search_queries = []
            batch_documents = []
            batch_sequences = []

            # Collect URLs to fetch across all sequences
            all_urls_to_fetch = set()
            url_snippets = {}
            url_sequence_map = {}  # Map URL to list of sequences needing it

            # Process each sequence and collect URLs
            for seq, out in zip(sequences_needing_generation, outputs):
                text = out.outputs[0].text
                seq['history'].append(text)
                # Append generated text to prompt and output
                seq['prompt'] += text
                seq['output'] += text

                # Extract search query
                search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

                # If a search query is present and needs to be executed
                if search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY):
                    if seq['search_count'] < args.max_search_limit and search_query not in seq['executed_search_queries']:
                        # Execute search, use cache if available
                        if search_query in search_cache:
                            results = search_cache[search_query]
                            print(f"Using cached search results for query: \"{search_query}\"")
                        else:
                            try:
                                results = bing_web_search(search_query, args.bing_subscription_key, args.bing_endpoint, market='en-US', language='en')
                                search_cache[search_query] = results
                                print(f"Executed and cached search for query: \"{search_query}\"")
                            except Exception as e:
                                print(f"Error during search query '{search_query}': {e}")
                                search_cache[search_query] = {}
                                results = {}

                        # Extract relevant information from Bing search results
                        relevant_info = extract_relevant_info(results)[:args.top_k]
                        seq['relevant_info'] = relevant_info

                        # Extract URLs and snippets
                        urls_to_fetch = [it['url'] for it in relevant_info]
                        snippets = {info['url']: info['snippet'] for info in relevant_info if 'snippet' in info}

                        # Filter URLs that are not cached
                        urls_to_fetch_filtered = [u for u in urls_to_fetch if u not in url_cache]
                        cached_urls = [u for u in urls_to_fetch if u in url_cache]

                        # Store info for all_urls_to_fetch and url_snippets
                        for url in urls_to_fetch_filtered:
                            all_urls_to_fetch.add(url)
                            url_snippets[url] = snippets.get(url, "")

                        all_reasoning_steps = seq['output']
                        all_reasoning_steps = all_reasoning_steps.replace('\n\n', '\n').split("\n")

                        truncated_prev_reasoning = ""
                        for i, step in enumerate(all_reasoning_steps):
                            truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"

                        prev_steps = truncated_prev_reasoning.split('\n\n')
                        if len(prev_steps) <= 5:
                            truncated_prev_reasoning = '\n\n'.join(prev_steps)
                        else:
                            truncated_prev_reasoning = ''
                            for i, step in enumerate(prev_steps):
                                if i == 0 or i >= len(prev_steps) - 4 or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step:
                                    truncated_prev_reasoning += step + '\n\n'
                                else:
                                    if truncated_prev_reasoning[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                                        truncated_prev_reasoning += '...\n\n'
                        truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')

                        # Collect parameters for batch processing
                        batch_relevant_info.append(relevant_info)
                        batch_original_questions.append(seq['item']['Question'])
                        batch_prev_reasonings.append(truncated_prev_reasoning)
                        batch_search_queries.append(search_query)
                        batch_sequences.append(seq)

                        # Update search count and executed queries
                        seq['search_count'] += 1
                        seq['executed_search_queries'].add(search_query)

                    elif seq['search_count'] >= args.max_search_limit:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. You are not allowed to search.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print(f"Search limit reached for query: \"{search_query}\"")

                    elif search_query in seq['executed_search_queries']:
                        limit_message = f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. Please refer to previous results.\n{END_SEARCH_RESULT}\n"
                        seq['prompt'] += limit_message
                        seq['output'] += limit_message
                        seq['history'].append(limit_message)
                        print(f"Repeated search for query: \"{search_query}\"")

                else:
                    # If no search query needs to be executed, mark the sequence as finished
                    seq['finished'] = True
                    print("Sequence marked as complete.")

            # Batch fetch all URLs at once to optimize speed
            if all_urls_to_fetch:
                print(f"Fetching {len(all_urls_to_fetch)} URLs...")
                try:
                    fetched_contents = fetch_page_content(
                        list(all_urls_to_fetch),
                        use_jina=args.use_jina,
                        jina_api_key=args.jina_api_key,
                        # snippets=url_snippets  # Do not pass snippets when updating url_cache directly
                    )
                    print(f"Fetched {len(fetched_contents)} URLs successfully.")
                except Exception as e:
                    print(f"Error during batch URL fetching: {e}")
                    fetched_contents = {url: f"Error fetching URL: {e}" for url in all_urls_to_fetch}
                # Update cache with fetched contents
                for url, content in fetched_contents.items():
                    url_cache[url] = content

            # After fetching, prepare formatted documents for batch processing
            for relevant_info in batch_relevant_info:
                formatted_documents = ""
                for i, doc_info in enumerate(relevant_info):
                    url = doc_info['url']
                    raw_context = url_cache.get(url, "")
                    doc_info['snippet'] = doc_info['snippet'].replace('<b>','').replace('</b>','')            
                    success, filtered_context = extract_snippet_with_context(raw_context, doc_info['snippet'], context_chars=args.max_doc_len)
                    if success:
                        context = filtered_context
                    else:
                        context = raw_context[:args.max_doc_len*2]

                    doc_info['context'] = context
                    formatted_documents += f"**Web Page {i + 1}:**\n"
                    formatted_documents += json.dumps(doc_info, ensure_ascii=False, indent=2) + "\n"
                    
                batch_documents.append(formatted_documents)

            # After fetching, prepare for batch processing if there are any
            if batch_sequences:
                print(f"Batch processing {len(batch_sequences)} sequences with generate_webpage_to_reasonchain_batch...")
                webpage_analyses = await generate_webpage_to_reasonchain_batch(
                    client=client,
                    original_questions=batch_original_questions,
                    prev_reasonings=batch_prev_reasonings,
                    search_queries=batch_search_queries,
                    documents=batch_documents,
                    dataset_name=args.dataset_name,
                    batch_output_records=batch_output_records,
                    max_tokens=args.max_tokens,
                    model_name=args.model_name,
                    concurrent_limit=args.concurrent_limit,
                )
                print("Batch generation completed, assigning outputs to sequences...")

                for seq, analysis in zip(batch_sequences, webpage_analyses):
                    if isinstance(analysis, str):
                        append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)
                    else:
                        append_text = replace_recent_steps(seq['output'], analysis)
                        seq['prompt'] += append_text
                        seq['output'] += append_text
                        seq['history'].append(append_text)

        # Check if all sequences are finished
        unfinished = [seq for seq in active_sequences if not seq['finished']]
        if not unfinished:
            break
        else:
            if turn >= args.max_turn:
                print(f"Maximum number of turns ({args.max_turn}) reached, stopping.")
                break

    total_time = time.time() - start_time

    # ---------------------- Save Batch Output Records to JSON File ----------------------
    # Define output JSON file path
    t = time.localtime()
    batch_output_file = os.path.join(output_dir, f'{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.info_extract.json')

    # Save batch_output_records to JSON file
    with open(batch_output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_output_records, f, ensure_ascii=False, indent=2)

    print(f"Batch outputs saved to {batch_output_file}")

    # Prepare output list for evaluation
    output_list = [seq['output'] for seq in active_sequences]

    # Modify the evaluation section
    if args.eval:
        # Run evaluation
        run_evaluation(filtered_data, input_list, output_list, args.dataset_name, output_dir, total_time, args.split)
    else:
        # Save results without evaluation
        t = time.localtime()
        result_json_name = f'{args.split}.{t.tm_mon}.{t.tm_mday},{t.tm_hour}:{t.tm_min}.json'
        
        # Prepare results
        for item, seq in zip(filtered_data, active_sequences):
            item['Output'] = seq['output']
            
        # Save prediction results
        with open(os.path.join(output_dir, result_json_name), mode='w', encoding='utf-8') as json_file:
            json.dump(filtered_data, json_file, indent=4, ensure_ascii=False)

    # ---------------------- Update Search and URL Cache ----------------------
    print('Updating Search and URL Cache...')
    # Load existing caches or initialize empty dictionaries
    if os.path.exists(search_cache_path):
        with open(search_cache_path, 'r', encoding='utf-8') as f:
            search_cache_new = json.load(f)
    else:
        search_cache_new = {}

    if os.path.exists(url_cache_path):
        with open(url_cache_path, 'r', encoding='utf-8') as f:
            url_cache_new = json.load(f)
    else:
        url_cache_new = {}

    search_cache.update(search_cache_new)
    url_cache.update(url_cache_new)

    save_caches()

    print("Process completed.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
