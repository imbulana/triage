def evaluate_item(i, query, answer, benchmark, tokenizer):
    DEEPSEEK_MODEL = "deepseek-v3-250324"
    DEEPSEEK_URL = "xxxx"
    client = OpenAI(api_key='xxxx',base_url=DEEPSEEK_URL)
    sys_prompt="""
        ---Role---
        You are an expert tasked with evaluating answer to the question based on four criteria: **Comprehensiveness**, **Diversity**, **Directness** and **Empowerment**.
        """

    prompt = f"""
        Your task is to evaluate the following answer based on four criteria. For each criterion, assign a score from 1 to 10 , following the detailed scoring rubric.

        When explaining your score, you must refer directly to specific parts of the answer to justify your reasoning. Avoid general statements — your explanation must be grounded in the content provided.

        - **Comprehensiveness**: 
        How much detail does the answer provide to cover all aspects and details of the question?
        
        - **Diversity**:
        How varied and rich is the answer in providing different perspectives and insights on the question?
      
        - **Empowerment**: 
        How well does the answer help the reader understand and make informed judgments about the topic?

        - **Overall Quality**:
        Provide an overall evaluation based on the combined performance across all four dimensions. Consider both content quality and answer usefulness to the question.
        
        Scoring Guidelines:

        "1-2": "Low score description: Clearly deficient in this aspect, with significant issues.", 
        "3-4": "Below average score description: Lacking in several important areas, with noticeable problems.", 
        "5-6": "Average score description: Adequate but not exemplary, meets basic expectations with some minor issues.",
        "7-8": "Above average score description: Generally strong but with minor shortcomings.", 
        "9-10": "High score description: Outstanding in this aspect, with no noticeable issues."

        Here is the question:
        {query}

        Here are the  answer:

        {answer}

        Evaluate the answer using the  criteria listed above and provide detailed explanations for each criterion with reference to the text. 
        Output your evaluation in the following JSON format:

        {{
            "Comprehensiveness": {{
                "score": "[1-10]",
                "Explanation": "[Provide explanation here]"
            }},
            "Empowerment": {{
                "score": "[1-10]",
                "Explanation": "[Provide explanation here]"
            }},
            "Diversity": {{
                "score": "[1-10]",
                "Explanation": "[Provide explanation here]"
            }},
            "Overall Quality": {{
                "score": "[1-10]",
                "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
            }}
        }}
        """
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    cur_token_cost = len(tokenizer.encode(messages[0]['content'] + messages[1]['content']))

    max_retries = 5
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=messages,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}}
            )
            content = response.choices[0].message.content
            json_text = '\n'.join(content.strip().split('\n')[1:-1])
            evaluation = json.loads(json_text)
            return i, evaluation, cur_token_cost
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print(f"[{i}] Failed after retries: {e}")
                return i, None, cur_token_cost
def evaluate_example(query_file,result_file,output_file,dataset):
    total_token_cost = 0
    DEEPSEEK_MODEL = "xxxx"
    DEEPSEEK_URL = "xxxx"
    client = OpenAI(api_key='xxx',base_url=DEEPSEEK_URL)
    
    queries = []
    with open(query_file, "r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                if "mix" in query_file:
                    query = json_obj.get("input")
                    MAX_QUERIES=130
                else:
                    query = json_obj.get("query")
                    MAX_QUERIES=100
                queries.append(query)
            except json.JSONDecodeError as e:
                print(
                f"JSON decoding error in file {query_file} at line {line_number}: {e}"
                )
    queries = queries[:MAX_QUERIES]
    with open(result_file, "r") as f:
        answers = f.readlines()
    answers = [json.loads(i)["answer"] for i in answers][:MAX_QUERIES]
    with open(f"datasets/{dataset}/{dataset}.jsonl", "r") as f:
        benchmarks=f.readlines()
    benchmarks=[json.loads(i)["answers"] for i in benchmarks][:MAX_QUERIES]
    
    total_token_cost = 0
    results = [None] * len(queries)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(evaluate_item, i, q, a, b, tokenizer, ): i
            for i, (q, a, b) in enumerate(zip(queries, answers, benchmarks))
        }

        for future in as_completed(futures):
            i, evaluation, token_cost = future.result()
            total_token_cost += token_cost
            results[i] = evaluation
            if evaluation is not None:
                print(f"Successfully evaluated {i+1}/{len(queries)}")

    # 只写入成功的结果
    with jsonlines.open(output_file, mode="w") as writer:
        for eval_item in results:
            if eval_item is not None:
                writer.write(eval_item)

    return total_token_cost