import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm
import re
from openai import OpenAI
import httpx


from tools.logger_factory import setup_logger
from tools import io_file 
import prompt_kg_judge

logger = setup_logger("kg_score")



class llm_client():
    def __init__(self, args):
        self.model = args["llm_model"]
        self.base_url = args["llm_url"]
        self.api_key = args["llm_api_key"]
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, http_client=httpx.Client(verify=False))
    def call(self, user_prompt: str, system_prompt: str = "") -> str:

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt if system_prompt else "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        self.response = completion.choices[0].message.content

class TripleScorer:
    def __init__(self, triple_path:str, triple_soure_path:str , output_path:str = None):
        """
        初始化打分类
        """
        self.triples = io_file.read(triple_path)
        self.triple_sources = io_file.read(triple_soure_path)
        self.triple_sources = {(item["page_idx"], item["paragraph_idx"]): item["text"] for item in self.triple_sources}

        logger.info(f"Loaded {len(self.triples)} triples from {triple_path}")

        self.prompt = prompt_kg_judge.score_triple_prompt
        
        llm_client_args = {"llm_model":"Qwen2.5-72B", "llm_url":"http://0.0.0.0:8001/v1", "llm_api_key":"EMPTY"}
        self.client = llm_client(llm_client_args)

        self.output_path = output_path if output_path else triple_path.replace(".jsonl", ".scores.jsonl")
    
    def parse_triple(self, triple_str:str) -> dict[dict]:
        """
        解析三元组字符串为结构化字典
        :param triple_str: "Head|relation|Tail" 格式的字符串
        :return: {"head":..., "relation":..., "tail":...}
        """

        parts = triple_str.split("\t")
        return {
            "head": parts[0].strip().strip("<").strip(">"),
            "relation": parts[1].strip().strip("<").strip(">"),
            "tail": parts[2].strip().strip("<").strip(">")
        }

    def score_triple(self, triple_data:dict) -> dict:
        """
        :param jsonl_line: JSONL格式的输入数据
        :return: 包含所有打分结果的字典
        """
        # 解析输入数据
        triple_str = triple_data["triple"]
        source_text = triple_data["source_text"]
        triple = self.parse_triple(triple_str)

        prompt = self.prompt.format(
            source_text=source_text,
            head_entity=triple["head"],
            relation=triple["relation"],
            tail_entity=triple["tail"]
        )
        
        response = self._call_llm(prompt)
        parsed_score = self.parse_result(response)

        return parsed_score

    def parse_result(self, response: str) -> dict:
        try:
            # 新增正则表达式提取核心字段
            score_match = re.search(r'"score"\s*:\s*(\d*\.?\d+)', response, re.DOTALL)
            rationale_match = re.search(r'"rationale"\s*:\s*"((?:\\"|[^"])*)"', response, re.DOTALL)
            
            if score_match and rationale_match:
                # 提取并验证分数范围
                score = float(score_match.group(1))
                if 0.0 <= score <= 1.0:
                    return {
                        "score": score,
                        "rationale": rationale_match.group(1).replace('\\"', '"')
                    }
            
            # 备用解析方案：尝试提取纯数字评分
            numeric_score = re.search(r'\b(\d\.\d)\b', response)
            if numeric_score:
                score = float(numeric_score.group(1))
                if 0.0 <= score <= 1.0:
                    # 提取完整引号内容作为理由
                    quote_match = re.search(r'"([^"]+)"', response)
                    return {
                        "score": score,
                        "rationale": quote_match.group(1) if quote_match else "Partial match found"
                    }
                    
            raise ValueError("Field extraction failed")
            
        except Exception as e:
            logger.error(f"Parse error with regex: {str(e)}")
            return {
                "score": -1.0,
                "rationale": f"Parsing failed: {str(e)}"
            }

    def _call_llm(self, prompt:str) -> str:

        self.client.call(user_prompt=prompt)
        response = self.client.response

        return response

    def run_one(self, triple_data:dict) -> dict:
        """
        处理单个三元组
        """
        triple_score = self.score_triple(triple_data)
        return triple_score

    def run(self) -> dict:
        """
        批量处理三元组（多线程加速）
        :return: 处理后的三元组列表
        """
        results = []
        scores = []
        
        # 使用线程池并发处理
        # max_workers = multiprocessing.cpu_count() - 1 
        max_workers = 64
        logger.info(f"Using {max_workers} threads for processing.")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 创建future到data的映射
            future_to_data = {}
            for data in self.triples:
                # 提前处理source_text
                data["source_text"] = self.triple_sources[(data["page_idx"], data["paragraph_idx"])]
                future = executor.submit(self.run_one, data)
                future_to_data[future] = data

            # 进度条跟踪
            for future in tqdm(as_completed(future_to_data), 
                              total=len(future_to_data), 
                              desc="Scoring triples..."):
                data = future_to_data[future]
                try:
                    result = future.result()
                    data["scores"] = result
                    results.append(data)
                    scores.append(result["score"])
                except Exception as e:
                    logger.error(f"Processing error: {str(e)}")
                    continue

        logger.info(f"Scoring complete. Results num: {len(results)}")

        io_file.write(self.output_path, results, mode="w")

        logger.info(f"Results saved to: {self.output_path}")
    
        if scores:
            valid_scores = [score for score in scores if score != -1.0]
            logger.info(f"Valid scores count: {len(valid_scores)} | Invalid score count: {len(scores) - len(valid_scores)} | Total scores count: {len(scores)}")
            if valid_scores:
                logger.info(f"Average score: {sum(valid_scores) / len(valid_scores):.2f}")
            else:
                logger.info("No valid scores found.")

        return results


if __name__ == "__main__":
    
    # 测试文件
    triple_source_path = "data/wtr03_e_by_page_block-head_100.jsonl"
    triple_path = "data/processed_wtr_reports-kg-vllm-test_qwen2.5-7B/wtr03_e_by_page_block-head_100-all_dbpedia_head_entity/new_triples_wtr03_e_by_page_block-head_100.jsonl"
    triple_score_path = triple_path.replace(".jsonl", "-score.jsonl")

    triple_scorer = TripleScorer(triple_path=triple_path, triple_soure_path=triple_source_path, output_path=triple_score_path)

    triple_scorer.run()


        




