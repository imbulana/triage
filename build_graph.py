import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import field
import json
import os
import logging
import time
import numpy as np
import tiktoken
from tqdm import tqdm
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from _cluster_utils import Hierarchical_Clustering
from tools.utils import write_jsonl
from database_utils import build_vector_search,create_db_table_mysql,insert_data_to_mysql
from llm_settings import load_llm_settings
import multiprocessing
logger=logging.getLogger(__name__)
load_dotenv()

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
LLM_SETTINGS = load_llm_settings()
OPENAI_API_KEY = LLM_SETTINGS["api_key"]
OPENAI_BASE_URL = LLM_SETTINGS["base_url"]
OPENAI_CHAT_MODEL = LLM_SETTINGS["commonkg_model"]
OPENAI_EMBEDDING_MODEL = LLM_SETTINGS["embedding_model"]
TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0


def make_llm_func():
    client_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        client_kwargs["base_url"] = OPENAI_BASE_URL
    client = OpenAI(**client_kwargs)

    def _openai_generate_text(prompt, system_prompt=None, history_messages=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=OPENAI_CHAT_MODEL,
            messages=messages,
            **kwargs
        )
        return response.choices[0].message.content or ""

    return _openai_generate_text

def get_common_rag_res(WORKING_DIR):
    entity_path=f"{WORKING_DIR}/entity.jsonl"
    relation_path=f"{WORKING_DIR}/relation.jsonl"
    # i=0
    e_dic={}
    with open(entity_path,"r")as f:
        for xline in f:
            
            line=json.loads(xline)
            entity_name=str(line['entity_name'])
            description=line['description']
            source_id=line['source_id']
            if entity_name not in e_dic.keys():
                e_dic[entity_name]=dict(
                    entity_name=str(entity_name),
                    description=description,
                    source_id=source_id,
                    degree=0,
                )
            else:
                e_dic[entity_name]['description']+="|Here is another description : "+ description
                if e_dic[entity_name]['source_id']!= source_id:
                    e_dic[entity_name]['source_id']+= "|"+source_id
                    
    #         i+=1
    #         if i==1000:
    #             break
    # i=0
    r_dic={}
    with open(relation_path,"r")as f:
        for xline in f:
            
            line=json.loads(xline)
            src_tgt=str(line['src_tgt'])
            tgt_src=str(line['tgt_src'])
            description=line['description']
            weight=1
            source_id=line['source_id']
            r_dic[(src_tgt,tgt_src)]={
                'src_tgt':str(src_tgt),
                'tgt_src':str(tgt_src),
                'description':description,
                'weight':weight,
                'source_id':source_id
            }
            # e_dic[src_tgt]['degree']+=1
            # e_dic[tgt_src]['degree']+=1
            # i+=1
            # if i==1000:
            #     break
    
    logger.info("Loaded %s unique entities and %s unique relations from %s", len(e_dic), len(r_dic), WORKING_DIR)
    return e_dic,r_dic


def embedding(texts: list[str]) -> np.ndarray: #vllm serve
    model_name = OPENAI_EMBEDDING_MODEL
    client_kwargs = {"api_key": LLM_SETTINGS["embedding_api_key"]}
    if LLM_SETTINGS["embedding_base_url"]:
        client_kwargs["base_url"] = LLM_SETTINGS["embedding_base_url"]
    client = OpenAI(**client_kwargs)
    embedding = client.embeddings.create(
        input=texts,
        model=model_name,
    )
    final_embedding = [d.embedding for d in embedding.data]
    text_count = len(texts) if isinstance(texts, list) else 1
    logger.info("Embedded %s texts with model=%s", text_count, model_name)
    return np.array(final_embedding)
def embedding_init(entities:list[dict])-> list[dict]: 
    texts=[truncate_text(i['description']) for i in entities]
    model_name = OPENAI_EMBEDDING_MODEL
    client_kwargs = {"api_key": LLM_SETTINGS["embedding_api_key"]}
    if LLM_SETTINGS["embedding_base_url"]:
        client_kwargs["base_url"] = LLM_SETTINGS["embedding_base_url"]
    client = OpenAI(**client_kwargs)
    embedding = client.embeddings.create(
        input=texts,
        model=model_name,
    )
    final_embedding = [d.embedding for d in embedding.data]
    for i, entity in enumerate(entities):
        entity['vector'] = np.array(final_embedding[i])
    return entities
tokenizer = tiktoken.get_encoding("cl100k_base")
def truncate_text(text, max_tokens=4096):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text
def embedding_data(entity_results):
    entities = [v for k, v in entity_results.items()]
    entity_with_embeddings=[]
    embeddings_batch_size = 64
    num_embeddings_batches = (len(entities) + embeddings_batch_size - 1) // embeddings_batch_size
    logger.info(
        "Preparing embeddings for %s entities across %s batches (batch_size=%s)",
        len(entities),
        num_embeddings_batches,
        embeddings_batch_size,
    )
    
    batches = [
        entities[i * embeddings_batch_size : min((i + 1) * embeddings_batch_size, len(entities))]
        for i in range(num_embeddings_batches)
    ]

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(embedding_init, batch) for batch in batches]
        for index, future in enumerate(tqdm(as_completed(futures), total=len(futures)), start=1):
            result = future.result()
            entity_with_embeddings.extend(result)
            logger.info("Completed embedding batch %s/%s", index, len(futures))

    for i in entity_with_embeddings:
        entiy_name=i['entity_name']
        vector=i['vector']
        entity_results[entiy_name]['vector']=vector
    return entity_results



    
            
def hierarchical_clustering(global_config):
    started_at = time.monotonic()
    logger.info("Starting hierarchical clustering build for %s", global_config['working_dir'])
    entity_results,relation_results=get_common_rag_res(global_config['working_dir'])
    logger.info("Starting embedding stage")
    all_entities=embedding_data(entity_results)
    hierarchical_cluster = Hierarchical_Clustering()
    logger.info("Starting clustering stage with max_workers=%s", global_config['max_workers'])
    all_entities,generate_relations,community =hierarchical_cluster.perform_clustering(global_config=global_config,entities=all_entities,relations=relation_results,\
        WORKING_DIR=WORKING_DIR,max_workers=global_config['max_workers'])
    try :
        logger.info("Starting vector search build")
        all_entities[-1]['vector']=embedding(all_entities[-1]['description'])
        build_vector_search(all_entities, f"{WORKING_DIR}")
        logger.info("Vector search build complete")
    except Exception as e:
        print(f"Error in build_vector_search: {e}")
    for layer in all_entities:
        if type(layer) != list :
            if "vector" in layer.keys():
                del layer["vector"]
            continue
        for item in layer:
            if "vector" in item.keys():
                del item["vector"]
            if len(layer)==1:
                item['parent']='root'
    save_relation=[
    v for k, v in generate_relations.items()
]
    save_community=[
    v for k, v in community.items()
]
    logger.info(
        "Writing generated outputs: relations=%s communities=%s",
        len(save_relation),
        len(save_community),
    )
    write_jsonl(save_relation, f"{global_config['working_dir']}/generate_relations.json")
    write_jsonl(save_community, f"{global_config['working_dir']}/community.json")
    logger.info("Creating MySQL tables for %s", global_config['working_dir'])
    create_db_table_mysql(global_config['working_dir'])
    logger.info("Inserting clustered data into MySQL for %s", global_config['working_dir'])
    insert_data_to_mysql(global_config['working_dir'])
    logger.info("Hierarchical clustering build complete in %.1fs", time.monotonic() - started_at)
    
if __name__=="__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)  # 强制设置
    except RuntimeError:
        pass  # 已经设置过，忽略
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="/data/zyz/LeanRAG/ttt")
    parser.add_argument("-n", "--num", type=int, default=2)
    parser.add_argument("--verbose-logging", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose_logging else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    WORKING_DIR = args.path
    num=args.num
    global_config={}
    global_config['max_workers']=num*4
    global_config['working_dir']=WORKING_DIR
    global_config['use_llm_func']=make_llm_func()
    global_config['embeddings_func']=embedding
    global_config["special_community_report_llm_kwargs"]=field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    hierarchical_clustering(global_config)
