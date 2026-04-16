import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import field
import json
import os
import logging
import numpy as np
from openai import OpenAI
import tiktoken
from tqdm import tqdm
import yaml
from openai import AsyncOpenAI, OpenAI
from _cluster_utils import Hierarchical_Clustering
from tools.utils import write_jsonl,InstanceManager
from database_utils import build_vector_search,create_db_table_mysql,insert_data_to_mysql
import requests
import multiprocessing
logger=logging.getLogger(__name__)

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
MODEL = config['deepseek']['model']
DEEPSEEK_API_KEY = config['deepseek']['api_key']
DEEPSEEK_URL = config['deepseek']['base_url']
EMBEDDING_MODEL = config['glm']['model']
EMBEDDING_URL = config['glm']['base_url']
TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0

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
    
    
    return e_dic,r_dic


def embedding(texts: list[str]) -> np.ndarray: #vllm serve
    model_name = EMBEDDING_MODEL
    client = OpenAI(
        api_key=EMBEDDING_MODEL,
        base_url=EMBEDDING_URL
    ) 
    embedding = client.embeddings.create(
        input=texts,
        model=model_name,
    )
    final_embedding = [d.embedding for d in embedding.data]
    return np.array(final_embedding)
def embedding_init(entities:list[dict])-> list[dict]: 
    texts=[truncate_text(i['description']) for i in entities]
    model_name = EMBEDDING_MODEL
    client = OpenAI(
        api_key=EMBEDDING_MODEL,
        base_url=EMBEDDING_URL
    ) 
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
    
    batches = [
        entities[i * embeddings_batch_size : min((i + 1) * embeddings_batch_size, len(entities))]
        for i in range(num_embeddings_batches)
    ]

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(embedding_init, batch) for batch in batches]
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            entity_with_embeddings.extend(result)

    for i in entity_with_embeddings:
        entiy_name=i['entity_name']
        vector=i['vector']
        entity_results[entiy_name]['vector']=vector
    return entity_results



    
            
def hierarchical_clustering(global_config):
    entity_results,relation_results=get_common_rag_res(global_config['working_dir'])
    all_entities=embedding_data(entity_results)
    hierarchical_cluster = Hierarchical_Clustering()
    all_entities,generate_relations,community =hierarchical_cluster.perform_clustering(global_config=global_config,entities=all_entities,relations=relation_results,\
        WORKING_DIR=WORKING_DIR,max_workers=global_config['max_workers'])
    try :
        all_entities[-1]['vector']=embedding(all_entities[-1]['description'])
        build_vector_search(all_entities, f"{WORKING_DIR}")
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
    write_jsonl(save_relation, f"{global_config['working_dir']}/generate_relations.json")
    write_jsonl(save_community, f"{global_config['working_dir']}/community.json")
    create_db_table_mysql(global_config['working_dir'])
    insert_data_to_mysql(global_config['working_dir'])
    
if __name__=="__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)  # 强制设置
    except RuntimeError:
        pass  # 已经设置过，忽略
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, default="/data/zyz/LeanRAG/ttt")
    parser.add_argument("-n", "--num", type=int, default=2)
    args = parser.parse_args()

    WORKING_DIR = args.path
    num=args.num
    instanceManager=InstanceManager(
        url="http://xxxx",
        ports=[8001 for i in range(num)],
        gpus=[i for i in range(num)],
        generate_model="qwen3_32b",
        startup_delay=30
    )
    global_config={}
    global_config['max_workers']=num*4
    global_config['working_dir']=WORKING_DIR
    global_config['use_llm_func']=instanceManager.generate_text
    global_config['embeddings_func']=embedding
    global_config["special_community_report_llm_kwargs"]=field(
        default_factory=lambda: {"response_format": {"type": "json_object"}}
    )
    hierarchical_clustering(global_config)