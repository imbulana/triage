from collections import Counter, defaultdict
from dataclasses import field
import json
import os
import time
import logging
import numpy as np
import pymysql
import tiktoken
from tqdm import tqdm
import yaml
from dotenv import load_dotenv
from openai import  OpenAI
from database_utils import build_vector_search,search_vector_search,find_tree_root,\
    search_nodes_link,search_nodes,search_community,search_chunks,get_text_units,find_path
from llm_settings import load_llm_settings
from prompt import GRAPH_FIELD_SEP, PROMPTS
from itertools import combinations

logger=logging.getLogger(__name__)
load_dotenv()
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
LLM_SETTINGS = load_llm_settings()
OPENAI_API_KEY = LLM_SETTINGS["api_key"]
OPENAI_BASE_URL = LLM_SETTINGS["base_url"]
OPENAI_EMBEDDING_MODEL = LLM_SETTINGS["embedding_model"]
LEANRAG_RESPONSE_MAX_TOKENS = int(config.get("model_params", {}).get("leanrag_response_max_tokens", 220))
LEANRAG_CONTEXT_MAX_TOKENS = int(config.get("model_params", {}).get("leanrag_context_max_tokens", 4200))
TOTAL_TOKEN_COST = 0
TOTAL_API_CALL_COST = 0

def embedding(texts: list[str]) -> np.ndarray:
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
    return np.array(final_embedding)

tokenizer = tiktoken.get_encoding("cl100k_base")
def truncate_text(text, max_tokens=4096):
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    truncated_text = tokenizer.decode(tokens)
    return truncated_text

def truncate_section(text, max_tokens):
    if max_tokens <= 0:
        return ""
    return truncate_text(text or "", max_tokens=max_tokens)

def build_budgeted_context(entity_descriptions, aggregation_descriptions, reasoning_path_information_description, text_units):
    sections = [
        ("entity_information", entity_descriptions, 900),
        ("aggregation_entity_information", aggregation_descriptions, 800),
        ("reasoning_path_information", reasoning_path_information_description, 900),
        ("text_units", text_units, 1600),
    ]
    total_budget = LEANRAG_CONTEXT_MAX_TOKENS
    used = 0
    rendered = []
    truncated_sections = []
    for name, value, preferred_budget in sections:
        remaining = max(0, total_budget - used)
        section_budget = min(preferred_budget, remaining)
        original_tokens = len(tokenizer.encode(value or ""))
        section_text = truncate_section(value, section_budget)
        used += len(tokenizer.encode(section_text))
        if original_tokens > section_budget:
            truncated_sections.append(name)
        rendered.append(f"    {name}:\n    {section_text}")
    describe = "\n".join(rendered)
    return describe, truncated_sections

def get_reasoning_chain(global_config,entities_set):
    maybe_edges=list(combinations(entities_set,2))
    reasoning_path=[]
    reasoning_path_information=[]
    db_name=global_config['working_dir'].split("/")[-1]
    information_record=[]
    for edge in maybe_edges:
        a_path=[]
        b_path=[]
        node1=edge[0]
        node2=edge[1]
        node1_tree=find_tree_root(db_name,node1)
        node2_tree=find_tree_root(db_name,node2)
        
        # if node1_tree[1]!=node2_tree[1] :
        #     print("debug")
        for index,(i,j) in enumerate(zip(node1_tree,node2_tree)):
            if i==j:
                a_path.append(i)
                break
            if i in b_path or j in a_path:
                break
            if i!=j :
                a_path.append(i)
                b_path.append(j)
            
            
        reasoning_path.append(a_path+[b_path[len(b_path)-1-i] for  i in range(len(b_path))]) 
        a_path=list(set(a_path))
        b_path=list(set(b_path))
        for maybe_edge in list(combinations(a_path+b_path,2)):
            if maybe_edge[0]==maybe_edge[1]:
                continue
            information=search_nodes_link(maybe_edge[0],maybe_edge[1],global_config['working_dir'])
            if information==None:
                continue
            information_record.append(information)
            reasoning_path_information.append([maybe_edge[0],maybe_edge[1],information[2]])
    # columns=['src_tgt','tgt_src','path_description']
    # reasoning_path_information_description="\t\t".join(columns)+"\n"
    temp_relations_information=list(set([information[2] for information in reasoning_path_information]))
    reasoning_path_information_description="\n".join(temp_relations_information)  
    return  reasoning_path,reasoning_path_information_description

def get_entity_description(global_config,entities_set,mode=0):
    
    
    
    columns=['entity_name','parent','description']
    entity_descriptions="\t\t".join(columns)+"\n"
    entity_descriptions+="\n".join([information[0]+"\t\t"+information[1]+"\t\t"+information[2] for information in entities_set])

    return entity_descriptions
        
def get_aggregation_description(global_config,reasoning_path,if_findings=False):
    
    aggregation_results=[]
    
    communities=set([community for each_path in reasoning_path for community in each_path])
    for community in communities:
        temp=search_community(community,global_config['working_dir'])
        if temp=="":
            continue
        aggregation_results.append(temp)
    if if_findings:
        columns=['entity_name','entity_description','findings']
        aggregation_descriptions="\t\t".join(columns)+"\n"
        aggregation_descriptions+="\n".join([information[0]+"\t\t"+str(information[1])+"\t\t"+information[2] for information in aggregation_results])
    else:
        columns=['entity_name','entity_description']
        aggregation_descriptions="\t\t".join(columns)+"\n"
        aggregation_descriptions+="\n".join([information[0]+"\t\t"+str(information[1]) for information in aggregation_results])
    return aggregation_descriptions,communities
def query_graph(global_config,db,query):
    use_llm_func: callable = global_config["use_llm_func"]
    embedding: callable=global_config["embeddings_func"]
    b=time.time()
    level_mode=global_config['level_mode']
    topk=global_config['topk']
    chunks_file=global_config["chunks_file"]
    entity_results=search_vector_search(global_config['working_dir'],embedding(query),topk=topk,level_mode=level_mode)
    v=time.time()
    res_entity=[i[0]for i in entity_results]
    chunks=[i[-1]for i in entity_results]
    entity_descriptions=get_entity_description(global_config,entity_results)
    reasoning_path,reasoning_path_information_description=get_reasoning_chain(global_config,res_entity)
    # reasoning_path,reasoning_path_information_description=get_path_chain(global_config,res_entity)
    aggregation_descriptions,aggregation=get_aggregation_description(global_config,reasoning_path)
    # chunks=search_chunks(global_config['working_dir'],aggregation)
    text_units=get_text_units(global_config['working_dir'],chunks,chunks_file,k=5)
    describe,truncated_sections=build_budgeted_context(
        entity_descriptions,
        aggregation_descriptions,
        reasoning_path_information_description,
        text_units,
    )
    e=time.time()
    
    # print(describe)
    sys_prompt =PROMPTS["rag_response"].format(context_data=describe)
    evidence_prompt = (
        "Evidence request:\n"
        f"{query}\n\n"
        "Return only a concise KG evidence summary supported by the data tables. "
        "Do not provide advice, disclaimers, or account-specific caveats."
    )
    response=use_llm_func(
        evidence_prompt,
        system_prompt=sys_prompt,
        max_tokens=LEANRAG_RESPONSE_MAX_TOKENS,
        __trace_name="llm.leanrag_response",
        __trace_metadata={
            "purpose": "leanrag_evidence_response",
            "working_dir": global_config.get("working_dir"),
            "topk": topk,
            "level_mode": level_mode,
            "context_chars": len(describe),
            "context_tokens_estimate": len(tokenizer.encode(describe)),
            "truncated_sections": truncated_sections,
        },
    )
    g=time.time()
    logger.info("embedding time: %.2fs", v - b)
    logger.info("query time: %.2fs", e - v)
    logger.info("response time: %.2fs", g - e)
    return describe,response
if __name__=="__main__":
    from database_utils import _mysql_connection

    db = _mysql_connection()
    global_config={}
    WORKING_DIR = f"/data/zyz/trag_ds/exp/lean_full_cs10_top10_chunk5/mix"
    global_config['chunks_file']="/data/zyz/trag_ds/hi_ex/mix/kv_store_text_chunks.json"
    global_config['embeddings_func']=embedding
    global_config['working_dir']=WORKING_DIR
    global_config['topk']=10
    global_config['level_mode']=1
    chat_model = LLM_SETTINGS["chat_model"]
    chat_client_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        chat_client_kwargs["base_url"] = OPENAI_BASE_URL
    chat_client = OpenAI(**chat_client_kwargs)

    def _openai_generate_text(prompt, system_prompt=None, history_messages=None, **kwargs):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        response = chat_client.chat.completions.create(model=chat_model, messages=messages, **kwargs)
        return response.choices[0].message.content or ""

    global_config['use_llm_func']=_openai_generate_text
    query="What is the maturity date of the credit agreement?"
    topk=10
    ref,response=query_graph(global_config,db,query)
    print(ref)
    print("#"*20)
    print(response)
    db.close()
    
