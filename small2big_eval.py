# -*- coding: utf-8 -*-
"""small2big 检索 + 生成：子块精确检索 → 找父块 → 父块全文做上下文生成答案。"""
import os, sys, json
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv("D:/langchain/.env")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

from eval_questions import EVAL_SET

emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True},
)
sub_db = Chroma(collection_name="small2big", embedding_function=emb, persist_directory="./rag_db")
parent_db = Chroma(collection_name="mixed_database", embedding_function=emb, persist_directory="./rag_db")
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

SYSTEM_PROMPT = (
    "你是一名网络安全知识库助手，只能基于下面的上下文回答问题。\n"
    "上下文：\n{context}\n\n"
    "问题：{question}\n\n"
    "要求：漏洞编号、术语必须与上下文原文一致；上下文没有答案就明确说未找到；"
    "不要提供上下文没有的信息。"
)


def retrieve_parents(query, k=8):
    """子块检索 Top-k → 去重取 parent_id → 返回父块全文列表。"""
    sub_hits = sub_db.similarity_search(query, k=k)
    parent_ids = []
    for d in sub_hits:
        pid = d.metadata.get("parent_id")
        if pid and pid not in parent_ids:
            parent_ids.append(pid)
    if not parent_ids:
        return []
    parent_data = parent_db.get(ids=parent_ids, include=["documents"])
    return parent_data["documents"]


results = []
for i, item in enumerate(EVAL_SET, 1):
    q = item["question"]
    contexts = retrieve_parents(q)
    prompt = SYSTEM_PROMPT.format(context="\n\n".join(contexts), question=q)
    answer = llm.invoke(prompt).content
    results.append({
        "question": q,
        "ground_truth": item["ground_truth"],
        "contexts": contexts,
        "answer": answer,
    })
    print(f"[{i}/{len(EVAL_SET)}] {q[:30]}... -> {answer[:50]}")

with open("eval_results_small2big.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n已保存 eval_results_small2big.json")
