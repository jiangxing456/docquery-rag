# -*- coding: utf-8 -*-
"""混合检索（BM25 + 向量）+ reranker：两路召回 RRF 融合，bge-reranker 精排。

对比 baseline 纯向量检索，看 context_recall 能否提升。
"""
import os, sys, json, re
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv("D:/langchain/.env")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from rank_bm25 import BM25Okapi

from eval_questions import EVAL_SET

emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True},
)
db = Chroma(collection_name="mixed_database", embedding_function=emb, persist_directory="./rag_db")
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0.3,
)

# 读全部 chunk
all_data = db.get(include=["documents"])
docs = all_data["documents"]
doc_to_idx = {d: i for i, d in enumerate(docs)}


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


bm25 = BM25Okapi([tokenize(d) for d in docs])
print("BM25 索引就绪，文档数:", len(docs))


def rrf_fuse(vec_idxs, bm25_idxs, k=60):
    scores = {}
    for rank, idx in enumerate(vec_idxs):
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(bm25_idxs):
        scores[idx] = scores.get(idx, 0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


def hybrid_search(query, k=5):
    # BM25 路
    bm25_top = bm25.get_top_n(tokenize(query), docs, n=20)
    bm25_idxs = [doc_to_idx[d] for d in bm25_top]
    # 向量路
    vec_hits = db.similarity_search(query, k=20)
    vec_idxs = [doc_to_idx[h.page_content] for h in vec_hits if h.page_content in doc_to_idx]
    # RRF 融合两路排名，取 Top-k（reranker 精排因模型下载失败暂跳过）
    fused = rrf_fuse(vec_idxs, bm25_idxs)
    return [docs[i] for i, _ in fused[:k]]


SYSTEM_PROMPT = (
    "你是一名网络安全知识库助手，只能基于下面的上下文回答问题。\n"
    "上下文：\n{context}\n\n"
    "问题：{question}\n\n"
    "要求：漏洞编号、术语必须与上下文原文一致；上下文没有答案就明确说未找到；"
    "不要提供上下文没有的信息。"
)

results = []
for i, item in enumerate(EVAL_SET, 1):
    q = item["question"]
    contexts = hybrid_search(q)
    prompt = SYSTEM_PROMPT.format(context="\n\n".join(contexts), question=q)
    answer = llm.invoke(prompt).content
    results.append({
        "question": q,
        "ground_truth": item["ground_truth"],
        "contexts": contexts,
        "answer": answer,
    })
    print(f"[{i}/{len(EVAL_SET)}] {q[:30]}... -> {answer[:50]}")

with open("eval_results_hybrid.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\n已保存 eval_results_hybrid.json")
