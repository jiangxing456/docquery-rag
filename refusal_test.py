# -*- coding: utf-8 -*-
"""拒答兜底测试：验证两类问题是否被正确分流。"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import app as A
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True})
db = Chroma(collection_name="decoupling", embedding_function=emb, persist_directory="./rag_db")

def decide(q):
    hits = db.similarity_search_with_score(q, k=3)
    best = min(d for _, d in hits) if hits else float("inf")
    refuse = best > A.REPLY_DISTANCE_THRESHOLD
    return best, ("拒答" if refuse else "回答")

ANSWERABLE = ["cRT 是什么？", "LWS 全称是什么？", "为什么用零样本方法？",
    "论文用什么本体构建知识图谱？", "实验用了哪些数据集？", "tri-Clustering 做什么？", "论文的核心贡献是什么？"]
NOT_ANSWERABLE = ["今天天气怎么样？", "北京到上海的高铁票价是多少？", "Python 怎么安装依赖？",
    "推荐一部电影", "怎么申请护照？", "附近有什么好吃的店？", "明天股市会涨吗？"]

print(f"拒答阈值 = {A.REPLY_DISTANCE_THRESHOLD}\n")
ok_a = ok_r = 0
print("=== 该回答的问题 ===")
for q in ANSWERABLE:
    d, v = decide(q)
    ok = v == "回答"; ok_a += ok
    print(f"  {'✓' if ok else '✗'} {d:.4f} -> {v} | {q}")
print("\n=== 该拒答的问题 ===")
for q in NOT_ANSWERABLE:
    d, v = decide(q)
    ok = v == "拒答"; ok_r += ok
    print(f"  {'✓' if ok else '✗'} {d:.4f} -> {v} | {q}")

print(f"\n该答正确: {ok_a}/{len(ANSWERABLE)}")
print(f"该拒正确: {ok_r}/{len(NOT_ANSWERABLE)}")
print(f"总准确率: {(ok_a+ok_r)/(len(ANSWERABLE)+len(NOT_ANSWERABLE))*100:.0f}%")
