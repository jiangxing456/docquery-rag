# -*- coding: utf-8 -*-
"""拒答阈值探测：区分「该答」和「该拒答」两类问题的距离分布。"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"normalize_embeddings": True})
db = Chroma(collection_name="decoupling", embedding_function=emb, persist_directory="./rag_db")

ANSWERABLE = [  # 论文里有答案，应该答
    "cRT 是什么？", "LWS 全称是什么？", "为什么用零样本方法？",
    "论文用什么本体构建知识图谱？", "实验用了哪些数据集？",
    "tri-Clustering 做什么？", "论文的核心贡献是什么？",
]
NOT_ANSWERABLE = [  # 论文里没有，应该拒答
    "今天天气怎么样？", "北京到上海的高铁票价是多少？",
    "Python 怎么安装依赖？", "推荐一部电影",
    "怎么申请护照？", "附近有什么好吃的店？",
    "明天股市会涨吗？",
]

def top_dist(q, k=3):
    r = db.similarity_search_with_score(q, k=k)
    return min(d for _, d in r)   # 取最相似的那个距离

print("=== 该回答的问题（最小距离，越小越相似）===")
a = []
for q in ANSWERABLE:
    d = top_dist(q); a.append(d)
    print(f"  {d:.4f}  {q}")

print("\n=== 该拒答的问题 ===")
b = []
for q in NOT_ANSWERABLE:
    d = top_dist(q); b.append(d)
    print(f"  {d:.4f}  {q}")

print(f"\n该答: min={min(a):.4f} max={max(a):.4f} 均值={np.mean(a):.4f}")
print(f"该拒: min={min(b):.4f} max={max(b):.4f} 均值={np.mean(b):.4f}")
print(f"\n间隔: 该答最大={max(a):.4f} / 该拒最小={min(b):.4f}  ->  {'有安全间隔' if max(a) < min(b) else '⚠️ 区间重叠，单靠距离阈值无法完美区分'}")

# 换算成余弦相似度（归一化向量：L2距离 d → cos = 1 - d^2/2）
print("\n=== 换算成余弦相似度 ===")
ca = [1 - d*d/2 for d in a]; cb = [1 - d*d/2 for d in b]
print(f"该答: min={min(ca):.4f} max={max(ca):.4f}")
print(f"该拒: min={min(cb):.4f} max={max(cb):.4f}")
