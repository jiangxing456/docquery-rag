# -*- coding: utf-8 -*-
"""测不同阈值下的命中率，并验证是否会误命中（不同问题被错判为相同）。"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",
                           encode_kwargs={"normalize_embeddings": True})

SIMILAR = [  # 同义改写，应该命中
    ("这篇论文解决的核心问题是什么？", "这篇论文主要解决的是什么问题"),
    ("论文提出的核心方法是什么？", "作者提出的核心方法是什么"),
    ("现有方法通常采用什么策略处理类别不平衡？", "以往的做法一般怎么解决类别不平衡"),
    ("用什么采样方式学习表征最好？", "学习表征时用什么采样策略效果最好"),
    ("cRT 是什么？", "cRT 具体指什么"),
    ("LWS 是什么？", "LWS 全称是什么"),
    ("论文在哪些基准数据集上验证了方法？", "实验是在哪些数据集上做的"),
    ("论文的主要结论是什么？", "这篇论文得出了什么结论"),
]

DIFFERENT = [  # 不同问题，绝对不能命中
    ("cRT 是什么？", "LWS 是什么？"),
    ("用什么采样方式学习表征最好？", "论文在哪些数据集上验证？"),
    ("论文提出的核心方法是什么？", "论文发表在哪个期刊？"),
]

def cos(a, b):
    return float(np.asarray(a) @ np.asarray(b))

print("=== 同义改写的相似度（越高越该命中）===")
sim_vals = []
for q1, q2 in SIMILAR:
    s = cos(emb.embed_query(q1), emb.embed_query(q2))
    sim_vals.append(s)
    print(f"  {s:.4f}  {q2[:30]}")
print(f"\n同义改写相似度区间: {min(sim_vals):.4f} ~ {max(sim_vals):.4f}, 均值 {sum(sim_vals)/len(sim_vals):.4f}")

print("\n=== 不同问题的相似度（越低越好，越高越危险）===")
diff_vals = []
for q1, q2 in DIFFERENT:
    s = cos(emb.embed_query(q1), emb.embed_query(q2))
    diff_vals.append(s)
    print(f"  {s:.4f}  {q1[:18]} vs {q2[:18]}")
print(f"\n不同问题相似度区间: {min(diff_vals):.4f} ~ {max(diff_vals):.4f}")

lo, hi = max(diff_vals), min(sim_vals)
print(f"\n=== 安全阈值区间: ({hi:.4f}, {lo:.4f}] ===")
if lo < hi:
    for th in [0.80, 0.85, 0.90, 0.92, 0.95]:
        hit = sum(1 for v in sim_vals if v >= th)
        wrong = sum(1 for v in diff_vals if v >= th)
        print(f"  阈值 {th:.2f}: 命中 {hit}/{len(sim_vals)} ({hit/len(sim_vals)*100:.0f}%), 误命中 {wrong}/{len(diff_vals)}")
else:
    print("  ⚠️ 区间重叠，无法用单一阈值完美区分（这正是语义缓存的固有矛盾）")
