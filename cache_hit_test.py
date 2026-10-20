# -*- coding: utf-8 -*-
"""语义缓存命中率测试：10 个标准问题 + 10 个改写问法，看能命中多少。"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from langchain_community.embeddings import HuggingFaceEmbeddings
from semantic_cache import lookup, store, stats, backend_name

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",
                           encode_kwargs={"normalize_embeddings": True})

# (标准问法, 相似改写问法)
PAIRS = [
    ("这篇论文解决的核心问题是什么？", "这篇论文主要解决的是什么问题"),
    ("论文提出的核心方法是什么？", "作者提出的核心方法是什么"),
    ("现有方法通常采用什么策略处理类别不平衡？", "以往的做法一般怎么解决类别不平衡"),
    ("用什么采样方式学习表征最好？", "学习表征时用什么采样策略效果最好"),
    ("cRT 是什么？", "cRT 具体指什么"),
    ("LWS 是什么？", "LWS 全称是什么"),
    ("论文在哪些基准数据集上验证了方法？", "实验是在哪些数据集上做的"),
    ("论文的主要结论是什么？", "这篇论文得出了什么结论"),
]

print(f"后端: {backend_name()}")
print(f"embedding 模型加载完成\n")

# 先全部存入
t0 = time.time()
for q, _ in PAIRS:
    store(q, f"答案::{q}", emb)
t1 = time.time()
print(f"写入 {len(PAIRS)} 条缓存，耗时 {t1-t0:.2f} 秒\n")

# 再用改写问法去查，统计命中
hit = 0
total = 0
t_hit = 0.0
t_miss = 0.0
for q_orig, q_rewrite in PAIRS:
    # 改写问法先算一次 embedding（模拟真实场景的检索开销）
    t2 = time.time()
    got = lookup(q_rewrite, emb)
    dt = time.time() - t2
    total += 1
    if got:
        hit += 1
        t_hit += dt
        print(f"  命中 | {q_rewrite[:26]:<28} 相似查缓存 {dt*1000:.1f} ms")
    else:
        t_miss += dt
        print(f"  未中 | {q_rewrite[:26]:<28} {dt*1000:.1f} ms")

print(f"\n命中率: {hit}/{total} = {hit/total*100:.0f}%")
print(f"命中平均耗时: {t_hit/max(hit,1)*1000:.1f} ms")
print(f"未中平均耗时: {t_miss/max(total-hit,1)*1000:.1f} ms")
print(f"缓存统计: {stats()}")
