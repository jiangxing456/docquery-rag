# -*- coding: utf-8 -*-
"""语义缓存最终测试：命中率 + 响应时间降幅。"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from dotenv import load_dotenv
load_dotenv("D:/langchain/.env")
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from semantic_cache import lookup, store, stats

emb = HuggingFaceEmbeddings(model_name="BAAI/bge-m3",
                           encode_kwargs={"normalize_embeddings": True})
llm = ChatOpenAI(model="deepseek-chat", api_key=os.getenv("DEEPSEEK_API_KEY"),
                 base_url="https://api.deepseek.com", temperature=0.3)

PAIRS = [
    ("论文提出的核心方法是什么？", "作者提出的核心方法是什么"),
    ("cRT 是什么？", "cRT 具体指什么"),
    ("LWS 是什么？", "LWS 全称是什么"),
    ("论文的主要结论是什么？", "这篇论文得出了什么结论"),
    ("用什么采样方式学习表征最好？", "学习表征时用什么采样策略效果最好"),
    ("这篇论文解决的核心问题是什么？", "这篇论文主要解决的是什么问题"),
]

print("=== 第一次提问（走完整 RAG 链：检索 + 生成）===")
full_times = []
for q, _ in PAIRS:
    t0 = time.time()
    ans = llm.invoke(f"用一句话回答：{q}").content
    dt = time.time() - t0
    full_times.append(dt)
    store(q, ans, emb)
    print(f"  {dt:6.2f} 秒 | {q[:24]}")

print("\n=== 第二次提问（换相似问法，查缓存）===")
cache_times = []
hits = 0
for _, q2 in PAIRS:
    t0 = time.time()
    got = lookup(q2, emb)
    dt = time.time() - t0
    cache_times.append(dt)
    if got:
        hits += 1
    print(f"  {dt*1000:6.1f} 毫秒 | {q2[:24]}")

avg_full = sum(full_times) / len(full_times)
avg_cache = sum(cache_times) / len(cache_times)
print(f"\n命中率: {hits}/{len(PAIRS)} = {hits/len(PAIRS)*100:.0f}%")
print(f"完整链路平均: {avg_full:.2f} 秒")
print(f"命中缓存平均: {avg_cache*1000:.0f} 毫秒")
print(f"提速: {avg_full/(avg_cache+0.0001):.0f}x  （节省 {avg_full - avg_cache:.2f} 秒）")
print(f"缓存统计: {stats()}")
