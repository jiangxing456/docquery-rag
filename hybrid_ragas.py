# -*- coding: utf-8 -*-
"""评测第二步：读 eval_results_hybrid.json，用 Ragas 跑四个指标。

指标说明：
- faithfulness       答案是否忠实于检索到的上下文（有没有编造）
- context_recall     标准答案的信息有没有被检索到（召回）
- answer_relevancy   答案是否切题
- context_precision  检索到的上下文里有多少真的相关（精确）
"""
import os, sys, json
sys.stdout.reconfigure(encoding='utf-8')

from dotenv import load_dotenv
load_dotenv("D:/langchain/.env")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ragas 的 llms/base.py 裸 import 了 VertexAI（Google 的），本机没装。
# 评测只用 DeepSeek（ChatOpenAI），用 stub 让 import 通过即可。
import sys as _sys
import types as _types
_fake_vai = _types.ModuleType("langchain_community.chat_models.vertexai")
_fake_vai.ChatVertexAI = type("_FakeChatVertexAI", (), {})
_sys.modules["langchain_community.chat_models.vertexai"] = _fake_vai
import langchain_community.llms as _llms_mod
if not hasattr(_llms_mod, "VertexAI"):
    _llms_mod.VertexAI = type("_FakeVertexAI", (), {})

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import Faithfulness, ContextRecall, AnswerRelevancy, ContextPrecision

results = json.load(open("eval_results_hybrid.json", encoding="utf-8"))

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)
emb = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    encode_kwargs={"normalize_embeddings": True},
)

samples = [
    SingleTurnSample(
        user_input=r["question"],
        response=r["answer"],
        retrieved_contexts=r["contexts"],
        reference=r["ground_truth"],
    )
    for r in results
]
dataset = EvaluationDataset(samples=samples)

print("评测中（LLM 当裁判，10 条 × 4 指标）...")
result = evaluate(
    dataset,
    metrics=[Faithfulness(), ContextRecall(), AnswerRelevancy(), ContextPrecision()],
    llm=llm,
    embeddings=emb,
)

df = result.to_pandas()
cols = ["user_input", "faithfulness", "context_recall", "answer_relevancy", "context_precision"]
print("\n逐条结果：")
print(df[cols].to_string(index=False))

print("\n平均指标：")
for c in ["faithfulness", "context_recall", "answer_relevancy", "context_precision"]:
    print(f"  {c:<20} {df[c].mean():.3f}")

df.to_csv("eval_metrics_hybrid.csv", index=False, encoding="utf-8-sig")
print("\n已保存 eval_metrics_hybrid.csv")
