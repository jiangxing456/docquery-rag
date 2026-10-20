# -*- coding: utf-8 -*-
"""评测集（golden set）：基于已入库的安全论文《Zero-shot fine-grained entity typing
in information security based on ontology》构建的 10 个问答对。

每个条目 = question（问题）+ ground_truth（标准答案，能从论文里查到的真实内容）。
评测时：对 question 检索 Top-5 上下文 → DeepSeek 生成 answer → Ragas 用 LLM 当裁判
对比 answer 与 ground_truth，输出 faithfulness / context_recall 等指标。
"""

EVAL_SET = [
    {
        "question": "这篇论文解决的核心问题是什么？",
        "ground_truth": "信息安全领域缺乏标注实体，论文提出一种基于本体的零样本细粒度实体类型识别方法。",
    },
    {
        "question": "论文用什么本体来做实体类型识别？",
        "ground_truth": "Unified Cybersecurity Ontology（UCO），即统一网络安全本体。",
    },
    {
        "question": "为什么现有的 UCO 不够用？",
        "ground_truth": "UCO 中某些类别标签没有明确的领域特征，某些缩写也无法直接获得。",
    },
    {
        "question": "MEHC 是什么？",
        "ground_truth": "MEHC 是 mixed entities and hierarchy of UCO 的缩写，是一种利用类别层次来表示类别标签的新方法。",
    },
    {
        "question": "tri-Clustering 算法的作用是什么？",
        "ground_truth": "tri-Clustering 是对 k-means 的改进，用于重新聚类粗粒度分类结果、或确定新实体的类型，以提高细粒度实体类型识别的性能。",
    },
    {
        "question": "论文做的是粗粒度还是细粒度的实体类型识别？",
        "ground_truth": "细粒度（fine-grained）实体类型识别。",
    },
    {
        "question": "为什么采用零样本方法而不是有监督方法？",
        "ground_truth": "因为缺少 distant supervision（远程监督），信息安全领域缺乏标注数据，所以采用零样本方法。",
    },
    {
        "question": "论文用 UCO 构建了什么？",
        "ground_truth": "利用 UCO 构建形式化的知识图谱，以填补信息安全领域的空白。",
    },
    {
        "question": "论文与哪些方法做了对比？",
        "ground_truth": "与其他 zero-shot 和 few-shot 的细粒度实体类型识别方法做对比。",
    },
    {
        "question": "这篇论文发表在哪个期刊？",
        "ground_truth": "Knowledge-Based Systems（知识基系统）。",
    },
]
