# -*- coding: utf-8 -*-
"""Decoupling(ICLR 2020) 论文的评测集：10 个问答对。

基于论文《Decoupling Representation and Classifier for Long-Tailed Recognition》，
每个 ground_truth 都能从论文文本里查到。
"""

EVAL_SET_DECOUPLING = [
    {
        "question": "这篇论文解决的核心问题是什么？",
        "ground_truth": "长尾分布（类别不平衡）对深度学习分类模型的挑战。",
    },
    {
        "question": "论文提出的核心方法是什么？",
        "ground_truth": "把学习过程解耦成表征学习（representation learning）和分类器学习（classification）两部分。",
    },
    {
        "question": "现有方法通常采用什么策略来处理类别不平衡？",
        "ground_truth": "class-balancing 策略，如 loss 重加权、数据重采样、从头部类到尾部类的迁移学习，且大多是联合学习表征和分类器。",
    },
    {
        "question": "论文的第一个惊人发现是什么？",
        "ground_truth": "数据不平衡可能不是学习高质量表征的问题。",
    },
    {
        "question": "用什么采样方式学习表征最好？",
        "ground_truth": "instance-balanced（自然）采样。",
    },
    {
        "question": "表征学好后如何调整分类器来获得强的长尾识别能力？",
        "ground_truth": "用 class-balanced 采样重新训练分类器（cRT），或用 learnable weight scaling（LWS）。",
    },
    {
        "question": "cRT 是什么？",
        "ground_truth": "classifier Re-Training，表征学好后用 class-balanced 采样重新训练分类器。",
    },
    {
        "question": "LWS 是什么？",
        "ground_truth": "learnable weight scaling，对分类器权重做归一化并学习缩放系数。",
    },
    {
        "question": "论文在哪些基准数据集上验证了方法？",
        "ground_truth": "ImageNet-LT、Places-LT、iNaturalist。",
    },
    {
        "question": "论文的主要结论是什么？",
        "ground_truth": "用解耦表征和分类器的简单方法，能超越精心设计的损失函数、采样策略，甚至带 memory 的复杂模块。",
    },
]
