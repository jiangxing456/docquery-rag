# SecQuery：网络安全知识库问答系统

SecQuery 用检索增强生成（RAG）回答安全文档中的问题。上传 CVE 资料、安全公告或合规文档 PDF 后，可以直接提问；回答由检索到的文档片段和 DeepSeek 生成。

## 功能

- 使用本地 BGE-m3 模型生成中英文文本向量，存入 ChromaDB。
- 按文件内容哈希跳过重复上传；同名文件更新时替换旧索引。
- 检索距离超过阈值时拒答，避免对无关问题编造答案。
- 对相似问题复用语义缓存，减少重复调用模型；优先使用 Redis，未连接时使用内存缓存。
- 在页面侧边栏输入 DeepSeek API Key，无须写入代码。

## 工作流程

```text
PDF → 文本解析 → 分块（800 字符，重叠 120）→ BGE-m3 向量化 → ChromaDB
问题 → 语义缓存 → 检索与拒答判断 → 相关片段 → DeepSeek → 回答
```

界面使用 Streamlit，检索链使用 LangChain。向量库保存在本地 `rag_db/`；生成回答时，检索到的文档内容会发送给 DeepSeek API。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

启动后，在侧边栏输入 DeepSeek API Key，上传 PDF 并点击 **Submit & Process**，然后在主界面提问。首次运行需要下载 BGE-m3 模型；不配置 Redis 时会使用进程内缓存。

## 当前限制

- 仅支持有文字层的 PDF；扫描件需要先做 OCR。
- 拒答阈值 `1.11` 根据小规模样本设定，换文档集合后需要重新验证。
- 回答未附带原文页码或引用；重要结论仍应对照原始文档核查。
