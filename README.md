# MedAgent

一个面向医药问答场景的智能 Agent Demo，支持多模型切换、知识库检索、长期记忆摘要，以及在模型或向量检索不可用时的本地回退。

## 项目定位

这个项目聚焦于一个相对高风险、对答案可信度要求较高的垂直场景：医药知识问答。目标不是做通用聊天，而是验证以下能力：

- 多个 OpenAI 兼容模型接口的统一接入
- 基于本地知识库的 RAG 检索增强
- 会话历史与摘要记忆管理
- 故障回退与基本安全约束

当前实现也适合作为后续扩展评测、引用溯源和安全策略的基础版本。

## 核心功能

- 多模型 Provider 切换：支持 `OpenAI`、`ModelScope`、`MiniMax`
- 多源知识库接入：支持 `md`、`txt`、`json`、`csv`、`docx`、`url`
- RAG 检索：优先使用 `FAISS + Embeddings`
- 回退检索：Embedding 不可用时自动退回本地关键词检索
- 来源展示：回答尾部自动附带命中的知识来源
- 长期记忆：对历史对话进行 markdown 摘要并持久化
- 故障兜底：模型调用失败时回退到本地知识库内容摘要
- Web UI：基于 Streamlit 提供交互式问答界面

## 技术栈

- Python
- Streamlit
- LangChain
- OpenAI SDK
- FAISS
- python-docx
- BeautifulSoup4

## 系统架构

```text
User
  |
  v
Streamlit UI
  |
  v
MedicalAgent
  |------------------------------|
  |                              |
  v                              v
LLM Provider Adapter         Memory Manager
  |                              |
  v                              v
OpenAI / ModelScope /       Markdown History +
MiniMax                     Summary Persistence
  |
  v
Retriever
  |------------------------------|
  |                              |
  v                              v
FAISS Vector Search         Keyword Fallback
  |
  v
Local Knowledge Base
```

## 目录结构

```text
medagent/
├─ app.py
├─ config.py
├─ requirements.txt
├─ agents/
│  └─ medical_agent.py
├─ rag/
│  └─ retriever.py
├─ knowledge/
│  ├─ drugs.md
│  ├─ products.md
│  ├─ qa.md
│  ├─ sample.docx
│  ├─ sample.url
│  └─ sample.urls
└─ memory/
   ├─ conversation_history.md
   └─ conversation_summary.md
```

## 本地启动

### 1. 创建环境并安装依赖

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env`，并至少配置一个可用 Provider 的 API Key。

示例：

```env
LLM_PROVIDER=modelscope
EMBEDDING_PROVIDER=modelscope
MODELSCOPE_API_KEY=your-modelscope-key
```

### 3. 启动应用

```bash
streamlit run app.py
```

默认会在浏览器中打开本地页面。

## 使用流程

1. 在侧边栏选择模型 Provider
2. 输入对应 API Key，或从 `.env` 自动读取
3. 点击“初始化 Agent”
4. 在主界面输入医药相关问题

建议尝试以下问题：

- 二甲双胍可以长期服用吗？
- 忘记服药应该怎么处理？
- 某类降压药适合哺乳期使用吗？

## 当前亮点

- 不是纯聊天壳子，而是包含检索、记忆、回退三条能力链路
- 支持多 Provider 统一接入，便于展示模型适配能力
- 使用本地知识库和索引缓存，具备一定工程完整度
- 问答链路支持本地指标落盘，可记录命中、回退和耗时
- 针对医药场景加入了系统提示词和回答约束

## 当前不足

- 还没有标准化评测集和量化指标
- 缺少自动化测试
- 缺少日志分析和效果监控
- 安全策略主要依赖 prompt，规则层兜底仍不够强

## 建议下一步迭代

- 增加评测脚本和测试数据集，量化 RAG 效果
- 为回答增加更细粒度的命中片段展示
- 增加单元测试与异常场景测试
- 补充日志、耗时统计、命中率统计
- 增强高风险医疗问题的拒答和提醒策略

## 评测

项目已预留基础评测目录，可先从检索层开始量化：

```bash
python eval/run_eval.py
```

当前脚本会读取 [eval/qa_dataset.jsonl](/e:/agent/medagent/eval/qa_dataset.jsonl) 中的样本，对检索结果做基础统计，包括：

- 检索命中率
- 来源命中率
- 关键词覆盖率

评测说明见 [eval/README.md](/e:/agent/medagent/eval/README.md)。

## 测试

项目当前提供基础单元测试，可直接执行：

```bash
pytest
```

覆盖范围包括：

- provider 配置读取
- markdown memory 持久化
- 来源拼接
- 检索回退逻辑

## 日志与指标

每次问答会将基础运行指标写入 `logs/chat_metrics.jsonl`，便于后续做简单分析。

当前落盘字段包括：

- `provider`
- `embedding_provider`
- `knowledge_hit`
- `retrieved_doc_count`
- `source_labels`
- `fallback_used`
- `status`
- `error_type`
- `duration_ms`

## 已知限制

- 本项目仅用于技术演示，不构成医疗建议
- 当前知识库内容较少，答案质量高度依赖样本覆盖
- 不同 Provider 的模型能力和兼容性存在差异
