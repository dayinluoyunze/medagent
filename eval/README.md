# Evaluation

这个目录用于存放项目评测样本和评测脚本，目标是把“能跑”变成“可量化说明效果”。

## 当前内容

- `qa_dataset.jsonl`: 检索评测样本
- `run_eval.py`: 离线评测脚本
- `answer_dataset.jsonl`: 回答级评测样本
- `run_answer_eval.py`: 回答级评测脚本，需要可用 LLM API Key

## 评测目标

当前脚本优先评估检索层，不强依赖在线模型，便于在本地快速验证：

- Top-K 检索命中率
- 期望来源命中率
- 期望关键词覆盖率

如果本地没有可用 embedding API Key，脚本会自动退回关键词检索逻辑；如果有可用 embedding 配置，则会直接评估向量检索结果。

## 运行方式

在项目根目录执行：

```bash
python eval/run_eval.py
```

也可以指定参数：

```bash
python eval/run_eval.py --k 4 --dataset eval/qa_dataset.jsonl
```

CI 中推荐带最低阈值运行：

```bash
python eval/run_eval.py --min-retrieval-hit-rate 0.6 --min-keyword-coverage-rate 0.4
```

回答级评测示例：

```bash
python eval/run_answer_eval.py --provider minimax --api-key your-minimax-key
```

## 输出说明

- `retrieval_hit_rate`: 至少命中一个期望关键词的比例
- `source_hit_rate`: 期望来源文件命中的比例
- `keyword_coverage_rate`: 所有期望关键词的整体覆盖率
- 阈值参数不达标时脚本返回非零退出码，适合接入 CI

## 后续建议

- 增加人工标注字段，如正确/部分正确/错误
- 对比 `vector search` 与 `keyword fallback` 的差异
- 记录不同 Provider 下的回答质量
