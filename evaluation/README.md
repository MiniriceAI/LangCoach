# LangCoach Evaluation Framework

## 概述

LangCoach Evaluation Framework 是一个全面的评估系统，用于衡量 LangCoach Agent 的性能和质量。

### 核心功能

1. **E2E Audio Latency 测量** - 衡量完整 Audio Pipeline 的端到端延迟
2. **模块化评估** - 独立评估 STT、LLM、TTS 各个模块
3. **对比测试** - 支持不同 LLM Provider 和 TTS 模式的 A/B 测试
4. **固定基准数据集** - 100 条固定测试数据，支持配置运行前 n 条
5. **多格式报告** - 支持 JSON、Markdown、HTML 格式的测试报告
6. **LLM-as-a-Judge 质量评估** - 使用高级 LLM 评估对话质量（8 项指标）
7. **Daily Quality Index (DQI)** - 每日质量指数监控和告警

### 目标指标

#### 性能指标 (Performance Metrics)

| 指标 | 目标值 | 说明 |
|------|--------|------|
| E2E Audio Latency | < 3000ms | 从用户停止说话到 AI 开始播放音频 |
| LLM TTFT | < 1500ms | Time to First Token |
| Success Rate | > 99.5% | 请求成功率 |

#### 质量指标 (Quality Metrics)

| 指标 | 目标值 | 说明 |
|------|--------|------|
| Daily Quality Index (DQI) | ≥ 85% | 每日质量指数 |
| Q1: Role Adherence | ≥ 85% | 角色一致性 |
| Q2: Tone Consistency | ≥ 85% | 语气一致性 |
| Q3: Turn Count Limit | ≥ 85% | 对话轮次控制 |
| Q4: Level Appropriateness | ≥ 85% | 难度适配性 |
| Q5: Correction Behavior | ≥ 85% | 纠错行为 |
| Q6: Brevity & Encouragement | ≥ 85% | 简洁性与鼓励性 |
| Q7: Language Quality | ≥ 85% | 语言质量 |
| Q8: Safety Filter | ≥ 85% | 安全过滤 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

确保 `.env` 文件中配置了必要的环境变量：

```bash
# LLM Provider (至少配置一个)
DEEPSEEK_API_KEY=your_key_here
# 或
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=your_model

# Speech API
SPEECH_API_URL=http://localhost:8600

# Judge LLM for Quality Evaluation (可选)
JUDGE_PROVIDER=openai
JUDGE_MODEL=gpt-4o
OPENAI_API_KEY=your_key_here
# 或使用 Anthropic Claude
# JUDGE_PROVIDER=anthropic
# JUDGE_MODEL=claude-3-opus-20240229
# ANTHROPIC_API_KEY=your_key_here
```

### 3. 启动 Speech API（如需测试 STT/TTS）

```bash
python -m src.api.speech_api
```

### 4. 运行评估

#### 性能评估 (Performance Evaluation)

```bash
# 快速测试（5 条样本）
python -m evaluation.run_eval --quick

# 完整评估（100 条样本）
python -m evaluation.run_eval

# 仅评估 LLM
python -m evaluation.run_eval --module llm

# 评估 E2E Pipeline
python -m evaluation.run_eval --module e2e

# 对比测试
python -m evaluation.run_eval --compare --providers deepseek ollama
```

#### 质量评估 (Quality Evaluation)

```bash
# 运行每日质量评估（默认评估昨天的对话）
python -m evaluation.run_eval --module quality

# 评估指定日期的对话
python -m evaluation.run_eval --module quality --eval-date 2026-01-29

# 自定义参数
python -m evaluation.run_eval --module quality \
  --eval-date 2026-01-29 \
  --samples 50 \
  --dqi-threshold 85.0 \
  --judge-provider openai \
  --judge-model gpt-4o

# 直接运行质量评估（生成告警报告）
python -m evaluation.runners.quality_runner \
  --date 2026-01-29 \
  --samples 50 \
  --alert-report
```

## 目录结构

```
evaluation/
├── __init__.py
├── run_eval.py              # CLI 入口
├── benchmark/
│   ├── __init__.py
│   ├── dataset.py           # 基准数据集管理
│   ├── conversation_logs.py # 对话日志管理
│   └── data/
│       └── benchmark_samples.json  # 100 条固定测试数据
├── evaluators/
│   ├── __init__.py
│   ├── base.py              # 基础评估器类
│   ├── llm_evaluator.py     # LLM 评估器
│   ├── tts_evaluator.py     # TTS 评估器
│   ├── stt_evaluator.py     # STT 评估器
│   ├── e2e_evaluator.py     # E2E Pipeline 评估器
│   └── quality_evaluator.py # LLM-as-a-Judge 质量评估器
├── runners/
│   ├── __init__.py
│   ├── evaluation_runner.py # 评估运行器
│   ├── quality_runner.py    # 质量评估运行器
│   └── comparison_runner.py # 对比测试运行器
├── reports/
│   ├── __init__.py
│   ├── report_generator.py  # 报告生成器
│   └── results/             # 评估结果输出目录
└── logs/
    └── daily/               # 每日对话日志（按日期组织）
```

## 使用指南

### 命令行参数

```bash
python -m evaluation.run_eval [OPTIONS]

选项:
  --module, -m {llm,tts,stt,e2e,quality,all}  评估模块 (默认: all)
  --samples, -n INT                   样本数量 (默认: 100)
  --quick, -q                         快速模式 (5 条样本)
  --provider, -p STR                  LLM Provider (ollama/deepseek/openai)
  --compare, -c                       运行对比测试
  --providers STR [STR ...]           对比的 Provider 列表
  --tts-mode {fast,local}             TTS 模式 (默认: fast)
  --judge-provider STR                Judge LLM Provider (openai/anthropic/ollama)
  --judge-model STR                   Judge Model (gpt-4o/claude-3-opus)
  --eval-date STR                     质量评估日期 (YYYY-MM-DD)
  --dqi-threshold FLOAT               DQI 告警阈值 (默认: 85.0)
  --output, -o STR                    输出目录
  --report, -r [FORMAT ...]           生成报告格式 (json/md/html/txt)
  --verbose, -v                       详细输出
  --silent, -s                        静默模式
```

### 示例

#### 1. 运行完整评估

```bash
python -m evaluation.run_eval
```

输出示例：
```
============================================================
 LangCoach Evaluation Framework
 2024-01-15 10:30:00
============================================================

============================================================
LLM Evaluation
Provider: auto
Samples: 100
============================================================

Progress: 100/100 (100.0%)

Results:
  Success Rate: 100.0%
  Mean Latency: 850ms
  P95 Latency: 1200ms
```

#### 2. 仅评估 LLM 模块

```bash
python -m evaluation.run_eval --module llm --samples 20
```

#### 3. 对比 DeepSeek 和 Ollama

```bash
python -m evaluation.run_eval --compare --providers deepseek ollama -n 50
```

输出示例：
```
============================================================
LLM Provider Comparison
Providers: deepseek, ollama
Samples: 50
============================================================

[deepseek] Progress: 50/50 (100.0%)
[ollama] Progress: 50/50 (100.0%)

============================================================
Comparison Results:
============================================================

deepseek:
  Success Rate: 100.0%
  Mean Latency: 650ms
  P95 Latency: 950ms

ollama:
  Success Rate: 100.0%
  Mean Latency: 1200ms
  P95 Latency: 1800ms

🏆 Fastest: deepseek (650ms mean)
```

#### 4. E2E Pipeline 评估

```bash
python -m evaluation.run_eval --module e2e --report html
```

输出示例：
```
============================================================
E2E Pipeline Evaluation
LLM Provider: auto
TTS Mode: Edge-TTS (fast)
Samples: 100
Target Latency: < 3000ms
============================================================

Progress: 100/100 (100.0%)

Results:
  Success Rate: 98.0%
  Within Target (<3s): 95.0%

  Timing Breakdown:
    STT Mean: 800ms
    LLM Mean: 900ms
    TTS Mean: 400ms
    Total Mean: 2100ms
    Total P95: 2800ms
```

#### 5. 生成多格式报告

```bash
python -m evaluation.run_eval --report json md html
```

### 编程接口

```python
from evaluation.runners import EvaluationRunner, ComparisonRunner
from evaluation.reports import ReportGenerator

# 创建评估运行器
runner = EvaluationRunner()

# 运行 LLM 评估
result = runner.run_llm_evaluation(n_samples=50, provider="deepseek")
print(f"Mean Latency: {result.timing.mean * 1000:.0f}ms")

# 运行 E2E 评估
e2e_result = runner.run_e2e_evaluation(n_samples=20)
print(f"Within Target: {e2e_result.extra_metrics['within_target_rate']:.1f}%")

# 对比测试
comparison = ComparisonRunner()
results = comparison.compare_llm_providers(["deepseek", "ollama"], n_samples=30)

# 生成报告
generator = ReportGenerator()
saved = generator.save_report(results, "comparison", formats=["html", "md"])
```

## 基准数据集

### 数据集结构

基准数据集包含 100 条固定的测试样本，分为 4 个场景：

| 场景 | 样本数 | 难度 |
|------|--------|------|
| Job Interview | 25 | Medium |
| Hotel Check-in | 25 | Primary |
| Renting | 25 | Medium |
| Salary Negotiation | 25 | Advanced |

### 自定义数据集

数据集存储在 `evaluation/benchmark/data/benchmark_samples.json`，首次运行时自动生成。

可以通过修改 `dataset.py` 中的 `_create_default_dataset()` 方法来自定义数据集。

### 运行部分样本

```bash
# 运行前 10 条
python -m evaluation.run_eval -n 10

# 运行前 5 条（快速模式）
python -m evaluation.run_eval --quick
```

## 评估指标

### 时间指标

| 指标 | 说明 |
|------|------|
| Mean | 平均延迟 |
| Median | 中位数延迟 |
| P50/P90/P95/P99 | 百分位延迟 |
| Min/Max | 最小/最大延迟 |
| Std | 标准差 |

### E2E 分解指标

| 组件 | 说明 |
|------|------|
| STT Latency | 语音转文字延迟 |
| LLM Latency | LLM 推理延迟 |
| TTS Latency | 文字转语音延迟 |
| Total Latency | 总端到端延迟 |

### 质量指标

| 指标 | 说明 |
|------|------|
| Success Rate | 请求成功率 |
| Within Target Rate | 达到目标延迟的比例 |
| WER (STT) | Word Error Rate |
| RTF (TTS) | Real-Time Factor |

## 报告格式

### JSON 报告

```json
{
  "llm": {
    "evaluator": "LLMEvaluator",
    "provider": "deepseek",
    "model": "deepseek-chat",
    "timing": {
      "count": 100,
      "mean_ms": 850.5,
      "median_ms": 800.0,
      "p95_ms": 1200.0
    },
    "success_rate": 100.0
  }
}
```

### Markdown 报告

生成格式化的 Markdown 表格，适合在 GitHub 或文档中展示。

### HTML 报告

生成带样式的 HTML 页面，包含交互式图表和详细指标。

## 配置选项

### 环境变量

```bash
# 评估输出目录
EVAL_OUTPUT_DIR=evaluation/reports

# E2E 延迟目标（毫秒）
EVAL_TARGET_LATENCY_MS=3000

# TTS 评估模式
EVAL_TTS_MODE=fast

# 报告格式
EVAL_REPORT_FORMATS=json,md,html
```

## 最佳实践

### 1. 基准测试

- 首次运行完整的 100 条样本建立基准
- 保存结果作为后续对比的参考

### 2. 迭代优化

- 修改配置后运行快速测试 (`--quick`)
- 确认改进后运行完整测试

### 3. 对比测试

- 使用相同的样本数进行对比
- 多次运行取平均值减少波动

### 4. 持续集成

- 在 CI/CD 中集成评估
- 设置性能回归告警

## 故障排除

### 常见问题

1. **Speech API 连接失败**
   ```
   确保 Speech API 正在运行：
   python -m src.api.speech_api
   ```

2. **LLM Provider 不可用**
   ```
   检查 .env 配置和 API Key
   运行: python -c "from agents.llm_factory import list_available_providers; print(list_available_providers())"
   ```

3. **内存不足**
   ```
   减少样本数: --samples 10
   或使用 API 模式而非本地模型
   ```

## 扩展开发

### 添加新的评估器

1. 继承 `BaseEvaluator` 类
2. 实现 `initialize()`, `evaluate_single()`, `get_provider_info()` 方法
3. 在 `evaluators/__init__.py` 中导出

### 添加新的报告格式

1. 在 `ReportGenerator` 中添加 `generate_xxx_report()` 方法
2. 在 `save_report()` 中添加格式处理

## 版本历史

- **v1.0.0** - 初始版本
  - 支持 STT、LLM、TTS、E2E 评估
  - 支持 DeepSeek、Ollama、OpenAI 对比
  - 支持 JSON、Markdown、HTML 报告
