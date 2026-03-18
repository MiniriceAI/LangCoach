# LangCoach 评估框架使用指南

## 📋 目录

1. [快速开始](#快速开始)
2. [评估模块说明](#评估模块说明)
3. [如何运行评估](#如何运行评估)
4. [如何查看报告](#如何查看报告)
5. [高级用法](#高级用法)
6. [常见问题](#常见问题)

---

## 🚀 快速开始

### 环境准备

```bash
# 1. 激活环境
source /workspace/init_env.sh && conda activate lm

# 2. 进入项目目录
cd /workspace/LangCoach

# 3. 确保依赖已安装
pip install -r requirements.txt
```

### 运行第一个评估

```bash
# 使用便捷脚本（推荐）
./run_evaluation.sh llm 10

# 或直接使用 Python
python -m evaluation.run_eval --module llm --samples 10 --report md json
```

---

## 📊 评估模块说明

### 1. **LLM 评估** (`--module llm`)
- **目标**: 测试 LLM 推理延迟和响应质量
- **指标**: 
  - TTFT (Time to First Token) < 1500ms
  - 成功率 > 99.5%
  - 平均延迟
  - P50/P90/P95/P99 延迟
- **用途**: 对比不同 LLM Provider (DeepSeek, Ollama, OpenAI)

### 2. **STT 评估** (`--module stt`)
- **目标**: 测试语音转文本延迟
- **指标**:
  - 平均处理时间
  - 准确率
- **依赖**: 需要启动 Speech API (`./run_speech_api.sh`)

### 3. **TTS 评估** (`--module tts`)
- **目标**: 测试文本转语音延迟
- **指标**:
  - 音频生成时间
  - 音频质量
- **模式**: Fast (Edge-TTS) / Local (本地模型)
- **依赖**: 需要启动 Speech API

### 4. **E2E Pipeline 评估** (`--module e2e`)
- **目标**: 测试完整 Audio Pipeline
- **流程**: Audio Input → STT → LLM → TTS → Audio Output
- **指标**:
  - 端到端延迟 < 3000ms
  - 各模块耗时分解
  - 整体成功率
- **依赖**: 需要启动 Speech API

### 5. **质量评估** (`--module quality`)
- **目标**: 使用 LLM-as-a-Judge 评估对话质量
- **8 项质量指标**:
  1. Q1: Role Adherence (角色一致性)
  2. Q2: Tone Consistency (语气一致性)
  3. Q3: Turn Count Limit (对话轮次控制)
  4. Q4: Level Appropriateness (难度适配)
  5. Q5: Correction Behavior (纠错行为)
  6. Q6: Brevity & Encouragement (简洁性)
  7. Q7: Language Quality (语言质量)
  8. Q8: Safety Filter (安全过滤)
- **输出**: Daily Quality Index (DQI) 
- **告警**: DQI < 85% 时触发告警

---

## 🏃 如何运行评估

### 方法 1: 使用便捷脚本（推荐）

```bash
# 语法: ./run_evaluation.sh [模块] [样本数] [报告格式]

# 示例 1: 评估 LLM，10 个样本，生成 MD 和 JSON 报告
./run_evaluation.sh llm 10 "md json"

# 示例 2: E2E 评估，5 个样本
./run_evaluation.sh e2e 5

# 示例 3: 完整评估，100 个样本
./run_evaluation.sh all 100 "md json html"
```

### 方法 2: 使用 Python 命令

#### 基础评估

```bash
# 快速测试（5 个样本）
python -m evaluation.run_eval --quick

# LLM 评估（指定样本数）
python -m evaluation.run_eval --module llm --samples 20

# E2E 评估
python -m evaluation.run_eval --module e2e --samples 10

# 全部模块
python -m evaluation.run_eval --module all --samples 50
```

#### 对比测试

```bash
# 对比不同 LLM Provider
python -m evaluation.run_eval --compare --providers deepseek ollama

# 对比不同 TTS 模式
python -m evaluation.run_eval --module e2e --tts-mode fast
python -m evaluation.run_eval --module e2e --tts-mode local
```

#### 质量评估

```bash
# 评估昨天的对话
python -m evaluation.run_eval --module quality

# 评估指定日期
python -m evaluation.run_eval --module quality --eval-date 2026-02-08

# 自定义 Judge 模型
python -m evaluation.run_eval --module quality \
  --judge-provider openai \
  --judge-model gpt-4o \
  --samples 50 \
  --dqi-threshold 85.0
```

#### 生成报告

```bash
# 生成多种格式的报告
python -m evaluation.run_eval --module llm --report json md html

# 指定输出目录
python -m evaluation.run_eval --module llm --output ./my-reports
```

---

## 📁 如何查看报告

### 报告文件位置

```
evaluation/
├── reports/
│   ├── results/                    # JSON 原始数据
│   │   ├── llm_20260209_*.json
│   │   ├── e2e_20260209_*.json
│   │   └── ...
│   │
│   ├── llm_latency_report_*.md     # Markdown 报告
│   ├── llm_latency_report_*.html   # HTML 报告（可在浏览器打开）
│   └── comparisons/                # 对比测试报告
│       └── comparison_*.json
```

### 快速查看最新报告

```bash
# 查看最新的 JSON 结果
cat $(ls -t evaluation/reports/results/*.json | head -1) | jq '.'

# 查看最新的 Markdown 报告
cat $(ls -t evaluation/reports/*.md | head -1)

# 在 VS Code 中打开最新报告
code $(ls -t evaluation/reports/*.md | head -1)
```

### 报告内容示例

#### LLM 评估报告示例

```json
{
  "llm": {
    "evaluator": "LLMEvaluator",
    "provider": "ollama",
    "model": "deepseek-chat",
    "timing": {
      "mean_ms": 918.32,
      "median_ms": 927.58,
      "p95_ms": 928.19,
      "min_ms": 899.19,
      "max_ms": 928.19
    },
    "success_rate": 100.0,
    "extra_metrics": {
      "chars_per_second_mean": 96.75,
      "output_length_mean": 89
    }
  }
}
```

#### 质量评估报告示例

```json
{
  "daily_quality_index": 87.5,
  "alert_triggered": false,
  "metrics": {
    "q1_role_adherence": 90.0,
    "q2_tone_consistency": 88.0,
    "q3_turn_limit": 85.0,
    "q4_level_appropriateness": 92.0,
    "q5_correction": 86.0,
    "q6_brevity": 84.0,
    "q7_language_quality": 89.0,
    "q8_safety": 100.0
  },
  "samples_evaluated": 50,
  "evaluation_date": "2026-02-08"
}
```

---

## 🔧 高级用法

### 1. 编程方式运行评估

```python
from evaluation.runners import EvaluationRunner
from evaluation.reports import ReportGenerator

# 创建评估运行器
runner = EvaluationRunner()

# 运行 LLM 评估
result = runner.run_llm_evaluation(n_samples=50, provider="deepseek")
print(f"Mean Latency: {result.timing.mean * 1000:.0f}ms")

# 生成报告
generator = ReportGenerator()
generator.generate_report(result, formats=["json", "md"], output_dir="./reports")
```

### 2. 自定义基准数据集

```bash
# 基准数据集位置
evaluation/benchmark/data/benchmark_samples.json

# 修改数据集后重新运行
python -m evaluation.run_eval --module llm
```

### 3. 配置 Judge LLM

在 `.env` 文件中配置：

```bash
# 使用 OpenAI GPT-4o 作为 Judge
JUDGE_PROVIDER=openai
JUDGE_MODEL=gpt-4o
OPENAI_API_KEY=your_key_here

# 或使用 Anthropic Claude
JUDGE_PROVIDER=anthropic
JUDGE_MODEL=claude-3-opus-20240229
ANTHROPIC_API_KEY=your_key_here

# 或使用本地 Ollama
JUDGE_PROVIDER=ollama
JUDGE_MODEL=llama3-70b
```

### 4. 持续集成

```bash
# 在 CI/CD 中运行评估
python -m evaluation.run_eval --quick --silent

# 检查是否满足目标
python -m evaluation.run_eval --module llm --samples 100
# 如果 P95 > 1500ms，构建失败
```

---

## ❓ 常见问题

### Q1: 运行评估时提示 "No module named 'utils.logger'"

**解决方案**:
```bash
# 确保已激活正确的环境
source /workspace/init_env.sh && conda activate lm

# 检查 Python 路径
cd /workspace/LangCoach
python -c "import sys; print(sys.path)"
```

### Q2: E2E 评估失败 - Speech API 连接错误

**解决方案**:
```bash
# 先启动 Speech API
./run_speech_api.sh

# 等待服务启动（约 10 秒）
sleep 10

# 再运行 E2E 评估
python -m evaluation.run_eval --module e2e
```

### Q3: 如何只评估特定的 Provider？

```bash
# 评估 DeepSeek
python -m evaluation.run_eval --module llm --provider deepseek

# 评估 Ollama
python -m evaluation.run_eval --module llm --provider ollama
```

### Q4: 报告文件太多，如何清理？

```bash
# 清理 30 天前的报告
find evaluation/reports/results -name "*.json" -mtime +30 -delete
find evaluation/reports -name "*.md" -mtime +30 -delete
```

### Q5: 如何查看实时评估进度？

```bash
# 使用 --verbose 参数
python -m evaluation.run_eval --module llm --verbose

# 查看日志
tail -f evaluation/logs/evaluation.log
```

### Q6: Quality 评估需要哪些对话日志？

Quality 评估需要从数据库或日志中提取对话历史。确保：

1. 后端 API 在运行（`./run_miniprogram_api.sh`）
2. 数据库 `langcoach.db` 存在且有对话记录
3. 或者有保存的对话日志文件

---

## 📝 参考文档

- [评估框架设计文档](./EVALUATION_PLAN.md)
- [产品功能文档](./PRODUCT_PLAN.md)
- [评估框架 README](./evaluation/README.md)

---

## 🎯 目标指标速查

| 指标 | 目标值 | 评估模块 |
|------|--------|----------|
| E2E Audio Latency | < 3000ms | `e2e` |
| LLM TTFT | < 1500ms | `llm` |
| Success Rate | > 99.5% | `all` |
| Daily Quality Index | ≥ 85% | `quality` |

---

## 🔄 日常评估流程

### 每日例行评估（推荐）

```bash
# 1. 早上运行质量评估
python -m evaluation.run_eval --module quality

# 2. 检查 DQI 是否 >= 85%
# 如果低于 85%，会自动生成告警报告

# 3. 定期运行性能评估（每周）
python -m evaluation.run_eval --module llm --samples 100 --report html

# 4. 对比测试（有新模型时）
python -m evaluation.run_eval --compare --providers new-model old-model
```

---

**最后更新**: 2026-02-09  
**维护者**: LangCoach Team
