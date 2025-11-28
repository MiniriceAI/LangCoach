# LangCoach Test Suite

## 概述

本项目包含完整的单元测试套件，目标覆盖率达到 80%。

## 运行测试

### 使用 pytest 直接运行

```bash
# 运行所有测试
pytest

# 运行特定模块的测试
pytest tests/agents/

# 运行并显示覆盖率
pytest --cov=src --cov-report=term-missing

# 生成 HTML 覆盖率报告
pytest --cov=src --cov-report=html
```

### 使用自动化脚本

```bash
# 运行测试脚本（会自动检查覆盖率）
./tests/run_tests.sh
```

## 测试结构

```
tests/
├── __init__.py
├── conftest.py          # 共享的 fixtures 和配置
├── agents/              # Agent 模块测试
│   ├── test_agent_base.py
│   ├── test_scenario_agent.py
│   ├── test_vocab_agent.py
│   └── test_session_history.py
├── utils/               # Utils 模块测试
│   └── test_logger.py
└── run_tests.sh         # 自动化测试脚本
```

## 覆盖率要求

- 目标覆盖率：80%
- 当前覆盖率：运行 `pytest --cov=src` 查看

## CI/CD

测试已集成到 GitHub Actions，每次 push 或 PR 都会自动运行测试。

## 编写新测试

1. 在对应的测试目录创建 `test_*.py` 文件
2. 使用 pytest fixtures（定义在 `conftest.py`）
3. 使用 mock 来隔离外部依赖（API、文件系统等）
4. 确保测试独立且可重复运行

## 注意事项

- 测试使用 mock 来避免调用真实的 API
- 使用临时文件进行文件操作测试
- 每个测试后会自动清理 session store

