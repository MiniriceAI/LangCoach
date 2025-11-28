# Docker 部署指南

## 概述

本项目提供了完整的 Docker 容器化部署方案，支持生产环境和开发环境两种模式。

## 快速开始

### 方式一：使用 Docker Compose（推荐）

1. **设置环境变量**
   ```bash
   export DEEPSEEK_API_KEY=your_api_key
   # 或创建 .env 文件
   echo "DEEPSEEK_API_KEY=your_api_key" > .env
   ```

2. **启动服务**
   ```bash
   docker-compose up -d
   ```

3. **查看日志**
   ```bash
   docker-compose logs -f
   ```

4. **停止服务**
   ```bash
   docker-compose down
   ```

### 方式二：使用 Docker 命令

1. **构建镜像**
   ```bash
   docker build -t langcoach:latest .
   ```

2. **运行容器**
   ```bash
   docker run -d \
     --name langcoach \
     -p 7860:7860 \
     -e DEEPSEEK_API_KEY=your_api_key \
     -v $(pwd)/logs:/app/logs \
     --restart unless-stopped \
     langcoach:latest
   ```

### 方式三：使用自动化脚本

1. **构建镜像**
   ```bash
   ./scripts/docker-build.sh
   ```

2. **运行容器**
   ```bash
   export DEEPSEEK_API_KEY=your_api_key
   ./scripts/docker-run.sh
   ```

## 开发环境

开发环境支持代码热重载，修改代码后自动生效：

```bash
docker-compose -f docker-compose.dev.yml up
```

开发环境会挂载源代码目录，支持实时修改。

## 镜像说明

### 多阶段构建

Dockerfile 使用多阶段构建优化镜像大小：
- **Stage 1 (builder):** 安装编译依赖和 Python 包
- **Stage 2 (runtime):** 仅复制运行时需要的文件

### 镜像特性

- 基于 `python:3.10-slim`，镜像体积小
- 包含健康检查
- 支持环境变量配置
- 日志目录挂载
- 自动重启策略

## 环境变量

### LLM 提供者配置（按优先级）

| 变量名 | 说明 | 必需 | 优先级 |
|--------|------|------|--------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | 否* | 1（最高） |
| `OPENAI_API_KEY` | OpenAI API 密钥 | 否* | 2 |
| `OPENAI_MODEL` | OpenAI 模型名称 | 否 | 2（默认: gpt-4o-mini） |
| `OLLAMA_BASE_URL` | Ollama 服务地址 | 否 | 3（默认: http://localhost:11434） |
| `OLLAMA_MODEL` | Ollama 模型名称 | 否 | 3（默认: llama3.1:8b-instruct-q8_0） |

*至少需要配置一个 LLM 提供者的 API 密钥，或者确保 Ollama 在本地运行。

### 其他环境变量

| 变量名 | 说明 | 必需 |
|--------|------|------|
| `PYTHONUNBUFFERED` | Python 输出缓冲设置 | 否（默认已设置） |

### 配置示例

**使用 DeepSeek（推荐）：**
```bash
export DEEPSEEK_API_KEY=your_deepseek_api_key
```

**使用 OpenAI：**
```bash
export OPENAI_API_KEY=your_openai_api_key
export OPENAI_MODEL=gpt-4o-mini  # 可选
```

**使用 Ollama（本地部署）：**
```bash
# 确保 Ollama 服务正在运行
export OLLAMA_BASE_URL=http://localhost:11434  # 可选
export OLLAMA_MODEL=llama3.1:8b-instruct-q8_0  # 可选
```

## 端口说明

- **7860:** Gradio 应用默认端口
- 可通过 `-p` 参数映射到其他端口

## 数据持久化

- **日志目录:** `./logs` 挂载到容器内的 `/app/logs`
- 日志文件会持久化到宿主机

## 健康检查

容器包含健康检查，每 30 秒检查一次：
```bash
# 查看健康状态
docker inspect --format='{{.State.Health.Status}}' langcoach
```

## 常用命令

```bash
# 查看运行中的容器
docker ps

# 查看容器日志
docker logs -f langcoach

# 进入容器
docker exec -it langcoach bash

# 停止容器
docker stop langcoach

# 重启容器
docker restart langcoach

# 删除容器
docker rm langcoach

# 查看镜像
docker images langcoach

# 删除镜像
docker rmi langcoach:latest
```

## 故障排查

### 容器无法启动

1. 检查环境变量是否设置：
   ```bash
   docker inspect langcoach | grep -A 10 Env
   ```

2. 查看容器日志：
   ```bash
   docker logs langcoach
   ```

### 端口被占用

如果 7860 端口被占用，可以映射到其他端口：
```bash
docker run -d -p 8080:7860 ...
```

### 权限问题

确保日志目录有写权限：
```bash
chmod 755 logs
```

## 生产环境建议

1. **使用环境变量文件**
   ```bash
   docker-compose --env-file .env.production up -d
   ```

2. **配置资源限制**
   在 `docker-compose.yml` 中添加：
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '2'
         memory: 2G
   ```

3. **使用反向代理**
   建议使用 Nginx 或 Traefik 作为反向代理

4. **监控和日志**
   集成日志收集系统（如 ELK、Loki）

## 更新镜像

```bash
# 重新构建
docker build -t langcoach:latest .

# 停止旧容器
docker stop langcoach
docker rm langcoach

# 启动新容器
docker-compose up -d
```

