#!/bin/bash
# Docker 容器运行脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}LangCoach Docker 容器启动${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${RED}错误: Docker 未安装${NC}"
    exit 1
fi

# 检查环境变量
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo -e "${YELLOW}警告: DEEPSEEK_API_KEY 环境变量未设置${NC}"
    echo -e "${YELLOW}请设置环境变量: export DEEPSEEK_API_KEY=your_api_key${NC}"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 镜像名称
IMAGE_NAME="langcoach:latest"

# 检查镜像是否存在
if ! docker images | grep -q "langcoach.*latest"; then
    echo -e "${YELLOW}镜像不存在，开始构建...${NC}"
    docker build -t ${IMAGE_NAME} .
fi

# 停止并删除旧容器（如果存在）
if docker ps -a | grep -q "langcoach"; then
    echo -e "${YELLOW}停止旧容器...${NC}"
    docker stop langcoach 2>/dev/null || true
    docker rm langcoach 2>/dev/null || true
fi

# 运行容器
echo -e "${GREEN}启动容器...${NC}"
docker run -d \
    --name langcoach \
    -p 7860:7860 \
    -e DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY} \
    -v $(pwd)/logs:/app/logs \
    --restart unless-stopped \
    ${IMAGE_NAME}

echo -e "${GREEN}✓ 容器启动成功！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}访问地址: http://localhost:7860${NC}"
echo -e "${GREEN}查看日志: docker logs -f langcoach${NC}"
echo -e "${GREEN}停止容器: docker stop langcoach${NC}"
echo -e "${GREEN}========================================${NC}"

