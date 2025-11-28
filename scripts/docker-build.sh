#!/bin/bash
# Docker 镜像构建脚本

set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}LangCoach Docker 镜像构建${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查 Docker 是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}错误: Docker 未安装，请先安装 Docker${NC}"
    exit 1
fi

# 镜像名称和标签
IMAGE_NAME="langcoach"
TAG="${1:-latest}"

echo -e "${GREEN}构建镜像: ${IMAGE_NAME}:${TAG}${NC}"

# 构建镜像
docker build -t ${IMAGE_NAME}:${TAG} .

echo -e "${GREEN}✓ 镜像构建完成！${NC}"
echo -e "${GREEN}镜像名称: ${IMAGE_NAME}:${TAG}${NC}"

# 显示镜像信息
echo -e "${GREEN}镜像大小:${NC}"
docker images ${IMAGE_NAME}:${TAG} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}使用方法:${NC}"
echo -e "  docker run -d -p 7860:7860 -e DEEPSEEK_API_KEY=your_key ${IMAGE_NAME}:${TAG}"
echo -e "  或使用 docker-compose: docker-compose up -d"
echo -e "${GREEN}========================================${NC}"

