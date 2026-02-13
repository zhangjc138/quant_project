#!/bin/bash
# quant_project Web界面启动脚本
# 一键启动Streamlit Web界面

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目路径
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  quant_project Web界面启动脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ 错误: 未找到 python3${NC}"
    echo "请先安装 Python 3.8+"
    exit 1
fi

echo -e "${GREEN}✓${NC} Python环境检查通过"

# 检查依赖
echo ""
echo -e "${YELLOW}📦 检查依赖...${NC}"

# 检查streamlit
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  未安装 streamlit，正在安装...${NC}"
    pip install streamlit -q
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ streamlit 安装失败${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓${NC} streamlit 已安装"

# 检查plotly
if ! python3 -c "import plotly" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  未安装 plotly，正在安装...${NC}"
    pip install plotly -q
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ plotly 安装失败${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}✓${NC} plotly 已安装"

echo ""
echo -e "${GREEN}✓${NC} 所有依赖检查完成"
echo ""

# 显示启动信息
echo -e "${BLUE}🚀 启动 Web 界面...${NC}"
echo ""
echo -e "  📍 地址: ${GREEN}http://localhost:8501${NC}"
echo -e "  📁 项目目录: ${GREEN}${PROJECT_DIR}${NC}"
echo ""
echo -e "${YELLOW}💡 提示:${NC}"
echo -e "  - 按 Ctrl+C 停止服务"
echo -e "  - 浏览器打开 http://localhost:8501"
echo ""

# 启动streamlit
exec python3 -m streamlit run app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --browser.gatherUsageStats false \
    --logger.level info
