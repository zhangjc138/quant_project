#!/bin/bash
# quant_project 屏幕录制脚本
# 用于录制Web界面演示视频

# 配置
OUTPUT_DIR="/home/zjc/.openclaw/workspace/quant_project"
STREAMLIT_PORT=8501
RECORDER_WINDOW="Streamlit"

# 启动Streamlit
echo "启动Streamlit服务..."
cd $OUTPUT_DIR
python3 -m streamlit run app.py &
STREAMLIT_PID=$!
sleep 5

# 等待用户准备
echo "请切换到Streamlit浏览器窗口，准备开始录制"
echo "按Enter开始录制30秒版本..."
read

# 录制30秒版本 (需要x11grab)
if command -v ffmpeg &> /dev/null; then
    echo "开始录制30秒演示..."
    ffmpeg -f x11grab -i :0.0 -t 00:00:30 -c:v libx264 -preset fast \
        "$OUTPUT_DIR/video_demo_30s.mp4" \
        -y 2>/dev/null || echo "请使用专业录屏软件"
else
    echo "ffmpeg未安装，请使用OBS或其他录屏软件"
fi

echo "录制完成！"

# 录制1分钟版本
echo "按Enter开始录制1分钟版本..."
read
ffmpeg -f x11grab -i :0.0 -t 00:01:00 -c:v libx264 -preset fast \
    "$OUTPUT_DIR/video_demo_1min.mp4" \
    -y 2>/dev/null || echo "请使用专业录屏软件"

# 停止Streamlit
kill $STREAMLIT_PID 2>/dev/null

echo "所有录制完成！"
