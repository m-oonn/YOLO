#!/bin/bash
# YOLO 前端项目浏览器自动化测试脚本
# 需要先安装: npm i -g agent-browser
# 并确保后端服务在 http://localhost:8000 运行

set -e

echo "=== YOLO 前端项目浏览器测试 ==="
echo ""

# 1. 启动前端开发服务器
echo "[1/6] 启动前端开发服务器..."
cd "$(dirname "$0")"
npm run dev &
DEV_PID=$!
sleep 5

# 2. 打开项目首页
echo "[2/6] 打开项目首页..."
agent-browser open http://localhost:8080
agent-browser wait --load networkidle
agent-browser screenshot --full ./test-screenshots/01-homepage.png

# 3. 测试仪表盘页面
echo "[3/6] 测试仪表盘页面..."
agent-browser snapshot -i
# 检查关键元素是否存在
agent-browser wait "text" "数据仪表盘"
agent-browser screenshot --full ./test-screenshots/02-dashboard.png

# 4. 测试导航菜单
echo "[4/6] 测试导航菜单..."
# 点击实时监控
agent-browser click "text" "实时监控"
agent-browser wait --load networkidle
agent-browser screenshot --full ./test-screenshots/03-monitor.png

# 点击事件记录
agent-browser click "text" "事件记录"
agent-browser wait --load networkidle
agent-browser screenshot --full ./test-screenshots/04-events.png

# 点击报警管理
agent-browser click "text" "报警管理"
agent-browser wait --load networkidle
agent-browser screenshot --full ./test-screenshots/05-alarms.png

# 点击系统配置
agent-browser click "text" "系统配置"
agent-browser wait --load networkidle
agent-browser screenshot --full ./test-screenshots/06-config.png

# 5. 测试按钮交互
echo "[5/6] 测试按钮交互..."
# 回到事件记录页面测试清空按钮
agent-browser click "text" "事件记录"
agent-browser wait --load networkidle

# 测试 ActionButton 的确认对话框
agent-browser click "text" "清空全部"
agent-browser wait --text "确认清空"
agent-browser screenshot ./test-screenshots/07-confirm-dialog.png

# 取消对话框
agent-browser dialog dismiss

# 6. 测试响应式布局
echo "[6/6] 测试响应式布局..."
agent-browser set viewport 375 812
agent-browser open http://localhost:8080
agent-browser wait --load networkidle
agent-browser screenshot --full ./test-screenshots/08-mobile.png

# 清理
agent-browser close
kill $DEV_PID 2>/dev/null || true

echo ""
echo "=== 测试完成 ==="
echo "截图保存在 ./test-screenshots/ 目录"
