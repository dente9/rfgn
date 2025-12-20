#!/bin/bash

# --- 配置区 ---
PORT=8080
PYTHON_BIN=$(which python)
JUPYTER_BIN=$(which jupyter)

echo ">>> [1/4] 正在安装 JupyterLab 及中文语言包..."
# 同时安装主程序、代理插件和中文包
$PYTHON_BIN -m pip install --quiet jupyterlab jupyter-server-proxy jupyterlab-language-pack-zh-CN -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ">>> [2/4] 正在清理残留进程 (8080端口)..."
# 暴力清理端口占用，防止启动失败
fuser -k $PORT/tcp > /dev/null 2>&1
pkill -f jupyter-lab
sleep 1

echo ">>> [3/4] 正在启动 Jupyter Lab..."
nohup $PYTHON_BIN -m jupyterlab \
    --ip=0.0.0.0 \
    --port=$PORT \
    --allow-root \
    --no-browser \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.allow_remote_access=True > ~/jupyter_debug.log 2>&1 &

echo "------------------------------------------------"
echo "✅ 环境修复完成！"
echo "🔗 访问地址: 服务器:端口号/lab"
echo "🌐 中文设置提示: 刷新页面 -> Settings -> Language -> Chinese"
echo "🛠️  代理示例: 启动端口为 N 的服务后，访问 服务器:端口号/proxy/N/"
echo "------------------------------------------------"
