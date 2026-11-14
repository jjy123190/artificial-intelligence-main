"""启动本地网页，不改变命令行算法的依赖或行为。"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import threading
import time
import urllib.request
import webbrowser


def serve(port: int = 8765, open_browser: bool = False):
    if not 1 <= port <= 65535:
        raise ValueError("端口必须在1–65535之间")
    if os.environ.get("EQUITY_WEB_FILE_LOG") == "1":
        runtime = Path(__file__).resolve().parents[1] / "outputs" / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        sys.stdout = (runtime / "server.log").open("a", encoding="utf-8", buffering=1)
        sys.stderr = (runtime / "server-error.log").open("a", encoding="utf-8", buffering=1)
    import uvicorn
    from .api import create_app
    url = f"http://127.0.0.1:{port}"
    print(f"股权推理工作台：{url}")
    print("仅监听本机。关闭此进程即可停止服务；网页调整不会改写原始CSV。")
    if open_browser:
        def open_when_ready():
            for _ in range(30):
                try:
                    with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
                        if response.status == 200:
                            webbrowser.open(url)
                            return
                except OSError:
                    time.sleep(.3)
        threading.Thread(target=open_when_ready, daemon=True).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=port, log_level="info", timeout_keep_alive=5)
