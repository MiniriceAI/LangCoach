import gradio as gr
import os
import sys
import socket
from pathlib import Path
from dotenv import load_dotenv
from tabs.scenario_tab import create_scenario_tab
from tabs.vocab_tab import create_vocab_tab
from tabs.speech_tab import create_speech_tab
from utils.logger import LOG

# 加载 .env 文件（如果存在）
# 在项目根目录查找 .env 文件
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
    LOG.info(f"✅ 已加载配置文件: {env_path}")
else:
    # 尝试从当前工作目录加载
    load_dotenv()
    LOG.debug("🔍 尝试从当前目录加载 .env 文件")

def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    """尝试停止占用端口的进程"""
    import subprocess
    try:
        # macOS/Linux 使用 lsof
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    subprocess.run(['kill', '-9', pid], check=True)
                    print(f"✅ 已停止进程 {pid} (端口 {port})")
                except subprocess.CalledProcessError:
                    pass
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return False

def is_gradio_reload_mode():
    """检查是否在 Gradio 热重载模式下运行"""
    # Gradio 热重载会设置这些环境变量
    return (
        os.getenv('GRADIO_WATCH_DIRS') is not None or
        os.getenv('GRADIO_WATCH_FILE') is not None or
        'gradio' in sys.argv[0].lower()
    )

# 创建 Gradio 应用（模块级变量，支持热重载）
with gr.Blocks(title="LangCoach 英语私教") as demo:
    create_scenario_tab()
    create_vocab_tab()
    create_speech_tab()

if __name__ == "__main__":
    # 从环境变量获取端口
    port = int(os.getenv('GRADIO_PORT', '8300'))
    force_restart = '--force' in sys.argv or os.getenv('GRADIO_FORCE_RESTART', '').lower() == 'true'

    # 检查命令行参数（排除 --force）
    args = [arg for arg in sys.argv[1:] if arg != '--force']
    if args:
        try:
            port = int(args[0])
        except ValueError:
            print(f"⚠️ 无效的端口号: {args[0]}，使用默认端口 8300")
            port = 8300

    # 在 Gradio 热重载模式下跳过端口检查（Gradio 会自己处理）
    if not is_gradio_reload_mode():
        # 检查端口是否被占用
        if is_port_in_use(port):
            print(f"⚠️ 端口 {port} 已被占用")
            if force_restart:
                print(f"🔄 尝试停止占用端口的进程...")
                if kill_process_on_port(port):
                    import time
                    time.sleep(1)  # 等待端口释放
                else:
                    print(f"❌ 无法自动停止占用端口的进程")
                    print(f"   请手动停止: lsof -ti :{port} | xargs kill -9")
                    print(f"   或使用其他端口: python src/main.py {port + 1} --force")
                    sys.exit(1)
            else:
                print(f"💡 提示:")
                print(f"   - 使用 --force 参数自动停止旧进程: python src/main.py --force")
                print(f"   - 或指定端口并强制重启: python src/main.py {port} --force")
                print(f"   - 或使用其他端口: python src/main.py {port + 1}")
                print(f"   - 或设置环境变量: GRADIO_FORCE_RESTART=true python src/main.py")
                sys.exit(1)

    # 启动应用
    print(f"🚀 启动 LangCoach 在端口 {port}...")
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=port
    )
