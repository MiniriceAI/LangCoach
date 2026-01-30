import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件（必须在其他导入之前！）
# 在项目根目录查找 .env 文件
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    # 尝试从当前工作目录加载
    load_dotenv()

# 现在可以安全地导入其他模块
import gradio as gr
import socket
from tabs.scenario_tab import create_scenario_tab
from tabs.vocab_tab import create_vocab_tab
from utils.logger import LOG

LOG.info(f"✅ 已加载配置文件: {env_path if env_path.exists() else '当前目录'}")

def is_port_in_use(port):
    """检查端口是否被占用"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    """尝试停止占用端口的进程"""
    import subprocess
    current_pid = os.getpid()
    parent_pid = os.getppid()
    
    # 获取所有祖先进程 PID（避免杀死自己的进程树）
    ancestor_pids = {current_pid, parent_pid}
    try:
        # 获取更多祖先进程
        ppid = parent_pid
        for _ in range(5):  # 最多追溯5层
            result = subprocess.run(['ps', '-o', 'ppid=', '-p', str(ppid)], 
                                   capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                ppid = int(result.stdout.strip())
                ancestor_pids.add(ppid)
            else:
                break
    except:
        pass
    
    try:
        # macOS/Linux 使用 lsof
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            killed = False
            for pid in pids:
                try:
                    pid_int = int(pid)
                    # 不要杀死当前进程树中的任何进程
                    if pid_int not in ancestor_pids:
                        subprocess.run(['kill', '-9', pid], check=True)
                        print(f"✅ 已停止进程 {pid} (端口 {port})")
                        killed = True
                except (ValueError, subprocess.CalledProcessError):
                    pass
            return killed
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    return False

# 热重载支持：在模块加载时自动清理端口
def _auto_cleanup_for_reload():
    """热重载时自动清理端口占用"""
    port = int(os.getenv('GRADIO_PORT', '8300'))
    # 检查是否在热重载模式下运行（gradio 命令会设置这个环境变量）
    is_reload_mode = os.getenv('GRADIO_WATCH_DIRS') is not None
    if is_reload_mode and is_port_in_use(port):
        print(f"🔄 热重载: 清理端口 {port}...")
        if kill_process_on_port(port):
            import time
            time.sleep(0.5)
        else:
            # 如果无法杀死进程，Gradio 会自动处理
            print(f"ℹ️ 端口 {port} 由 Gradio 管理，跳过清理")

_auto_cleanup_for_reload()

# 创建 Gradio 应用（模块级变量，支持热重载）
with gr.Blocks(title="LangCoach 英语私教") as demo:
    create_scenario_tab()
    create_vocab_tab()

if __name__ == "__main__":
    # 从环境变量获取端口
    port = int(os.getenv('GRADIO_PORT', '8300'))
    force_restart = '--force' in sys.argv or os.getenv('GRADIO_FORCE_RESTART', '').lower() == 'true'
    is_reload_mode = os.getenv('GRADIO_WATCH_DIRS') is not None

    # 检查命令行参数（排除 --force）
    args = [arg for arg in sys.argv[1:] if arg != '--force']
    if args:
        try:
            port = int(args[0])
        except ValueError:
            print(f"⚠️ 无效的端口号: {args[0]}，使用默认端口 8300")
            port = 8300

    # 热重载模式下跳过端口检查（Gradio 自己管理）
    if is_reload_mode:
        print(f"🔄 热重载模式：Gradio 管理端口 {port}")
    # 检查端口是否被占用
    elif is_port_in_use(port):
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
