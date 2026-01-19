#!/usr/bin/env python3
"""
Simple Speech API starter script.
Run with: python start_speech_api.py
"""
import os
import sys
import signal
import socket

def check_port(port):
    """Check if port is available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('0.0.0.0', port))
        sock.close()
        return True
    except OSError:
        return False

def kill_port(port):
    """Kill process on port."""
    import subprocess
    try:
        result = subprocess.run(
            ['fuser', '-k', f'{port}/tcp'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except:
        return False

def main():
    port = int(os.getenv('SPEECH_API_PORT', '8301'))
    host = os.getenv('SPEECH_API_HOST', '0.0.0.0')
    
    print(f"Starting Speech API on {host}:{port}")
    
    # Check and free port
    if not check_port(port):
        print(f"Port {port} is busy, attempting to free it...")
        kill_port(port)
        import time
        time.sleep(2)
        
        if not check_port(port):
            print(f"ERROR: Cannot free port {port}")
            sys.exit(1)
    
    print(f"Port {port} is available")
    
    # Import and run uvicorn
    import uvicorn
    
    uvicorn.run(
        "src.api.speech_api:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
