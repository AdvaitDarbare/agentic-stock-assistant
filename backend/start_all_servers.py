#!/usr/bin/env python3
"""
Start all MCP servers for the multi-agent system
"""

import subprocess
import time
import sys
import os

def start_server(script_path, port, name):
    """Start a single MCP server"""
    print(f"🚀 Starting {name} server on port {port}...")
    
    # Start server in background
    process = subprocess.Popen(
        [sys.executable, "-m", script_path.replace("/", ".").replace(".py", "")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.getcwd()
    )
    
    # Give it a moment to start
    time.sleep(1)
    
    if process.poll() is None:
        print(f"✅ {name} server started successfully (PID: {process.pid})")
        return process
    else:
        _, stderr = process.communicate()
        print(f"❌ Failed to start {name} server")
        print(f"Error: {stderr.decode()}")
        return None

def main():
    """Start all MCP servers"""
    print("🌟 Starting Multi-Agent System MCP Servers...")
    
    servers = [
        ("agents/sql_agent.py", 8010, "SQL Agent"),
        ("agents/news_agent.py", 8020, "News Agent"),
        ("agents/fallback_agent.py", 8030, "Fallback Agent"),
        ("agents/sentiment_agent.py", 8040, "Sentiment Agent"),
    ]
    
    processes = []
    
    for script, port, name in servers:
        if os.path.exists(script):
            proc = start_server(script, port, name)
            if proc:
                processes.append((proc, name))
        else:
            print(f"⚠️  {script} not found, skipping {name}")
    
    if processes:
        print(f"\n🎯 Started {len(processes)} MCP servers successfully!")
        print("\n📍 Server URLs:")
        for i, (script, port, name) in enumerate(servers):
            if i < len(processes):
                print(f"  • {name}: http://localhost:{port}/mcp")
        
        print("\n🔧 To test the system:")
        print("  poetry run python test_sentiment.py")
        print("  poetry run python test_persistence.py")
        print("  langgraph dev")
        
        print("\n⚠️  Press Ctrl+C to stop all servers")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping all servers...")
            for proc, name in processes:
                proc.terminate()
                print(f"  • Stopped {name}")
    else:
        print("❌ No servers started successfully")

if __name__ == "__main__":
    main()