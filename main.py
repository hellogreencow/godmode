#!/usr/bin/env python3
"""
GODMODE - A superhuman, ontological Question Foresight Engine

"See the questions before you ask them."

Usage:
    python main.py [--port PORT] [--host HOST]
"""

import asyncio
import argparse
import uvicorn
from godmode.api import app


def main():
    parser = argparse.ArgumentParser(description="GODMODE - Question Foresight Engine")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print("🚀 Starting GODMODE - Question Foresight Engine")
    print(f"📡 Server will be available at http://{args.host}:{args.port}")
    print("📚 API documentation at http://localhost:8000/docs")
    print("🔍 Health check at http://localhost:8000/health")
    
    uvicorn.run(
        "godmode.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()