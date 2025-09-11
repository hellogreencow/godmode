"""
Web interface for the GodMode hierarchical reasoning system.

This module provides a modern web interface for interacting with the reasoning engine,
including real-time visualization of hierarchical reasoning processes.
"""

from godmode.web.app import create_app, GodModeWebApp
from godmode.web.api import create_api_router
from godmode.web.websocket import WebSocketManager

__all__ = [
    "create_app",
    "GodModeWebApp",
    "create_api_router",
    "WebSocketManager",
]
