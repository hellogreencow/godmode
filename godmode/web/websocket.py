"""
WebSocket manager for real-time communication in GodMode web application.
"""

import json
import logging
from typing import Any, Dict, List

from fastapi import WebSocket


logger = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections for real-time communication."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket client {client_id} connected")
        
        # Send welcome message
        await self.send_personal_message(
            {"type": "connected", "client_id": client_id},
            client_id
        )
    
    def disconnect(self, client_id: str):
        """Disconnect a WebSocket client."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients."""
        disconnected_clients = []
        
        for client_id, connection in self.active_connections.items():
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Remove disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)
    
    def get_active_connections(self) -> List[str]:
        """Get list of active client IDs."""
        return list(self.active_connections.keys())
    
    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)
