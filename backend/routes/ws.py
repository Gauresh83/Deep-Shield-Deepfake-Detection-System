"""
WebSocket endpoints for real-time scan streaming and live alerts — Part 7.
"""
import asyncio
import json
import time
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from ..core.security import decode_token

router = APIRouter(tags=["WebSocket"])


class ConnectionManager:
    """Manages active WebSocket connections per user."""
    def __init__(self):
        self.active: Dict[int, Set[WebSocket]] = {}

    async def connect(self, ws: WebSocket, user_id: int):
        await ws.accept()
        self.active.setdefault(user_id, set()).add(ws)

    def disconnect(self, ws: WebSocket, user_id: int):
        self.active.get(user_id, set()).discard(ws)

    async def send_to_user(self, user_id: int, data: dict):
        dead = set()
        for ws in list(self.active.get(user_id, set())):
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active.get(user_id, set()).discard(ws)

    async def broadcast(self, data: dict):
        for uid in list(self.active.keys()):
            await self.send_to_user(uid, data)


manager = ConnectionManager()


@router.websocket("/ws/scan/{token}")
async def scan_websocket(ws: WebSocket, token: str):
    """
    WebSocket endpoint for real-time scan progress updates.
    Connect with: ws://localhost:8000/ws/scan/<access_token>

    Messages received from server:
      {"type": "connected", "user_id": N}
      {"type": "scan_start", "scan_id": N, "message": "..."}
      {"type": "module_progress", "module": "face|voice|nlp|fusion", "status": "running|done", "score": N}
      {"type": "scan_complete", "scan_id": N, "verdict": "...", "fusion_score": N, "severity": "..."}
      {"type": "alert", "severity": "...", "message": "..."}
      {"type": "error", "message": "..."}
      {"type": "ping"}
    """
    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub", 0))
    except Exception:
        await ws.close(code=4001)
        return

    await manager.connect(ws, user_id)

    try:
        await ws.send_json({"type": "connected", "user_id": user_id,
                            "message": "M3-ID real-time stream active"})
        while True:
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_json({"type": "pong", "ts": time.time()})
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws, user_id)


@router.websocket("/ws/alerts/{token}")
async def alerts_websocket(ws: WebSocket, token: str):
    """
    Dedicated alerts stream — receives CRITICAL/HIGH alerts in real time.
    """
    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub", 0))
    except Exception:
        await ws.close(code=4001)
        return

    await manager.connect(ws, user_id)
    try:
        await ws.send_json({"type": "alerts_connected", "user_id": user_id})
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=60.0)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(ws, user_id)


# Expose manager so other modules can push messages
ws_manager = manager
