import asyncio
import base64
import json
import logging
import os
from typing import Optional

import orjson
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import websockets
from websockets.client import connect as ws_connect

from .config import settings

logger = logging.getLogger("stefan-api-test-3")
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

app = FastAPI(title="stefan-api-test-3", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=settings.ALLOWED_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EchoIn(BaseModel):
    text: str = Field(..., max_length=1000)

@app.get("/healthz")
async def healthz():
    return {"ok": True, "service": "stefan-api-test-3"}

@app.post("/echo")
async def echo(payload: EchoIn):
    length = len(payload.text or "")
    logger.info("Echo text received: %s chars", length)
    if length > settings.MAX_TEXT_CHARS:
        raise HTTPException(status_code=400, detail=f"Texten är för lång (>{settings.MAX_TEXT_CHARS}).")
    return {"received_chars": length}

@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket):
    await ws.accept()
    try:
        # Signalera att backenden är redo
        await ws.send_text(json.dumps({"type": "status", "stage": "ready"}))

        msg = await ws.receive_text()
        try:
            data = json.loads(msg)
        except Exception:
            await ws.send_text(json.dumps({"type": "error", "message": "Invalid JSON"}))
            await ws.close(code=1003)
            return

        text: Optional[str] = (data.get("text") or "").strip()
        voice_id: str = data.get("voice_id") or settings.DEFAULT_VOICE_ID
        model_id: str = data.get("model_id") or settings.DEFAULT_MODEL_ID

        if not text:
            await ws.send_text(json.dumps({"type": "error", "message": "Tom text"}))
            await ws.close(code=1003)
            return

        if len(text) > settings.MAX_TEXT_CHARS:
            await ws.send_text(json.dumps({"type": "error", "message": f"Max {settings.MAX_TEXT_CHARS} tecken"}))
            await ws.close(code=1009)  # too big
            return

        await ws.send_text(orjson.dumps({"type": "status", "stage": "connecting-elevenlabs", "voice_id": voice_id}).decode())

        # Bygg ElevenLabs WS-URL med låg latens
        # output_format kan vara t.ex. mp3_44100_64 eller pcm_16000 (se docs)
        query = (
            f"?model_id={model_id}"
            f"&output_format=mp3_44100_64"
            f"&auto_mode=true"
        )
        eleven_ws_url = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input{query}"
        headers = [("xi-api-key", settings.ELEVENLABS_API_KEY)]

        async with ws_connect(eleven_ws_url, extra_headers=headers, open_timeout=30) as eleven:
            # Initiera session (se docs)
            init_msg = {
                "text": " ",  # kickstart
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "speed": 1.0,
                },
                # vissa klienter skickar även xi_api_key här; header räcker, men vi dubblar för kompatibilitet
                "xi_api_key": settings.ELEVENLABS_API_KEY,
            }
            await eleven.send(orjson.dumps(init_msg).decode())

            # Skicka användarens text och trigga generering
            await eleven.send(orjson.dumps({"text": text, "try_trigger_generation": True}).decode())
            # Tom text signalerar slut
            await eleven.send(orjson.dumps({"text": ""}).decode())

            await ws.send_text(orjson.dumps({"type": "status", "stage": "streaming"}).decode())

            # Vidarebefordra inkommande audio-chunkar som binära meddelanden
            async for raw in eleven:
                # ElevenLabs skickar JSON med base64-kodad audio
                try:
                    payload = json.loads(raw)
                except Exception:
                    # Om Eleven någon gång skickar binärt (ska inte hända), vidarebefordra direkt
                    if isinstance(raw, (bytes, bytearray)):
                        await ws.send_bytes(raw)
                    continue

                if "audio" in payload:
                    b = base64.b64decode(payload["audio"])
                    # skicka som binär WS-frame till frontend
                    await ws.send_bytes(b)

                if payload.get("isFinal") is True or payload.get("event") == "finalOutput":
                    break

            await ws.send_text(orjson.dumps({"type": "status", "stage": "done"}).decode())

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("WS error: %s", e)
        try:
            await ws.send_text(orjson.dumps({"type": "error", "message": str(e)}).decode())
        except Exception:
            pass
        try:
            await ws.close(code=1011)
        except Exception:
            pass
