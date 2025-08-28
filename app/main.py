import base64
import json
import logging
from typing import Optional

import orjson
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from websockets.client import connect as ws_connect

from .config import settings

logger = logging.getLogger("stefan-api-test-3")
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

app = FastAPI(title="stefan-api-test-3", version="0.1.1")

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
        raise HTTPException(
            status_code=400,
            detail=f"Texten är för lång (>{settings.MAX_TEXT_CHARS}).",
        )
    return {"received_chars": length}


@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket):
    await ws.accept()
    try:
        # Signalera att backenden är redo
        await ws.send_text(json.dumps({"type": "status", "stage": "ready"}))

        # Läs första klientmeddelandet
        raw = await ws.receive_text()
        try:
            data = json.loads(raw)
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
            await ws.send_text(
                json.dumps(
                    {"type": "error", "message": f"Max {settings.MAX_TEXT_CHARS} tecken"}
                )
            )
            await ws.close(code=1009)
            return

        await ws.send_text(
            orjson.dumps(
                {"type": "status", "stage": "connecting-elevenlabs", "voice_id": voice_id}
            ).decode()
        )

        # WS-URL till ElevenLabs (behåll MP3 så frontend kan spela upp blobben)
        query = f"?model_id={model_id}&output_format=mp3_44100_64"
        eleven_ws_url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input{query}"
        )
        headers = [("xi-api-key", settings.ELEVENLABS_API_KEY)]

        async with ws_connect(eleven_ws_url, extra_headers=headers, open_timeout=30) as eleven:
            # Initiera sessionen med röst- och generation-settings
            init_msg = {
                "text": " ",  # kickstart
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.8,
                    "use_speaker_boost": False,
                    "speed": 1.0,
                },
                # Lägre chunk-trösklar → snabbare första audio
                "generation_config": {
                    "chunk_length_schedule": [50, 90, 140]
                },
                "xi_api_key": settings.ELEVENLABS_API_KEY,
            }
            await eleven.send(orjson.dumps(init_msg).decode())

            # Skicka användarens text
            await eleven.send(orjson.dumps({"text": text}).decode())

            # Tvinga flush i slutet så korta texter alltid genereras
            await eleven.send(orjson.dumps({"text": "", "flush": True}).decode())

            await ws.send_text(orjson.dumps({"type": "status", "stage": "streaming"}).decode())

            # Vidarebefordra inkommande chunkar
            async for server_msg in eleven:
                # ElevenLabs skickar normalt JSON-frames
                try:
                    payload = json.loads(server_msg)
                except Exception:
                    # Om det mot förmodan kommer binärt → skicka vidare
                    if isinstance(server_msg, (bytes, bytearray)):
                        await ws.send_bytes(server_msg)
                    continue

                # Fel från ElevenLabs
                if payload.get("event") == "error" or "error" in payload:
                    err_msg = (
                        payload.get("message")
                        or payload.get("error")
                        or "Okänt fel från TTS-leverantören"
                    )
                    await ws.send_text(orjson.dumps({"type": "error", "message": err_msg}).decode())
                    break

                # Audio kan vara null/tom → hoppa över
                audio_b64 = payload.get("audio")
                if isinstance(audio_b64, str) and audio_b64:
                    try:
                        b = base64.b64decode(audio_b64)
                        if b:
                            await ws.send_bytes(b)
                    except Exception as e:
                        logger.warning("Kunde inte dekoda audio-chunk: %s", e)

                # Slutsignal
                if payload.get("isFinal") is True or payload.get("event") == "finalOutput":
                    break

            await ws.send_text(orjson.dumps({"type": "status", "stage": "done"}).decode())

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.exception("WS error: %s", e)
        # Försök skicka fel till klienten och stäng ordnat
        try:
            await ws.send_text(orjson.dumps({"type": "error", "message": str(e)}).decode())
        except Exception:
            pass
        try:
            await ws.close(code=1011)
        except Exception:
            pass
