# stefan-api-test-3 (FastAPI + ElevenLabs WS TTS)

Realtime TTS-backend som proxyar mot ElevenLabs WebSocket-API och streamar audio till frontend via WebSocket.

## Snabbstart (lokalt)

1) Skapa `.env` utifrån `.env.example` och fyll i:
   - `ELEVENLABS_API_KEY`
   - `DEFAULT_VOICE_ID` (ditt röst-ID från ElevenLabs)
2) Installera & kör
   ```sh
   make install
   make dev
   # servern på http://localhost:8080
   ```

## WebSocket-protokoll

**Kund → Backend** (`/ws/tts`): första meddelandet måste vara JSON:
```json
{"text": "Hej världen", "voice_id": null, "model_id": null}
```
- `text`: krävs, max `MAX_TEXT_CHARS` (default 1000)
- `voice_id`: valfritt (default från backend)
- `model_id`: valfritt (default `eleven_turbo_v2_5`)

**Backend → Kund**: text-meddelanden med status
```json
{"type":"status","stage":"ready|connecting-elevenlabs|streaming|done"}
```
samt **binära WS-frames** med `audio/mpeg`-bytes. Lägg till chunkarna i en `MediaSource` eller via WebAudio.

## Deployment på Render

- Använd `render.yaml` (Docker). Tjänstnamn: **stefan-api-test-3**.
- Sätt miljövariabler enligt `.env.example` (minst API-nyckel & voice-id).
- Render använder `PORT`; Dockerfile lyssnar på `8080` och respekterar `PORT`.

## CORS

CORS är konfigurerat med `ALLOWED_ORIGIN_REGEX`, default tillåts `*.lovable.app` och `localhost`. Justera vid behov.

## Frågor & felsökning

- 1000 tecken max (valbart via env)
- Ingen persistens; allt är streamat
- Loggning (stdout) för felsökning

## Licens

MIT
