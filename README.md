# TBS — Offline AI

A voice-first desktop assistant with eyes that runs **100% offline** — no internet, no API key, no installation. It listens for a wake word ("tbs"), looks through your camera, thinks with a local vision language model (llama.cpp), and answers out loud with a local neural voice (Piper). An optional Claude API mode is available when you have a key and a connection, but nothing requires one.

## Using it from a flash drive

1. Copy this whole **Offline AI** folder onto the drive (or any PC). Drive letter doesn't matter — everything resolves relative to the folder.
2. Double-click **START TBS.bat**.
3. First question in local mode loads the model into RAM — **the first load can take up to a minute** (the status bar shows "Loading local brain…"). After that, answers come in seconds.
4. Say **"tbs"** and speak, press **Push to talk**, or just type in the box. **Watch mode** narrates what the camera sees on an interval and speaks up when something notable happens.

Settings (the gear) persist to `config\settings.json` **inside this folder** — nothing is written to the host PC's registry or AppData. Zero traces, zero installation.

## Model tiers

| Tier | Model | Files (`models\llm\`) | Needs |
|---|---|---|---|
| Light | LFM2-VL 1.6B | `LFM2-VL-1.6B-Q4_0.gguf` + `mmproj-LFM2-VL-1.6B-Q8_0.gguf` | < 7 GB RAM |
| Quality | Gemma 3 4B | `gemma-3-4b-it-Q4_K_M.gguf` + `mmproj-model-f16.gguf` | ≥ 7 GB RAM |

**Auto (default)** picks the Quality tier on machines with 7 GB+ RAM, Light otherwise. To force a tier: gear → **Local model** → pick one. The **Engine** switch chooses Auto / Local only / Claude only (Claude needs an API key — optional).

## Voice

Two Piper voices ship in `models\piper\`: `en_GB-alan-medium` (British, default — lowest latency) and `en_US-ryan-high` (American, higher quality, slightly slower to start speaking). Switch in Settings → **Voice**; "Classic (SAPI)" uses the Windows built-in voice as a fallback.

## Host PC requirements

- Windows 10/11 x64
- **8 GB+ RAM recommended** (Quality tier; the Light tier runs on less)
- Microphone and camera are **optional** — without them you can still type commands; voice input and vision need them
- No admin rights, no install, no internet

## SmartScreen note

The exe is unsigned, so the first launch on a new PC may show a **"Windows protected your PC"** SmartScreen popup. Click **More info → Run anyway**. This happens once per PC.

## Folder map

| Folder | What it is |
|---|---|
| `app\` | The built application (`TBSOffline.exe`) |
| `engine\` | llama.cpp server (local brain runtime) |
| `models\` | `llm\` (vision models) · `piper\` (voices) · `vosk\` (speech-to-text) |
| `piper\` | Piper TTS runtime |
| `config\` | `settings.json` — the only place state lives |
| `screenshots\` | Frames saved when watch mode spots something notable |
| `src\` | Python source (not needed to run) |
| `_dev\` | Build tooling and tests — **safe to delete before copying to the flash drive** (saves ~350 MB) |
