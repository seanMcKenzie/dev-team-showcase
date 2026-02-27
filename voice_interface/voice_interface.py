#!/usr/bin/env python3
"""
K2S0 Voice Interface v4 — Full Agent Mode
Mic → Whisper STT → Discord (as Sean) → Real K2S0 replies → TTS → afplay

Routes through Discord so the REAL K2S0 (with full tools and memory) responds.
Messages tagged [voice] so K2S0 knows to keep replies short and spoken-word friendly.
"""

import os, sys, time, wave, tempfile, threading, subprocess
import urllib.request, urllib.error, json
import numpy as np
from typing import Optional

try:
    import sounddevice as sd
except ImportError:
    sys.exit("Missing: pip install sounddevice")

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing: pip install openai")

# ─── CONFIG ──────────────────────────────────────────────────────────────────

OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "")
DISCORD_BOT_TOKEN  = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_USER_TOKEN = os.environ.get("DISCORD_USER_TOKEN", "")
DISCORD_CHANNEL    = os.environ.get("DISCORD_CHANNEL_ID", "1476655601106026577")
K2S0_BOT_ID        = os.environ.get("K2S0_BOT_ID", "1476128387822129236")

SAMPLE_RATE        = 16000
CHANNELS           = 1
MIN_SPEECH_SECS    = 0.4
TTS_VOICE          = "fable"
POLL_INTERVAL      = 0.5    # seconds between reply polls — fast
REPLY_TIMEOUT      = 60     # max seconds to wait for K2S0 reply

client = OpenAI(api_key=OPENAI_API_KEY)

# ─── DISCORD ─────────────────────────────────────────────────────────────────

UA = "DiscordBot (https://github.com/seanMcKenzie/dev-team-showcase, 1.0)"

def discord_get(path: str) -> list:
    """GET request using bot token (for reading)."""
    auth = DISCORD_BOT_TOKEN if DISCORD_BOT_TOKEN.startswith("Bot ") else f"Bot {DISCORD_BOT_TOKEN}"
    req = urllib.request.Request(
        f"https://discord.com/api/v10{path}",
        headers={"Authorization": auth, "Content-Type": "application/json", "User-Agent": UA}
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 429:
            time.sleep(2)
        return []
    except Exception:
        return []

def discord_post(text: str) -> Optional[str]:
    """POST message as Sean (user token). Returns message ID."""
    req = urllib.request.Request(
        f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL}/messages",
        data=json.dumps({"content": text}).encode(),
        headers={"Authorization": DISCORD_USER_TOKEN, "Content-Type": "application/json", "User-Agent": UA},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read()).get("id")
    except Exception as e:
        print(f"   [POST error] {e}", flush=True)
        return None

def wait_for_reply(after_id: str) -> Optional[str]:
    """Poll for K2S0's reply after a given message ID."""
    print("⏳ Waiting for K2S0...", flush=True)
    deadline = time.time() + REPLY_TIMEOUT
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        msgs = discord_get(f"/channels/{DISCORD_CHANNEL}/messages?after={after_id}&limit=20")
        if not isinstance(msgs, list):
            continue
        for msg in msgs:
            author_id = msg.get("author", {}).get("id", "")
            content = msg.get("content", "").strip()
            if author_id == K2S0_BOT_ID and content:
                return content
    return None

# ─── AUDIO CAPTURE ───────────────────────────────────────────────────────────

def record_ptt() -> Optional[np.ndarray]:
    """Push-to-talk: Enter to start, Enter to stop."""
    input("\n⏎  Press ENTER to speak...")
    print("🔴 Recording — press ENTER to stop", flush=True)
    frames = []
    stop_event = threading.Event()

    def cb(indata, frame_count, t, status):
        frames.append(indata.copy())

    def wait_for_enter():
        input()
        stop_event.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        dtype="float32", blocksize=512, callback=cb):
        while not stop_event.is_set():
            time.sleep(0.05)

    if not frames:
        return None
    audio = np.concatenate(frames).flatten()
    duration = len(audio) / SAMPLE_RATE
    if duration < MIN_SPEECH_SECS:
        print("   Too short, ignored.", flush=True)
        return None
    print(f"   Captured {duration:.1f}s", flush=True)
    return audio

def to_wav(audio: np.ndarray) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    pcm = (audio * 32767).astype(np.int16)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return tmp.name

def transcribe(wav_path: str) -> str:
    print("📝 Transcribing...", flush=True)
    with open(wav_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1", file=f, language="en"
        )
    os.unlink(wav_path)
    return result.text.strip()

# ─── TTS ─────────────────────────────────────────────────────────────────────

def speak(text: str):
    # Strip markdown for cleaner TTS
    clean = text.replace("**", "").replace("*", "").replace("`", "").replace("#", "")
    try:
        r = client.audio.speech.create(model="tts-1", voice=TTS_VOICE, input=clean[:400])
        raw = tempfile.mktemp(suffix=".mp3")
        processed = tempfile.mktemp(suffix=".mp3")
        r.stream_to_file(raw)
        sox = subprocess.run(
            ["sox", raw, processed, "pitch", "-80", "treble", "+3"],
            capture_output=True
        )
        playfile = processed if sox.returncode == 0 else raw
        subprocess.run(["afplay", "-v", "0.7", playfile], check=False)
        for f in [raw, processed]:
            try:
                os.unlink(f)
            except Exception:
                pass
    except Exception as e:
        print(f"   TTS error: {e}", flush=True)
        subprocess.run(["say", clean[:200]], check=False)

# ─── MAIN ────────────────────────────────────────────────────────────────────

def validate():
    errors = []
    if not OPENAI_API_KEY:     errors.append("OPENAI_API_KEY")
    if not DISCORD_BOT_TOKEN:  errors.append("DISCORD_BOT_TOKEN")
    if not DISCORD_USER_TOKEN: errors.append("DISCORD_USER_TOKEN")
    if errors:
        print(f"❌ Missing env vars: {', '.join(errors)}")
        print("Run: set -a && source ~/.openclaw/.env && set +a")
        sys.exit(1)

def run():
    validate()
    print("─" * 55)
    print("  K2S0 Voice Interface v4 — Full Agent Mode")
    print("  Talking to the REAL K2S0 via Discord")
    print("  Push-to-talk | Ctrl+C to quit")
    print("─" * 55)

    while True:
        try:
            audio = record_ptt()
            if audio is None:
                continue

            wav = to_wav(audio)
            text = transcribe(wav)
            if not text:
                print("   (no transcription)", flush=True)
                continue

            print(f"🗣  You: {text}", flush=True)

            # Tag with [voice] so K2S0 keeps the reply short and spoken-word
            msg_id = discord_post(f"[voice] {text}")
            if not msg_id:
                print("   Failed to send to Discord.", flush=True)
                continue

            reply = wait_for_reply(msg_id)
            if reply:
                print(f"🔊 K2S0: {reply[:100]}{'...' if len(reply)>100 else ''}", flush=True)
                speak(reply)
            else:
                print("   (no reply within timeout)", flush=True)
                subprocess.run(["say", "No response from K2S0."], check=False)

        except KeyboardInterrupt:
            print("\nShutting down.")
            break
        except Exception as e:
            print(f"⚠️  {e}", flush=True)
            time.sleep(1)

if __name__ == "__main__":
    run()
