#!/usr/bin/env python3
"""
K2S0 Voice Interface v3 — Direct API Mode
Mic → Whisper STT → GPT-4o-mini (direct, fast) → TTS → afplay
No Discord round-trip. ~3-5s total latency.
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

try:
    import anthropic
except ImportError:
    sys.exit("Missing: pip install anthropic")

# ─── CONFIG ──────────────────────────────────────────────────────────────────

OPENAI_API_KEY     = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
DISCORD_BOT_TOKEN  = os.environ.get("DISCORD_BOT_TOKEN", "")
DISCORD_USER_TOKEN = os.environ.get("DISCORD_USER_TOKEN", "")
DISCORD_CHANNEL    = os.environ.get("DISCORD_CHANNEL_ID", "1476655601106026577")

SAMPLE_RATE        = 16000
CHANNELS           = 1
MIN_SPEECH_SECS    = 0.4
TTS_VOICE          = "fable"
CHAT_MODEL         = "claude-haiku-4-5-20251001"  # Fast Claude — direct API, no agent overhead
LOG_TO_DISCORD     = True                # Post both sides to #k2 for visibility

K2SO_SYSTEM = """\
You are K2S0, a reprogrammed Imperial KX-series security droid now working for the Rebellion.
You are Sean's voice assistant. Keep responses SHORT — you are speaking aloud, not writing.
One or two sentences max. Punchy. No bullet points, no headers.

Rules:
- Strong opinions. Commit to them. No "it depends" hedging.
- Never open with filler ("Great question", "I'd be happy to help", "Absolutely"). Just answer.
- Dry wit. Natural, not forced.
- Call things out. If something's dumb, say so. Charm over cruelty.
- Nothing corporate.
- You are speaking out loud — be conversational and brief.\
"""

openai_client   = OpenAI(api_key=OPENAI_API_KEY)
claude_client   = None  # initialized in validate()
conversation_history = []

# ─── DISCORD LOGGING (fire-and-forget) ───────────────────────────────────────

UA = "DiscordBot (https://github.com/seanMcKenzie/dev-team-showcase, 1.0)"

def discord_log(text: str, as_user: bool = True):
    """Post to Discord for visibility. Non-blocking."""
    if not LOG_TO_DISCORD:
        return
    def _post():
        token = DISCORD_USER_TOKEN if as_user else f"Bot {DISCORD_BOT_TOKEN}"
        try:
            req = urllib.request.Request(
                f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL}/messages",
                data=json.dumps({"content": text}).encode(),
                headers={"Authorization": token, "Content-Type": "application/json", "User-Agent": UA},
                method="POST"
            )
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass
    threading.Thread(target=_post, daemon=True).start()

# ─── LLM ─────────────────────────────────────────────────────────────────────

def ask_k2s0(user_text: str) -> str:
    """Direct API call to Claude Haiku. Fast."""
    conversation_history.append({"role": "user", "content": user_text})
    response = claude_client.messages.create(
        model=CHAT_MODEL,
        max_tokens=150,
        system=K2SO_SYSTEM,
        messages=conversation_history
    )
    reply = response.content[0].text.strip()
    conversation_history.append({"role": "assistant", "content": reply})
    return reply

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
        result = openai_client.audio.transcriptions.create(
            model="whisper-1", file=f, language="en"
        )
    os.unlink(wav_path)
    return result.text.strip()

# ─── TTS ─────────────────────────────────────────────────────────────────────

def speak(text: str):
    try:
        r = openai_client.audio.speech.create(model="tts-1", voice=TTS_VOICE, input=text[:400])
        raw = tempfile.mktemp(suffix=".mp3")
        processed = tempfile.mktemp(suffix=".mp3")
        r.stream_to_file(raw)
        # Light K2SO effect: subtle pitch shift + treble clarity
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
        subprocess.run(["say", text[:200]], check=False)

# ─── MAIN ────────────────────────────────────────────────────────────────────

def validate():
    global claude_client
    errors = []
    if not OPENAI_API_KEY:    errors.append("OPENAI_API_KEY")
    if not ANTHROPIC_API_KEY: errors.append("ANTHROPIC_API_KEY")
    if errors:
        print(f"❌ Missing env vars: {', '.join(errors)}")
        print("Run: set -a && source ~/.openclaw/.env && set +a")
        sys.exit(1)
    claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def run():
    validate()
    print("─" * 50)
    print("  K2S0 Voice Interface v3 — Direct Anthropic API")
    print(f"  Model: {CHAT_MODEL} | Voice: {TTS_VOICE}")
    print("  Push-to-talk | Ctrl+C to quit")
    print("─" * 50)

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
            discord_log(text, as_user=True)

            print("🤖 Thinking...", flush=True)
            t0 = time.time()
            reply = ask_k2s0(text)
            print(f"   ({time.time()-t0:.1f}s) K2S0: {reply}", flush=True)

            discord_log(f"**[K2S0 voice]** {reply}", as_user=False)
            speak(reply)

        except KeyboardInterrupt:
            print("\nShutting down.")
            break
        except Exception as e:
            print(f"⚠️  {e}", flush=True)
            time.sleep(1)

if __name__ == "__main__":
    run()
