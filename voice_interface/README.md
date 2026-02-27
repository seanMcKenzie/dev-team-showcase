# K2S0 Voice Interface

Duplex voice communication with K2S0.

```
You speak → Whisper STT → Discord #k2 → K2S0 replies → TTS playback
```

## Setup

```bash
cd voice_interface
pip install -r requirements.txt
```

On macOS you may also need PortAudio (for sounddevice):
```bash
brew install portaudio
```

## Required env vars

```bash
export OPENAI_API_KEY="sk-..."
export DISCORD_BOT_TOKEN="your_token_here"   # Bot token OR your personal token
export DISCORD_CHANNEL_ID="1476655601106026577"  # already set as default
export K2S0_BOT_ID="your_k2s0_bot_user_id"  # optional but recommended
```

## Run

```bash
python voice_interface.py
```

Then just talk. It listens, detects silence, transcribes, sends to Discord,
waits for K2S0 to reply, and plays it back.

## How it works

1. **Mic capture** — `sounddevice` streams from your default mic
2. **VAD** — simple RMS threshold; starts recording on speech, stops after ~1.8s silence
3. **STT** — OpenAI Whisper API
4. **Discord** — posts transcript to #k2 as you
5. **Reply detection** — polls the channel for K2S0's response
6. **Playback** — if K2S0 sends audio (TTS attachment), plays via `afplay`; otherwise falls back to macOS `say -v Fred`

## Notes

- `DISCORD_BOT_TOKEN`: If using a bot token, prefix with `Bot ` — the script handles both.
  If you want messages to appear as *you* (not a bot), use your personal Discord token.
- `K2S0_BOT_ID`: Get this from Discord dev mode → right-click the bot → Copy ID.
  Without it, the script matches any bot reply in the channel.
- TTS quality: K2S0's built-in TTS (ElevenLabs) sounds better than `say -v Fred`.
  The script prefers audio attachments when present.
