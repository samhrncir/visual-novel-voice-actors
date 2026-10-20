# VN Voice Actors

Voices the visual novel you're playing. The app sits on one screen and watches the game on your other screen. Every character gets their own voice, and each line is performed with a tone that fits it: whispered, teasing, angry, close to tears.

## How it works

1. **Watch.** Several times a second the app looks at the dialogue box you selected. It waits until a new line has finished typing out, and it learns to ignore blinking "continue" arrows and animated backgrounds.
2. **Read.** It sends a picture of the box, plus a small picture of the whole scene so Claude can see facial expressions, to Claude. Claude returns the speaker, the exact text, and how it should sound: emotion, intensity, whisper or shout, pace, and a short acting note such as "teasing, smug, drawn-out vowels".
3. **Cast.** The first time a character speaks, they get a voice matched to their gender, age and personality, different from everyone else's. The cast is saved per game, and you can change any voice.
4. **Speak.** The line is voiced with that direction by one of two engines:
   - **ElevenLabs** (cloud). v4 and v3 read audio tags such as `[whispers]`, `[laughs]` and `[sighs]` that Claude places in the line.
   - **Qwen3-TTS on your GPU** (free, private). It follows a plain-English instruction such as "Very scared, voice trembling, whispering."

If you click ahead while a line is being read, the app skips it and stops the old audio, like a voiced VN does.

## Setup

`setup.bat` has already been run on this PC. To set up somewhere else, install [uv](https://docs.astral.sh/uv/) and run `setup.bat`. Use `setup.bat --no-gpu` to skip the ~3 GB of local GPU voice packages.

Start the app with **`run.bat`**. Use `run.bat console` if you want to see log output.

## First run

1. **Settings → Claude:** paste your Anthropic API key (from [console.anthropic.com](https://console.anthropic.com/settings/keys)).
2. **Voices:** pick an engine in the top bar.
   - *Local GPU* works right away. The first time, it loads the model in about 15 seconds.
   - *ElevenLabs* needs your key in Settings → Voices.
3. Click **New game…** and type the game's title. Claude uses it to recognise characters it knows.
4. Under **Game screen**, choose the screen the game is on, click **Select dialogue box…**, drag a box around the text *and* the name plate, then press Enter.
5. Press **Start** (or **F9**) and play.

## Hotkeys

These work while the game has focus:

| Key | Action |
|---|---|
| **F9** | Start / pause following the game |
| **F8** | Read the current line now |
| **F10** | Replay the last line |
| **F11** | Stop speaking |

You can change them in Settings → Hotkeys.

## What it costs

| Part | Cost |
|---|---|
| Claude (reads each line) | Roughly $0.01 per line on **Opus 5.5** (the default and most accurate), about $0.005 on **Sonnet 5.5**, well under $0.001 on **Haiku 5.5**. The status bar shows a running estimate. Turning off "send a small picture of the whole scene" lowers it. |
| ElevenLabs | Characters spoken, taken from your plan's credits. Flash v2.5 costs half as much as v4/v3, but can't do emotion tags. |
| Local GPU | Free. It uses about 4.5 GB of video memory. |

Haiku 5.5 is also the fastest reader. Try it in Settings → Claude if lines feel slow to start.

## Tips

- **A line gets read twice, or is cut off mid-typing:** raise *Wait for text to stop changing* (Settings → Reading).
- **Short lines get missed:** lower *New line when this much changes*, or tap **F8**.
- **Wrong voice:** pick another in the Cast table, or click *Pick another voice*. Fixing a character's gender or age also helps future casting.
- **Voices in the local engine:** *built-in* mode has 9 voices with the best emotion control. *Designed* mode (Settings → Voices) gives unlimited voices described in words, needs another 4.5 GB download, and a voice can drift slightly between lines.
- **Don't want narration or inner thoughts voiced:** turn them off in Settings → Reading.

## Where things are kept

- `data/settings.json`: settings.
- `data/profiles/<game>.json`: each game's cast and dialogue box.
- `data/vn_voice.log`: log file.
- API keys are stored in Windows Credential Manager under `vn-voice-actors`, not in files.

## Code layout

```
vn_voice/
  screen.py      monitor list, screen grabs, "is there a new finished line?" detection
  reader.py      Claude: reads the box, returns speaker/text/emotion/acting direction
  casting.py     picks a distinct, fitting voice per character; saves the cast
  pipeline.py    capture thread -> reader thread -> audio thread
  audio.py       one audio thread: generates and plays speech, interruptible, keeps replays
  hotkeys.py     global hotkeys (Win32 RegisterHotKey)
  tts/           ElevenLabs and local Qwen3-TTS engines, plus tone helpers
  ui/            main window, dialogue-box picker, settings
tests/           pytest suite (no network or audio needed)
```

To run the tests: `.venv\Scripts\python -m pytest tests -q`
