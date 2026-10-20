"""Render before.mp3 (plain Kokoro) and after.mp3 (kokoro_natural) from the same text, so you can hear the difference.

    python make_samples.py            # needs kokoro-v1.0.onnx + voices-v1.0.bin here, and ffmpeg for mp3
"""
import subprocess

import numpy as np
import soundfile as sf

from kokoro_natural import Speaker

LINES = [
    "Yep. I booked it for 7:15 pm on 12/24. Their number is (212) 555-0123, and the total is $1,249.99 for 2 nights, right?",
    "Alright, here's the plan. The Airbnb is at 4804 N Main St, about 12 mins from the airport. Your ETA is 4:55 pm.",
    "Honestly? The RTX 4090 is ~30% faster at 1080p. Ooh, and it's $1.5k right now.",
    "Grab a Chick-fil-A on the way, it's open Mon-Sat 6:30am-10pm.",
]


def to_mp3(audio, sr, path):
    wav = path[:-4] + ".wav"
    sf.write(wav, audio, sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-b:a", "96k", path], check=True)
    subprocess.run(["rm", "-f", wav])


def render(kokoro, voice="af_heart", speed=1.0):
    sp = Speaker(kokoro, voice=voice, speed=speed)
    before, after, sr = [], [], 24000
    for line in LINES:
        a, sr = kokoro.create(line, voice=voice, speed=speed, lang=sp.lang)      # how Kokoro is usually called
        before += [a, np.zeros(int(0.9 * sr), np.float32)]
        a, sr = sp.speak(line)
        after += [a, np.zeros(int(0.9 * sr), np.float32)]
    to_mp3(np.concatenate(before), sr, "before.mp3")
    to_mp3(np.concatenate(after), sr, "after.mp3")


if __name__ == "__main__":
    from kokoro_onnx import Kokoro
    render(Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin"))
