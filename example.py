"""Speak one reply with Kokoro + kokoro-natural-speech.

    pip install kokoro-onnx soundfile
    # model files: https://github.com/thewh1teagle/kokoro-onnx/releases (kokoro-v1.0.onnx, voices-v1.0.bin)
    python example.py
"""
import soundfile as sf
from kokoro_onnx import Kokoro

from kokoro_natural import Speaker

kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")
sp = Speaker(kokoro, voice="af_heart")                              # defaults; or settings="my_settings.json"

text = "Yep. Their number is (212) 555-0123, and they close at 9:30 pm. It's $1,249.99 for 2 nights, right?"
print(sp.normalize(text))                                           # see what Kokoro will actually be given
audio, sr = sp.speak(text)
sf.write("out.wav", audio, sr)

# Streaming: start playing the first sentence while the rest renders
for i, (piece, sr) in enumerate(sp.stream("Alright, here's the plan. Leave at 7:15 am and take the 405.")):
    sf.write(f"out_piece{i}.wav", piece, sr)
