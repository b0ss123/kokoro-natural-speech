"""Only the text rules: no model needed. Feed the result to any TTS (strip the [hold] tag if your TTS can't use it)."""
from kokoro_natural import normalize

for line in ["Call (415) 555-0147 before 4:55 pm.",
             "It's $1.5k, or about $59/day for the SUV.",
             "Head to 4804 N Main St, it's ~12 mins away.",
             "Python 3.13 runs on the RTX 4090 at 1080p."]:
    print(f"{line}\n  -> {normalize(line).replace('[hold] ', '')}\n")

# your own additions on top of the defaults
print(normalize("Meet at Nguyen's at 7pm.", {"respell": {"Nguyen": "Win"}}))
