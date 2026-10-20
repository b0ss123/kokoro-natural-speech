# kokoro-natural-speech

Make [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) sound like a person talking, not a screen reader.

Kokoro's voice quality is great, but it reads text literally: `$1,249.99` is mangled, `ETA` becomes "eeda", `Airbnb`
becomes "airbimp", `~30%` is read "tilde thirty percent", and long sentences get random breath pauses. This is one Python
file that sits in front of Kokoro and fixes that: text rules, exact phonemes for problem words, and pause timing.

Kokoro itself is not modified. It works with [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx), and the text
rules work with any TTS.

## Hear it

`before.mp3` is plain Kokoro and `after.mp3` is the same text through kokoro-natural-speech (voice `af_heart`, speed 1.0).
Download them from this repo and listen. Text used (from `make_samples.py`):

> Yep. I booked it for 7:15 pm on 12/24. Their number is (212) 555-0123, and the total is $1,249.99 for 2 nights, right?
> Alright, here's the plan. The Airbnb is at 4804 N Main St, about 12 mins from the airport. Your ETA is 4:55 pm.
> Honestly? The RTX 4090 is ~30% faster at 1080p. Ooh, and it's $1.5k right now.
> Grab a Chick-fil-A on the way, it's open Mon-Sat 6:30am-10pm.

To check this objectively, both files were run through a speech recognizer. Here's what it heard:

| Written | Plain Kokoro | With kokoro-natural-speech |
|---|---|---|
| 12/24 | "12-24" | "December 24th" |
| Airbnb | "airbimp" | "Air B and B" |
| ETA | "EDA" | "ETA" |
| ~30% | "tilde 30%" | "about 30%" |
| $1.5k | "$1.5K" | "$1,500" |
| Chick-fil-A | "Chickvilla" | "Chick-fil-A" |
| Mon-Sat 6:30am-10pm | "months at 6,30am 10pm" | "Monday to Saturday 6:30 a.m. to 10 p.m." |
| 4804 N Main St | "4804N Main Street" | "4804 North Main Street" |
| 12 mins | "12 mins" | "12 minutes" |

The timing doesn't show in a transcript: a short break before the phone number, a real beat after "Yep" and before
"right?", and no random mid-sentence pauses. That part you have to hear.

## Quick start

```bash
pip install kokoro-onnx soundfile
# model files from https://github.com/thewh1teagle/kokoro-onnx/releases: kokoro-v1.0.onnx and voices-v1.0.bin
```

Copy `kokoro_natural.py` next to your code (it's one file, needs only numpy), or
install it with `pip install git+<this repo URL>`.

```python
import soundfile as sf
from kokoro_onnx import Kokoro
from kokoro_natural import Speaker

sp = Speaker(Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin"), voice="af_heart")

audio, sr = sp.speak("Yep. Their number is (212) 555-0123, and they close at 9:30 pm, right?")
sf.write("out.wav", audio, sr)

for piece, sr in sp.stream("Alright, here's the plan. Leave at 7:15 am."):   # play sentence 1 while the rest renders
    ...
```

**Text rules only** (for Piper, XTTS, a cloud TTS, anything):

```python
from kokoro_natural import normalize

normalize("It's $1,249.99, call (415) 555-0147 before 4:55 pm.")
# "It's one thousand two hundred forty-nine dollars and ninety-nine cents, call [hold] four one five, five five five,
#  oh one four seven before four fifty five P M."
```

`[hold]` marks the short break before a phone number. `Speaker` turns it into silence. If you use `normalize()` alone,
remove it or map it to your TTS's pause syntax. If your text comes from a chat model, run `clean_markdown()` first
(drops code blocks, says "link" for URLs, strips `*`, `#` and similar).

See `example.py` and `example_text_only.py`.

## What it fixes

**Text rules**
- **Money:** `$1,249.99`, `$1.5k`, `$2.3M`, `$300-400`, `$20/mo`
- **Times:** `4:55 pm` becomes "four fifty five P M", `10:00` becomes "ten o'clock", `7:05` becomes "seven oh five", and `9am-5pm` becomes "9 A M to 5 P M"
- **Dates and years:** `12/24/2026`, `Dec 5`, `March 2025`, `since 2019`, `the 90s`. Simple fractions like `1/2` and `3/4` are left as fractions, not read as dates.
- **Phone numbers:** digit by digit in groups, with a short break before the number
- **Addresses:** `4804 N Main St` becomes "forty-eight oh four North Main Street", plus `Blvd`, `Pkwy`, `Dr` (but not Dr Pepper), `I-35`, `the 405`
- **Units:** `12 mins`, `14mph`, `65W`, `225g`, `5000mAh`, `72°F`, `1 1/2 tsp`, `megs/s`; singular after 1 ("1 hour")
- **Tech:** `RTX 4090` becomes "forty ninety", `1080p`, `Python 3.13`, `Q4_K_M`, `GGUF`, `llama.cpp`, `GPU`, `API`, `SSD`
- **Symbols:** `~`, `->`, `&`, `w/`, `vs.`, `#3`, `3x`, `24/7`, `50/50`, `N/A`, and `example.com` becomes "example dot com"
- **Sports scores:** `won 34-30` becomes "34 to 30"
- **"live":** "we live here" vs "streaming live"
- **Respellings** for words Kokoro gets wrong (Airbnb, H-E-B, Tel Aviv, AMC, ...). Add your own.
- **Interjections Kokoro never says well** ("Ooh", "Yikes", "Ew") are dropped at the start of a sentence. "Nope," / "Wow," get a dash, so there's a real beat after them.
- `Honestly? I'd...` becomes `Honestly, I'd...`, and `Well... maybe` becomes `Well. Maybe` (Kokoro's pitch on these was off).

**Exact phonemes.** Some words no spelling trick fixes. `ETA` and `Chick-fil-A` get hand-picked IPA phonemes, and the
words around them are phonemized in context (said alone, "there's a" came out "there's AY").

**Audio timing**
- Kokoro renders one sentence at a time. In one long call it leaves about 0.7 s of dead air at every period.
- Pauses between sentences depend on context: short after "Yep.", longer after a long sentence or a question, shorter when the next sentence starts with "and" / "but" / "so". A small fixed variation keeps it from sounding like a metronome.
- Stray "breaths" Kokoro takes mid-sentence where there's no comma are shortened to a normal word gap. Pauses at real commas are kept but capped.
- A 1-2 word sentence is glued to the next one ("Yep. Done." becomes "Yep, done.") so it doesn't sound clipped, then the pause after "Yep" / "Alright" is stretched on the audio.
- Before a tag question (", right?") the short dip Kokoro leaves is stretched into a real pause. The sentence is still rendered whole, so "right" keeps its natural rising tone.

## Settings

Everything is in `DEFAULTS` at the top of `kokoro_natural.py`. Pass overrides as a dict or a JSON file. They merge on
top of the defaults:

```python
Speaker(kokoro, settings={"respell": {"Nguyen": "Win"}})           # add a respelling
Speaker(kokoro, settings={"respell": {"Airbnb": None}})            # None removes a default entry
Speaker(kokoro, settings="my_settings.json")                       # see my_settings.json
normalize(text, {"ampm_letters": False})                           # "4:55 in the afternoon"
```

| Key | What it does |
|---|---|
| `respell` | word → spelling Kokoro says right |
| `phonemes` | word → exact IPA (espeak style, as Kokoro uses), for words no respelling fixes |
| `say_as` | phrase → replacement (default `I'm gonna` → `I gonna`: Kokoro held "I'm" too long) |
| `gaps` | seconds between sentences: `base`, `per_word`, `question`, `comma`, `colon`, `min`, `max`, `jitter`, ... |
| `tighten` | stray-pause fix: `stray_min` (longer than this and not at punctuation = stray), `stray_to`, `punct_cap` |
| `head_pause` | pause after an opening word: `{"yep": 0.9, "alright": 0.4}` |
| `tag_pause` | pause before a tag question: `{"right": 0.5}` |
| `phone_pause`, `phone_pause_secs` | break before a phone number (on, 0.35 s) |
| `interj_drop` | interjections dropped at sentence start (`"Oh,"` with the comma means only "Oh," and not "Oh no") |
| `interj_dash` | interjections that get a dash instead of a comma |
| `q_opener_comma` | `Honestly?` becomes `Honestly,` |
| `word_stop` | `Okay, so` becomes `Okay. So` for the words listed (empty by default) |
| `ellipsis_word_stop` | `Well... maybe` becomes `Well. Maybe` |
| `ampm_letters` | `P M` (on) or "in the afternoon" (off) |
| `millions_word` | `1.2M` becomes "1.2 million" |
| `glue_short`, `split_sentences`, `split_dash` | sentence splitting |

`pieces(text, render, cfg)` is the timing pipeline on its own. Give it any `render(sentence) -> (audio, sample_rate)`
and it works with other TTS models too.

## Be aware

- **Tuned by ear for one setup:** voice `af_heart`, US English, short conversational replies from a voice assistant.
  Every pause length and respelling was picked by one listener comparing A/B samples. Other Kokoro voices will probably
  benefit from the text rules (those are about meaning, not sound), but the pause numbers and phonemes may need tuning.
  British voices (`b*`) use espeak `en-gb` and are not tested.
- US conventions: `$`, month/day dates (`12/24`), miles, Fahrenheit, US street abbreviations.
- These are regex rules, not a language model. They're careful about context (`St` is only "Street" after a house number,
  `Dr` isn't "Drive" before a capitalized word, `1/2` isn't a date), but odd text can still trip them. `normalize()` shows
  exactly what Kokoro will get, so it's easy to check.
- Plain numbers like "12" or "2 nights" are left as digits. Kokoro reads those fine.

## Credits

[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad (Apache-2.0) and
[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) by thewh1teagle (MIT). This project only wraps them.

## License

MIT, see `LICENSE`.
