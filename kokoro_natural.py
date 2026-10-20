"""kokoro_natural: make Kokoro TTS sound like a person. One file: copy it next to your code.

    from kokoro_onnx import Kokoro
    from kokoro_natural import Speaker

    sp = Speaker(Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin"), voice="af_heart")
    audio, sr = sp.speak("Yep. Their number is (212) 555-0123, ETA 4:55 pm, right?")

    from kokoro_natural import normalize     # just the text rules, for any TTS
    normalize("It's $1,249.99 at 9:30 pm")    # -> "It's one thousand two hundred forty-nine dollars and ninety-nine cents at nine thirty P M"

Kokoro itself is not modified. Three layers sit around it:
  1. text rules: numbers, money, times, phone numbers, addresses, units, abbreviations, respellings
  2. exact phonemes for words Kokoro keeps misreading (settings "phonemes"), with the words around them
     phonemized in context
  3. audio timing: stray mid-sentence pauses capped, natural gaps between sentences, a real beat before
     a ", right?" tag and after an opening "Yep" / "Alright", a short break before phone numbers

Settings: DEFAULTS below, with your overrides merged on top (a dict or a JSON file path):
    Speaker(kokoro, settings={"respell": {"Nguyen": "Win"}})     # add a respelling
    Speaker(kokoro, settings={"respell": {"Airbnb": None}})      # None removes a default entry
    Speaker(kokoro, settings="my_settings.json")
Dict sections (respell, gaps, phonemes, ...) merge key by key; lists, numbers and flags are replaced.

MIT License, see LICENSE.
"""
import copy
import json
import re
from pathlib import Path

import numpy as np

__version__ = "0.1.0"


# ---------- settings ----------
DEFAULTS = {'split_sentences': True,
 'glue_short': True,
 'split_dash': True,
 'gaps': {'comma': 0.22,
          'colon': 0.28,
          'mid_thought': 0.1,
          'short_sentence': 0.2,
          'short_words': 3,
          'base': 0.26,
          'per_word': 0.005,
          'per_word_cap': 25,
          'question': 0.04,
          'exclaim': -0.02,
          'ellipsis': 0.06,
          'continues': -0.06,
          'jitter': 0.03,
          'min': 0.18,
          'max': 0.42,
          'dash': 0.3},
 'tighten': {'enabled': True, 'silence_rms': 0.012, 'stray_min': 0.25, 'stray_to': 0.12, 'punct_window': 0.35, 'punct_cap': 0.4},
 'say_as': {"I'm gonna": 'I gonna'},
 'respell': {'Qwen': 'Kwen',
             'APK': 'Ay P K',
             'APKs': 'Ay P Ks',
             'GGUF': 'G G U F',
             'llama.cpp': 'llama C P P',
             'ESP32': 'E S P thirty-two',
             'ESP': 'E S P',
             'Ahrefs': 'eigh-hetch ref',
             'AMC': 'Ay M C',
             'Pkwy': 'Parkway',
             'Hwy': 'Highway',
             'movies': 'moovees',
             'Movies': 'Moovees',
             'Airbnb': 'Air B and B',
             'AirBnB': 'Air B and B',
             'airbnb': 'Air B and B',
             'H-E-B': 'aitch ee bee',
             'HEB': 'aitch ee bee',
             'Tel Aviv': 'Tellaveev',
             'Tel-Aviv': 'Tellaveev'},
 'phonemes': {'Chick-fil-A': 'tʃɪk fɪlˈeɪ', 'ETA': 'ˌiːtiːˈeɪ'},
 'tag_pause': {'right': 0.5},
 'head_pause': {'yep': 0.9, 'yup': 0.9, 'alright': 0.4, 'all right': 0.4},
 'ampm_letters': True,
 'millions_word': True,
 'ellipsis_word_stop': True,
 'interj_dash': ['Nope', 'Nah', 'Wow', 'Whoa', 'Damn', 'Dang', 'Ugh', 'Oof', 'Geez', 'Jeez', 'Ouch', 'Boom', 'Bingo'],
 'interj_drop': ['Ooh', 'Oooh', 'Ooo', 'Yikes', 'Whoo', 'Woo', 'Eww', 'Ew', 'Oh,'],
 'word_stop': [],
 'q_opener_comma': ['Honestly'],
 'phone_pause': True,
 'phone_pause_secs': 0.35}




def load(user=None) -> dict:
    if isinstance(user, (str, Path)):
        user = json.loads(Path(user).read_text(encoding="utf-8"))
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in (user or {}).items():
        if v is None:
            cfg.pop(k, None)
        elif isinstance(v, dict) and isinstance(cfg.get(k), dict):
            for kk, vv in v.items():
                if vv is None: cfg[k].pop(kk, None)
                else: cfg[k][kk] = vv
        else:
            cfg[k] = copy.deepcopy(v)
    return cfg


def get(cfg, section, key=None, default=None):
    """cfg[section] (or cfg[section][key]), with a default when it's missing."""
    v = cfg.get(section, {} if key else default)
    return v if key is None else (v.get(key, default) if isinstance(v, dict) else default)


# ---------- numbers ----------
_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def n2w(n: int) -> str:
    """12 -> twelve, 345 -> three hundred forty-five, 2500000 -> two million five hundred thousand."""
    if n < 20: return _ONES[n]
    if n < 100: return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000: return _ONES[n // 100] + " hundred" + (" " + n2w(n % 100) if n % 100 else "")
    for div, word in ((10**9, "billion"), (10**6, "million"), (1000, "thousand")):
        if n >= div: return n2w(n // div) + " " + word + (" " + n2w(n % div) if n % div else "")


def _dec(s: str) -> str:   # "3.5" -> "three point five"
    i, _, d = s.partition(".")
    return n2w(int(i or 0)) + (" point " + " ".join(_ONES[int(c)] for c in d) if d else "")


def _money(m) -> str:
    amt, scale = m.group(1).replace(",", ""), (m.group(2) or "").lower()
    scale = {"k": "thousand", "m": "million", "b": "billion"}.get(scale, scale)
    if scale == "thousand" and "." in amt and float(amt) < 10 and float(amt) * 10 % 1 == 0 and float(amt) % 1:
        return f"{n2w(int(float(amt) * 10))} hundred dollars"          # $1.5k -> fifteen hundred dollars
    if scale: return f"{_dec(amt)} {scale} dollars"
    d, _, c = amt.partition(".")
    d, c = int(d or 0), int((c + "00")[:2]) if c else 0
    out = f"{n2w(d)} dollar{'s' if d != 1 else ''}" if d or not c else ""
    if c: out += (" and " if out else "") + f"{n2w(c)} cent{'s' if c != 1 else ''}"
    return out


def _part(h, am, cfg):
    if get(cfg, "ampm_letters", default=False):            # "four fifty-five P M" (a lone "A" is read "uh", so "eigh em")
        return " eigh em" if am else " P M"
    h = h % 12
    if am: return " at night" if h == 0 else " in the morning"
    return " in the afternoon" if h < 5 else (" in the evening" if h < 10 else " at night")


_MONTHS = "January February March April May June July August September October November December".split()
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
_DIGIT = "oh one two three four five six seven eight nine".split()
_ABBR = {"GPU": "G P U", "CPU": "C P U", "AI": "eigh eye", "API": "eigh pee eye", "USB": "U S B", "SSH": "S S H", "SSD": "S S D",
         "LLM": "L L M", "URL": "U R L", "TTS": "T T S", "STT": "S T T", "CEO": "C E O", "TV": "T V",
         "HVAC": "H vac", "RAM": "ram", "OK": "okay"}


def _ordinal(n: int) -> str:
    w = n2w(n); sep = "-" if "-" in w else " "
    head, _, last = w.rpartition(sep)
    last = _ORD.get(last, last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return (head + sep if head else "") + last


def _year(y: int) -> str:
    if 2000 <= y < 2010: return "two thousand" + (" " + n2w(y % 100) if y % 100 else "")
    if y % 100 == 0: return n2w(y // 100) + " hundred"
    return n2w(y // 100) + " " + (n2w(y % 100) if y % 100 >= 10 else "oh " + n2w(y % 100))


def _date(m) -> str:
    mo, d, y = m.group(1), m.group(2), m.group(3)
    if not y and len(mo) == 1 and len(d) == 1: return m.group(0)          # 1/2, 3/4: fractions, not dates
    mo, d = int(mo), int(d)
    if not (1 <= mo <= 12 and 1 <= d <= 31): return m.group(0)
    out = f"{_MONTHS[mo - 1]} {_ordinal(d)}"
    if y: out += ", " + _year(int(y) if len(y) == 4 else 2000 + int(y))
    return out


def _phone(m) -> str:
    return ", ".join(" ".join(_DIGIT[int(c)] for c in g) for g in m.groups())


def say_symbols(text: str, cfg: dict) -> str:
    """Month names, dates, phone numbers, fractions, degrees and symbols the way people say them."""
    _MON = {"Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April", "Jun": "June", "Jul": "July", "Aug": "August",
            "Sep": "September", "Sept": "September", "Oct": "October", "Nov": "November", "Dec": "December"}
    text = re.sub(r"\b(Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)\b\.?", lambda m: _MON[m.group(1)], text)
    _FULL = "January|February|March|April|May|June|July|August|September|October|November|December"
    text = re.sub(r"\b(" + _FULL + r") (\d{1,2})(?:st|nd|rd|th)?\b(?![:\d])(?:,? (\d{4})\b)?",
                  lambda m: (f"{m.group(1)} {_ordinal(int(m.group(2)))}" + (", " + _year(int(m.group(3))) if m.group(3) else ""))
                  if 1 <= int(m.group(2)) <= 31 else m.group(0), text)                      # December 5, 2026 -> December fifth, twenty twenty-six
    text = re.sub(r"\b(" + _FULL + r") (\d{4})\b", lambda m: f"{m.group(1)} {_year(int(m.group(2)))}", text)
    if get(cfg, "phone_pause", default=False):              # a short break before a phone number: "their number is [hold] four six nine..."
        text = re.sub(r"(?<=[A-Za-z]),?\s+(?=\(?\d{3}\)?[-. ]\d{3}[-. ]\d{4}\b)", " [hold] ", text)
    text = re.sub(r"\(?\b(\d{3})\)?[-. ](\d{3})[-. ](\d{4})\b", _phone, text)
    text = re.sub(r"\b(\d+) (1/2|1/3|2/3|1/4|3/4)(?![\d/])",
                  lambda m: m.group(1) + " and " + {"1/2": "a half", "1/3": "a third", "2/3": "two thirds", "1/4": "a quarter", "3/4": "three quarters"}[m.group(2)], text)   # 1 1/2 cups
    for frac, word in (("1/2", "half"), ("1/3", "a third"), ("2/3", "two thirds"), ("1/4", "a quarter"), ("3/4", "three quarters")):
        text = re.sub(r"(?<![\d/])" + re.escape(frac) + r"(?![\d/])", word, text)
    text = re.sub(r"\b(\d{1,2})/(\d{1,2})(?:/(\d{4}|\d{2}))?\b(?!/)", _date, text)
    text = re.sub(r"(\d+)\s?°\s?F\b", lambda m: n2w(int(m.group(1))) + " degrees", text)
    text = re.sub(r"(\d+)\s?°\s?C\b", lambda m: n2w(int(m.group(1))) + " degrees Celsius", text)
    text = re.sub(r"(\d+)\s?°", lambda m: n2w(int(m.group(1))) + " degrees", text)
    text = re.sub(r"\bw/o\b", "without", text, flags=re.I)
    text = re.sub(r"\bw/(?=\s|\w)", "with ", text, flags=re.I)
    text = re.sub(r"\bvs\.?(?=\s)", "versus", text, flags=re.I)
    text = re.sub(r"\s&\s", " and ", text)
    text = re.sub(r"\bA\s+P\s+I\b", "API", text, flags=re.I)
    text = re.sub(r"\bA\s+I\b", "AI", text, flags=re.I)
    text = re.sub(r"\b(" + "|".join(_ABBR) + r")\b", lambda m: _ABBR[m.group(1)], text)
    return text


# ---------- clock ----------
_CW = ("twelve one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
       "eighteen nineteen").split()
_CT = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}


def _clock(m, cfg):
    """9:30 -> nine thirty, 10:00 -> ten o'clock, 7:05 -> seven oh five, 21:30 -> nine thirty."""
    h, mm, ap = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower()
    if h > 23:
        return m.group(0)
    hw = _CW[h % 12]
    if mm == 0:
        out = hw if ap else f"{hw} o'clock"
    elif mm < 10:
        out = f"{hw} oh {_CW[mm]}"
    elif mm < 20:
        out = f"{hw} {_CW[mm]}"
    else:
        t, o = divmod(mm, 10)
        out = f"{hw} {_CT[t]}" + (f" {_CW[o]}" if o else "")
    if ap:
        out += _part(h, ap.startswith("a"), cfg)
    return out


def say_clock(text: str, cfg: dict) -> str:
    text = re.sub(r"\b(\d{1,2}(?::\d{2})?\s?(?:[ap]\.?m\.?)?)\s?[–—-]\s?(?=\d{1,2}(?::\d{2})?\s?[ap]\.?m\b)", r"\1 to ", text, flags=re.I)   # 9am-5pm
    text = re.sub(r"(?<![:\d])\b(\d{1,2})\s?([ap])(?:\.\s?m\.?|m\b)",
                  lambda m: m.group(1) + _part(int(m.group(1)), m.group(2).lower() == "a", cfg), text, flags=re.I)   # 8am
    return re.sub(r"\b(\d{1,2}):([0-5]\d)(?![\d:])(?:\s?([ap])(?:\.\s?m\.|m\b))?", lambda m: _clock(m, cfg), text, flags=re.I)


# ---------- addresses, roads, units ----------
def _pair(n: str) -> str:
    """House numbers the way people say them: 4804 -> forty-eight oh-four, 7500 -> seventy-five hundred, 12830 -> twelve eight thirty."""
    def two(x):
        return "hundred" if x == "00" else ("oh " + _ONES[int(x)] if x[0] == "0" else n2w(int(x)))
    if n.endswith("000") or (len(n) == 3 and n.endswith("00")): return n2w(int(n))
    if len(n) == 3: return _ONES[int(n[0])] + " " + two(n[1:])
    if len(n) == 4: return n2w(int(n[:2])) + " " + two(n[2:])
    if len(n) == 5: return n2w(int(n[:2])) + " " + _pair(n[2:])
    return n


def _hwy(n: str) -> str:   # 121 -> one twenty-one, 1210 -> twelve ten, 75 -> seventy-five
    return n2w(int(n)) if len(n) <= 2 else _pair(n)


_SUF = {"Rd": "Road", "Pkwy": "Parkway", "Fwy": "Freeway", "Hwy": "Highway", "Blvd": "Boulevard", "Ave": "Avenue", "Ln": "Lane",
        "Ct": "Court", "Cir": "Circle", "Trl": "Trail", "Expy": "Expressway", "Pl": "Place", "Wy": "Way"}
_DIR = {"N": "North", "S": "South", "E": "East", "W": "West"}
_UNIT = {"mph": "miles per hour", "kmh": "kilometers per hour", "km": "kilometers", "lbs": "pounds", "lb": "pounds", "oz": "ounces",
         "hrs": "hours", "hr": "hours", "mins": "minutes", "min": "minutes", "secs": "seconds", "sec": "seconds", "ft": "feet",
         "GiB": "gigs", "GB": "gigs", "MB": "megs", "TB": "terabytes", "GHz": "gigahertz", "h": "hours"}
_ONE_UNIT = {"hours": "hour", "minutes": "minute", "seconds": "second", "miles per hour": "mile per hour", "pounds": "pound",
             "ounces": "ounce", "feet": "foot", "kilometers": "kilometer", "gigs": "gig", "megs": "meg", "terabytes": "terabyte"}
_SAYW = {"e.g.": "for example", "i.e.": "that is", "etc.": "etcetera", "approx.": "about", "approx": "about", "TX": "Texas",
         "IMEI": "I M E I", "ETA": "E T A", "EV": "E V", "EVs": "E Vs", "BBQ": "barbecue", "SUV": "S U V", "DM": "D M", "FAQ": "F A Q"}
_STREET = r"(?:St|Street|Rd|Road|Ave|Avenue|Pkwy|Parkway|Blvd|Boulevard|Fwy|Freeway|Dr|Drive|Ln|Lane|Way|Wy|Hwy|Highway|Ct|Court|Trl|Trail|Cir|Expy)"
_DAY = {"Mon": "Monday", "Tue": "Tuesday", "Tues": "Tuesday", "Wed": "Wednesday", "Thu": "Thursday", "Thur": "Thursday", "Thurs": "Thursday",
        "Fri": "Friday", "Sat": "Saturday", "Sun": "Sunday"}
_DECADE = {"10": "tens", "20": "twenties", "30": "thirties", "40": "forties", "50": "fifties", "60": "sixties", "70": "seventies",
           "80": "eighties", "90": "nineties"}
_UNIT2 = {"mAh": "milliamp hours", "mm": "millimeters", "cm": "centimeters", "kg": "kilograms", "mg": "milligrams", "g": "grams",
          "MP": "megapixel", "Hz": "hertz", "kHz": "kilohertz", "W": "watts", "kW": "kilowatts", "kWh": "kilowatt hours",
          "Gbps": "gigabits per second", "Mbps": "megabits per second", "Kbps": "kilobits per second", "PSI": "P S I", "psi": "P S I",
          "V": "volts", "mL": "milliliters", "ml": "milliliters", "L": "liters"}
_PER = {"s": "per second", "sec": "per second", "hr": "per hour", "h": "per hour", "mo": "a month", "yr": "a year", "day": "a day", "wk": "a week"}
SILENCE_TAGS = ("pause", "beat", "breath", "tick", "hold")


def _kilo(x: str) -> str:   # 83 -> eighty-three thousand, 2.5 -> twenty-five hundred
    v = float(x)
    if v % 1 and v < 10 and (v * 10) % 1 == 0: return n2w(int(v * 10)) + " hundred"
    return (_dec(x) if v % 1 else n2w(int(v))) + " thousand"


def _fix_live(text: str) -> str:
    """Verb "liv" vs adjective "lyve". Ambiguous phrases are left alone."""
    text = re.sub(r"\blive\s+on\s+(TV|television|the radio|air|stage)\b", lambda m: "lyve on " + m.group(1), text, flags=re.I)
    text = re.sub(r"\b(recorded|broadcast|broadcasting|aired|airing|streamed|streaming|performed|shown)\s+live\b",
                  lambda m: m.group(1) + " lyve", text, flags=re.I)
    text = re.sub(r"\b(watch|watching|watched|see|seeing|seen|hear|hearing|heard|view|viewing|catch|catching)\s+it\s+live\b",
                  lambda m: m.group(1) + " it lyve", text, flags=re.I)
    text = re.sub(r"\b(is|are|was|were|be|been|being|go|goes|went|going|stay|stays|staying)\s+live\b",
                  lambda m: m.group(1) + " lyve", text, flags=re.I)
    text = re.sub(r"\blive\s+(server|stream|streaming|feed|broadcast|show|event|wire|audio|video|tv|chat|session)\b",
                  lambda m: "lyve " + m.group(1), text, flags=re.I)
    text = re.sub(r"\b(I|you|we|they|to|can|could|would|should|where|who)\s+live\b", lambda m: m.group(1) + " liv", text, flags=re.I)
    text = re.sub(r"\blive\s+(here|there|nearby|alone|together)\b", lambda m: "liv " + m.group(1), text, flags=re.I)
    return text


def say_fixes(text: str, cfg: dict) -> str:
    """Things Kokoro reads wrong in real assistant replies: roads, units, years, symbols, interjections, respellings."""
    text = _fix_live(text)
    text = re.sub(r"\[(?!(?:" + "|".join(SILENCE_TAGS) + r")\])([^\]\n]{1,40})\]", r"\1", text)   # [open now] -> open now
    text = re.sub(r"\b50/50\b", "fifty-fifty", text)
    text = re.sub(r"\s*(?:→|->|⇒)\s*", ", then ", text)                                  # Settings -> Appearance
    text = re.sub(r"~\s?(?=\d)", "about ", text)
    text = re.sub(r"(?<![\w$])\+(?=\d)", "", text)                                               # +89 degrees
    for _f, _w in (("½", "a half"), ("¼", "a quarter"), ("¾", "three quarters"), ("⅓", "a third"), ("⅔", "two thirds"), ("⅛", "an eighth")):
        text = re.sub(r"(\d)\s?" + _f, r"\1 and " + _w, text); text = text.replace(_f, _w)       # 1 ½ cups, ¼ cup
    text = re.sub(r"\b(\d+)\s?(tbsp|Tbsp|tsp)\b\.?",
                  lambda m: m.group(1) + (" tablespoon" if m.group(2).lower() == "tbsp" else " teaspoon") + ("" if m.group(1) == "1" else "s"), text)
    text = re.sub(r"\b(tbsp|tsp)\b\.?", lambda m: "tablespoon" if m.group(1) == "tbsp" else "teaspoon", text)
    text = re.sub(r"\b[Nn]/[Aa]\b", "not available", text)
    text = re.sub(r"\b(\d{1,3})\s?[–—]\s?(\d{1,3})\b", r"\1 to \2", text)                     # 6–8 pancakes (en dash = range)
    text = re.sub(r"\b(\d{1,3})-(\d{1,3})\b(?=[^.!?]{0,40}\b(?:win|won|lose|lost|beat|game|score|final|against|victory|loss)\b)", r"\1 to \2", text)
    text = re.sub(r"\b((?:won|lost|beat|win|loss|final|score|scored)\b[^.!?\d]{0,25})(\d{1,3})-(\d{1,3})\b", r"\1\2 to \3", text)   # won 34-30
    text = re.sub(r"\b(about|the|a|to|and|of|in|is|it|that|for|on|at)\s+\1\b", r"\1", text, flags=re.I)   # "about about"
    if re.search(r"\b(version|python|node|ios|android|macos|release|v\d)\b", text, re.I):
        text = re.sub(r"\b(\d{1,2})\.(\d{2})\b(?!\.\d)", lambda m: n2w(int(m.group(1))) + " point " + n2w(int(m.group(2))), text)   # Python 3.13
    if re.search(r"\b(RTX|GTX|GPU|graphics|Radeon|GeForce)\b", text):
        text = re.sub(r"\b([1-5]0[5-9]0)\b", lambda m: _hwy(m.group(1)), text)                     # the 5080 and 5070
    text = re.sub(r"\bWW ?(II|2)\b", "World War Two", text); text = re.sub(r"\bWW ?(I|1)\b", "World War One", text)
    text = re.sub(r"\bMt\.? (?=[A-Z])", "Mount ", text); text = re.sub(r"\bJr\.?(?=\W|$)", "Junior", text); text = re.sub(r"\bSr\.(?=\W|$)", "Senior", text)
    text = re.sub(r"\b(Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)\s?[–—-]\s?(Mon|Tue|Tues|Wed|Thu|Thur|Thurs|Fri|Sat|Sun)\b",
                  lambda m: _DAY[m.group(1)] + " to " + _DAY[m.group(2)], text)                     # Mon-Fri
    text = re.sub(r"(?:\b|')(19|20)?([1-9]0)s\b", lambda m: _DECADE[m.group(2)], text)              # the 70s, 1990s, '80s
    text = re.sub(r"\bQ(\d)_?0\b", r"Q\1 oh", text)
    text = re.sub(r"\bQ(\d)_K(?:_([SML]))?\b", lambda m: f"Q{m.group(1)} K" + (" " + m.group(2) if m.group(2) else ""), text)   # Q4_K_M
    text = re.sub(r"(?<![\d.$,])(\d+(?:\.\d+)?)\s?k\b(?!\s?(?:m|g|b)\b)", lambda m: _kilo(m.group(1)), text)   # 83k, 2.5k
    text = re.sub(r"\b(\d{3,4})p\b", lambda m: _hwy(m.group(1)) + " p", text)                       # 1080p -> ten eighty p
    text = re.sub(r"\b([A-Z]{1,4})[- ]?(\d{3,4})\b(?! (?:[A-Z][\w']*\.? ){0,3}" + _STREET + r"\b)",
                  lambda m: m.group(0) if m.group(1) in ("I", "US", "SH", "FM", "AM", "PM", "TX") else m.group(1) + " " + _hwy(m.group(2)), text)   # RTX 4090
    text = re.sub(r"\b(\d{3,5})(?= (?:FM|SH|US|I)[- ]?\d)", lambda m: _pair(m.group(1)), text)       # 16600 FM 423
    if re.search(r"\b(highway|freeway|tollway|exit|interchange|toll|I-\d|FM \d|route)\b|\bthe \d{3}\b", text, re.I):
        text = re.sub(r"\b(the|or|and|onto|take|via|then) (\d{3})\b(?!\s?(?:%|percent|dollars|bucks|miles|people|reviews|points|degrees))",
                      lambda m: m.group(1) + " " + _hwy(m.group(2)), text)                    # take the 359 or 121
    text = re.sub(r"\bthe (\d{3})(?= (?:or|and|to|until|exit|toll|interchange|West|East|North|South|traffic)\b|[,.;])",
                  lambda m: "the " + _hwy(m.group(1)), text)
    text = re.sub(r"(\d)\s?(mAh|mm|cm|kg|mg|MP|Hz|kHz|kW|kWh|Gbps|Mbps|Kbps|PSI|psi|mL|ml)\b", lambda m: m.group(1) + " " + _UNIT2[m.group(2)], text)
    text = re.sub(r"(\d)(g|W|V|L)\b", lambda m: m.group(1) + " " + _UNIT2[m.group(2)], text)       # 225g, 65W only when attached ("6810 W Main St" is West)
    text = re.sub(r"\b(PSI|psi)\b", "P S I", text)
    text = re.sub(r"(?<=[A-Za-z0-9])/(s|sec|hr|h|mo|yr|day|wk)\b", lambda m: " " + _PER[m.group(1)], text)   # megs/s, $20/mo
    text = re.sub(r"(?<=\s)'(em)\b", "them", text); text = re.sub(r"(?<=\s)'cause\b", "because", text)
    text = re.sub(r"(?<=\s)'til\b", "until", text); text = re.sub(r"(?<=\s)'bout\b", "about", text)
    text = re.sub(r"\b(lol|lmao|lmfao|rofl)\b[.!]?\s*", "", text, flags=re.I)
    text = re.sub(r"\bF(?:t\.?|ort) Worth\b", "Fort Worth", text)
    text = re.sub(r"\b(\d{1,5} (?:[NSEW]\.? )?(?:[A-Z][\w']*\.? ){0,2}[A-Z][\w']*) St\b(?:\.(?=\s+[a-z\d]|,))?", r"\1 Street", text)   # 1234 Main St ("St" alone could be Saint)
    text = re.sub(r"\b(\d{3,5})(?= (?:[NSEW]\.? |North |South |East |West )?(?:[A-Z][\w']*\.? ){1,3}" + _STREET + r"\b)",
                  lambda m: _pair(m.group(1)), text)                                               # 4804 Main St
    text = re.sub(r"\b([NSEW])\.?(?= (?:[A-Z][\w']*\.? ){0,3}" + _STREET + r"\b)", lambda m: _DIR[m.group(1)], text)   # 10475 N County Rd
    text = re.sub(r"((?:\b[A-Z][a-z]+ ){1,3})Dr\b(?:\.(?=\s+[a-z\d]|,))?(?! [A-Z])", r"\1Drive", text)               # Elm Dr (not Dr Pepper)
    text = re.sub(r"(?<=[A-Za-z0-9] )(Rd|Pkwy|Fwy|Hwy|Blvd|Ave|Ln|Ct|Cir|Trl|Expy|Pl|Wy)\b(?:\.(?=\s+[a-z\d]|,))?", lambda m: _SUF[m.group(1)], text)
    text = re.sub(r"\b(I|US|SH|FM)[- ]?(\d{2,4})([NSEW])?\b",
                  lambda m: {"I": "I", "US": "U S", "SH": "State Highway", "FM": "F M"}[m.group(1)] + " " + _hwy(m.group(2)) + (" " + m.group(3) if m.group(3) else ""), text)
    text = re.sub(r"\$(\d+)\s?[–—-]\s?\$?(\d+)\b", r"$\1 to $\2", text)                  # $300-400
    text = re.sub(r"\b(\d+)\s?[–—-]\s?(\d+)(?=\s?(?:mins?|minutes?|hours?|hrs?|days?|weeks?|months?|years?|miles?|mi\b|degrees|°|percent|%|bucks|dollars|people|k\b|grand))",
                  r"\1 to \2", text)                                                                # 30-35 mins
    text = re.sub(r"(\d)\s?(mph|kmh|km|lbs?|oz|hrs?|mins?|secs?|ft|GiB|GB|MB|TB|GHz|h)\b", lambda m: m.group(1) + " " + _UNIT[m.group(2)], text)   # 14mph, 24h
    text = re.sub(r"(?<![\d.])\b1 (hours|minutes|seconds|miles per hour|pounds|ounces|feet|kilometers|gigs|megs|terabytes)\b",
                  lambda m: "1 " + _ONE_UNIT[m.group(1)], text)
    text = re.sub(r"(?<![\d$,.:/])\b(1[89]\d\d|20\d\d)\b(?!\d|[.,]\d|%|\s?(?:percent|dollars|bucks|miles|people|reviews|points|calories|feet|sq|square|times))",
                  lambda m: _year(int(m.group(1))), text)                                           # since 2019
    text = re.sub(r"(?<!\w)#\s?(\d+)", r"number \1", text)
    text = re.sub(r"\b(\d+)x\b", r"\1 times", text)
    text = re.sub(r"\b([A-Za-z0-9-]+)\.(com|net|org|io|ai|dev|app)\b", r"\1 dot \2", text)
    text = re.sub(r"\b([A-Za-z])&([A-Za-z])\b", lambda m: ("Ay" if m.group(1) in "Aa" else m.group(1)) + " and " + m.group(2), text)   # A&W
    text = re.sub(r"\b([A-Za-z]{2,})/([A-Za-z]{2,})\b", r"\1 or \2", text)                          # open/close
    _ij = get(cfg, "interj_dash") or []                      # "Nope, not today" -> "Nope— not today" (a real beat)
    if _ij:
        text = re.sub(r"(^|(?<=[.!?]\s))(" + "|".join(map(re.escape, _ij)) + r")\s*,\s+(?=\w)", lambda m: m.group(1) + m.group(2) + "— ", text, flags=re.I)
    _id = get(cfg, "interj_drop") or []                      # interjections Kokoro never says well ("Ooh", "Yikes"): drop them
    if _id:
        _alt = "|".join(re.escape(w[:-1]) + r"," if w.endswith(",") else re.escape(w) + r"[,.!—]*" for w in _id)   # "Oh," only with its comma
        text = re.sub(r"(^|(?<=[.!?]\s))(?:" + _alt + r")\s+(\w)", lambda m: m.group(1) + m.group(2).upper(), text, flags=re.I)
    _ws = get(cfg, "word_stop") or []                        # "Okay, so" -> "Okay. So"
    if _ws:
        text = re.sub(r"(^|(?<=[.!?]\s))(" + "|".join(map(re.escape, _ws)) + r"),\s+(\w)", lambda m: m.group(1) + m.group(2) + ". " + m.group(3).upper(), text)
    _qc = get(cfg, "q_opener_comma") or []                   # "Honestly? I'd" -> "Honestly, I'd"
    if _qc:
        text = re.sub(r"(^|(?<=[.!?]\s))(" + "|".join(map(re.escape, _qc)) + r")\?\s+", lambda m: m.group(1) + m.group(2) + ", ", text)
    if get(cfg, "ellipsis_word_stop", default=False):         # "Well... maybe" -> "Well. Maybe"
        text = re.sub(r"(^|(?<=[.!?]\s))(\w+)(?:\.\.\.|…)\s+(\w)", lambda m: m.group(1) + m.group(2) + ". " + m.group(3).upper(), text)
    for _k, _v in (get(cfg, "say_as") or {}).items():
        text = re.sub(r"\b" + re.escape(_k).replace("'", "['’]") + r"\b", _v, text)
    text = re.sub(r"\bQwen(?=\d)", "Kwen ", text)                                                  # Qwen3.5
    _rs = get(cfg, "respell") or {}
    if _rs:
        text = re.sub(r"(?<![\w.])(" + "|".join(re.escape(k) for k in sorted(_rs, key=len, reverse=True)) + r")(?![\w])", lambda m: _rs[m.group(1)], text)
    _sw = [k for k in _SAYW if k not in (get(cfg, "phonemes") or {})]                              # an exact-phoneme pick wins over the letters table
    text = re.sub(r"(?<![\w.])(" + "|".join(re.escape(k) for k in _sw) + r")(?![\w])", lambda m: _SAYW[m.group(1)], text)
    text = re.sub(r"\b(\w+)([.,!]{0,3}\s+\1\b){2,}", r"\1", text, flags=re.I)                     # "alright... alright... alright..."
    return text


def normalize(text: str, cfg=None) -> str:
    """The whole text pass. Write numbers the way people say them, so Kokoro never reads "$400" as "dollar four hundred".
    `cfg`: None (the defaults), a settings dict or a path; see settings.load()."""
    cfg = load(cfg)
    text = say_fixes(text, cfg)
    text = say_clock(text, cfg)
    text = say_symbols(text, cfg)
    text = re.sub(r"\$(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?:\s*(k|m|b|thousand|million|billion)\b)?", _money, text, flags=re.I)
    if get(cfg, "millions_word", default=False):             # "1.23M coins" -> "1.23 million coins"
        text = re.sub(r"(?<![\w.$])(\d+(?:\.\d+)?)\s?M\b", r"\1 million", text)
    text = re.sub(r"(?<![\w.])([0-9]+(?:,[0-9]{3})*(?:\.[0-9]{1,2})?)\s+(dollars?|bucks?)\b",
                  lambda m: _dec(m.group(1).replace(",", "")) + " " + m.group(2), text, flags=re.I)   # currency without a $ sign
    _digits = {w: n for n, w in enumerate("zero one two three four five six seven eight nine".split())}
    _digits["oh"] = 0
    text = re.sub(r"\b((?:(?:zero|oh|one|two|three|four|five|six|seven|eight|nine)[- ]){1,8}(?:zero|oh|one|two|three|four|five|six|seven|eight|nine))\s+(dollars?|bucks?)\b",
                  lambda m: n2w(int("".join(str(_digits[w]) for w in re.split("[- ]", m.group(1).lower())))) + " " + m.group(2), text, flags=re.I)
    text = re.sub(r"(\d+(?:\.\d+)?)\s?%", lambda m: _dec(m.group(1)) + " percent", text)
    text = re.sub(r"\b(\d{1,2}):(\d{2})\s?([ap])\.?m\.?\b", lambda m: _clock(m, cfg), text, flags=re.I)
    text = re.sub(r"\b24/7\b", "twenty-four seven", text)
    text = re.sub(r"[ \t]{2,}", " ", text)                                       # no accidental double spaces
    text = re.sub(r"\bhuh\b", "huhh", text, flags=re.I)
    text = re.sub(r"\bnoises\b", "noyzes", text, flags=re.I)
    text = re.sub(r"\bnoise\b", "noyz", text, flags=re.I)
    text = re.sub(r"(?<![\d.])(\d+)\.(\d+)(?![\d.])", lambda m: _dec(m.group(0)), text)
    return text


def clean_markdown(text: str) -> str:
    """Chat-model output -> speakable text: drop code blocks, say "link" for URLs, strip markdown symbols."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"https?://\S+", "link", text)
    text = re.sub(r"[*_#`>|]", "", text)
    return re.sub(r"\s+", " ", text).strip()




TAG_RE = re.compile(r"\[(" + "|".join(SILENCE_TAGS) + r")\]", re.I)
TAG_SECS = {"tick": .1, "beat": .15, "pause": .3, "breath": .6}
TAG_END = re.compile(r",\s*(right)\?[\"”')]*\s*$", re.I)
HEAD_START = re.compile(r"^\s*(yep|yup|alright|all right)\b[,.!—-]*\s+\w", re.I)


# ---------- audio timing ----------
def trim_edges(a, sr, tail, lead=0.04, thr=0.01):
    """Kokoro pads every clip with silence; back-to-back pieces stack it into odd double pauses. Keep only the gap we want."""
    idx = np.where(np.abs(a) > thr)[0]
    if not len(idx): return a
    out = a[max(idx[0] - int(lead * sr), 0): min(idx[-1] + int(tail * sr), len(a))]
    short = idx[-1] + int(tail * sr) - len(a)                # Kokoro's own tail can be shorter than the pause we want: pad
    return np.concatenate([out, np.zeros(short, np.float32)]) if short > 0 else out


def widen_tag(a, sr, want, tag_min=0.25, tag_max=0.6, look=1.2):
    """", right?" needs a real pause before "right", but rendering the tag as its own chunk makes "right" sound off. So the
    sentence renders in one piece and Kokoro's short comma dip before the tag (60-220 ms) is stretched to `want` seconds.
    The dip = the last quiet run that still has a whole tag word (>= tag_min s) after it. Nothing within tag_max s of the end
    (Kokoro ran "calendar, right?" together) = leave it alone, or a stop consonant inside the word gets stretched: "c...calendar"."""
    fr = int(0.01 * sr); n = len(a) // fr
    if n < 30: return a
    env = np.sqrt(np.mean(a[:n * fr].reshape(n, fr) ** 2, axis=1))
    voiced = np.where(env > 0.06 * env.max())[0]
    if not len(voiced): return a
    end, lo = voiced[-1], max(0, voiced[-1] - int(look / 0.01))
    quiet = env < 0.03 * env.max()
    i = end
    while i > lo:
        if (end - i) * 0.01 > tag_max:
            return a
        if quiet[i]:
            j = i
            while j > lo and quiet[j - 1]: j -= 1
            if i - j + 1 >= 2 and voiced[voiced < j].size and (end - i) * 0.01 >= tag_min:
                add = want - (i - j + 1) * 0.01
                if add <= 0: return a
                mid = (i + j) // 2 * fr
                return np.concatenate([a[:mid], np.zeros(int(add * sr), np.float32), a[mid:]])
            i = j
        i -= 1
    return a


def widen_head(a, sr, want, head_min=0.15, look=1.0):
    """A real pause after an opening "Yep" / "Alright". The short glued sentence ("Yep. Done." -> "Yep, done.") renders whole;
    the first quiet run after >= head_min s of speech (within `look` s) is stretched to `want` seconds."""
    fr = int(0.01 * sr); n = len(a) // fr
    if n < 30: return a
    env = np.sqrt(np.mean(a[:n * fr].reshape(n, fr) ** 2, axis=1))
    voiced = np.where(env > 0.06 * env.max())[0]
    if not len(voiced): return a
    quiet = env < 0.03 * env.max()
    i, hi = voiced[0] + int(head_min / 0.01), min(n - 1, voiced[0] + int(look / 0.01))
    while i < hi:
        if quiet[i]:
            j = i
            while j < hi and quiet[j + 1]: j += 1
            if j - i + 1 >= 2 and voiced[voiced > j].size:
                add = want - (j - i + 1) * 0.01
                if add <= 0: return a
                mid = (i + j) // 2 * fr
                return np.concatenate([a[:mid], np.zeros(int(add * sr), np.float32), a[mid:]])
            i = j
        i += 1
    return a


def gap_after(sent: str, nxt: str, cfg: dict) -> float:
    """Pause after a sentence, by context (not a fixed metronome). Numbers live in settings "gaps"."""
    g_ = lambda k, d: get(cfg, "gaps", k, d)
    end = sent.rstrip().rstrip("\"”')")[-1:]
    if end == ":": return g_("colon", 0.28)                  # "closest first: 54th Street..." a beat, then the list
    if end == "—" and get(cfg, "split_dash", default=False):
        return g_("dash", 0.30)                                 # "Got it— get the hardware...": a real beat at the dash
    if end in ",;—":
        if end == "," and len(sent.split()) <= g_("short_words", 3) and len(nxt.split()) > 3:
            return g_("comma_short", 0.40)
        return g_("comma", 0.22)
    if end == "?" and len(sent.split()) <= g_("short_words", 3):
        return g_("question_short", 0.50)
    if end not in ".!?…": return g_("mid_thought", 0.10)   # hard cut mid-thought: keep it flowing
    n = len(sent.split())
    g = g_("short_sentence", 0.20) if n <= g_("short_words", 3) else g_("base", 0.26) + min(n, g_("per_word_cap", 25)) * g_("per_word", 0.005)
    g += g_("question", 0.04) if end == "?" else g_("exclaim", -0.02) if end == "!" else g_("ellipsis", 0.06) if end == "…" else 0
    if re.match(r"(and|but|so|because|cause|then|plus|or)\b", nxt, re.I): g += g_("continues", -0.06)   # the thought continues
    g += (sum(map(ord, sent)) % 7 - 3) * g_("jitter", 0.03) / 3   # fixed per sentence: varied but repeatable
    return round(min(max(g, g_("min", 0.18)), g_("max", 0.42)), 3)


def tighten(a, sr, text, cfg):
    """Kokoro takes mid-sentence 'breaths' in long sentences where there's no comma ("kill those tabs or ... I'm gonna").
    Silences that line up with , ; : or a dash stay (capped at punct_cap); a stray one > stray_min becomes a normal word gap."""
    t_ = lambda k, d: get(cfg, "tighten", k, d)
    fr = int(0.01 * sr); n = len(a) // fr
    if n < 30 or not t_("enabled", True): return a
    rms = np.sqrt(np.mean(a[:n * fr].reshape(n, fr) ** 2, axis=1)); sil = rms < t_("silence_rms", 0.012)
    voiced = np.where(~sil)[0]
    if not len(voiced): return a
    lo, hi = voiced[0], voiced[-1]
    marks = [m.start() / max(len(text), 1) for m in re.finditer(r"[,;:—–]|\s-\s", text)]   # where punctuation should pause
    out, i, last = [], lo, 0
    while i <= hi:
        if sil[i]:
            j = i
            while j <= hi and sil[j]: j += 1
            dur = (j - i) * 0.01
            pos = ((i + j) / 2 - lo) / max(hi - lo, 1)
            punct = any(abs(pos - m) * (hi - lo) * 0.01 < t_("punct_window", 0.35) for m in marks)    # near where a comma falls
            keep = min(dur, t_("punct_cap", 0.40)) if punct else (t_("stray_to", 0.12) if dur > t_("stray_min", 0.25) else dur)
            if keep < dur:
                mid = i * fr + int(keep * sr / 2)
                out += [a[last:mid]]; last = j * fr - int(keep * sr / 2)
            i = j
        else:
            i += 1
    out.append(a[last:])
    return np.concatenate(out)


def sentences(text: str, cfg: dict):
    """Split into sentences; 1-2 word sentences are glued to the next one with a comma ("Yep. Done." -> "Yep, done.")."""
    split = re.compile(r"(?:(?<=[.!?…])|(?<=[.!?…][\"”')])|(?<=:))\s+(?=\S)"
                       + (r"|(?<=—)\s*(?=[A-Za-z])" if get(cfg, "split_dash", default=False) else ""))
    sents = [x for x in (split.split(text) if get(cfg, "split_sentences", default=True) else [text]) if re.search(r"[A-Za-z0-9]", x)]
    glued, hold = [], ""
    for s in sents:
        s = s.strip()
        if hold:
            head, _, tail = s.partition(" ")
            if len(head) > 1 and head[0].isupper() and head[1].islower():
                head = head[0].lower() + head[1:]
            s = (hold + " " + head + (" " + tail if tail else "")).strip()
            hold = ""
        if get(cfg, "glue_short", default=True) and len(s.split()) <= 2 and s.endswith(".") and not s.endswith("..."):
            hold = s[:-1].rstrip() + ","
            continue
        glued.append(s)
    if hold:
        glued.append(hold[:-1].rstrip() + ".")
    return glued


def pieces(text: str, render, cfg: dict, sr: int = 24000):
    """The model-agnostic delivery pipeline on already-normalized text. `render(sentence) -> (audio float32, sample_rate)`
    is the only model-specific part, so this works for any TTS, not just Kokoro."""
    parts = TAG_RE.split(text)
    emitted = False
    for i, p in enumerate(parts):
        if i % 2:                                           # a silence tag
            secs = float(get(cfg, "phone_pause_secs", default=0.35)) if p.lower() == "hold" else TAG_SECS[p.lower()]
            emitted = True
            yield np.zeros(int(secs * sr), np.float32), sr
            continue
        p = re.sub(r"\[[^\]]*\]", " ", p).strip()          # unknown tags are never read out loud
        sents = sentences(p, cfg)
        for j, sent in enumerate(sents):                     # one sentence at a time: in one long call Kokoro leaves ~0.7 s at every period
            a, sr = render(sent)
            emitted = True
            a = tighten(np.asarray(a, np.float32), sr, sent, cfg)
            if (m := TAG_END.search(sent)) and get(cfg, "tag_pause", m.group(1).lower()):
                a = widen_tag(a, sr, get(cfg, "tag_pause", m.group(1).lower()))      # after tighten (it caps comma pauses)
            if (m := HEAD_START.match(sent)) and get(cfg, "head_pause", m.group(1).lower()):
                a = widen_head(a, sr, get(cfg, "head_pause", m.group(1).lower()))
            yield trim_edges(a, sr, tail=gap_after(sent, sents[j + 1] if j + 1 < len(sents) else "", cfg)), sr
    if not emitted:
        yield np.zeros(int(.1 * sr), np.float32), sr


# ---------- Kokoro ----------
class Speaker:
    def __init__(self, kokoro, voice: str = "af_heart", speed: float = 1.0, settings=None, lang: str | None = None):
        """kokoro: a kokoro_onnx.Kokoro. settings: None (the defaults), a dict of overrides or a path to a JSON file.
        lang: espeak language; default en-gb for British voices (b*), en-us otherwise."""
        self.kokoro, self.voice, self.speed = kokoro, voice, speed
        self.cfg = load(settings)
        self.lang = lang or ("en-gb" if voice.startswith("b") else "en-us")
        self._dog = kokoro.tokenizer.phonemize("dog", self.lang).strip()

    def normalize(self, text: str) -> str:
        return normalize(text, self.cfg)

    def render(self, sent: str):
        """One sentence through Kokoro. Words in settings "phonemes" get those exact phonemes; the text before such a word is
        phonemized as if a word followed it ("there's a" alone is read "there's AY")."""
        ph = get(self.cfg, "phonemes") or {}
        keys = [k for k in ph if re.search(r"(?<!\w)" + re.escape(k) + r"(?!\w)", sent)]
        if not keys:
            return self.kokoro.create(sent, voice=self.voice, speed=self.speed, lang=self.lang)
        phon = lambda t: self.kokoro.tokenizer.phonemize(t, self.lang).strip()
        out, pos = [], 0
        for m in re.finditer(r"(?<!\w)(" + "|".join(map(re.escape, sorted(keys, key=len, reverse=True))) + r")(?!\w)", sent):
            if sent[pos:m.start()].strip():
                pre = phon(sent[pos:m.start()].rstrip() + " dog")
                out.append(pre[:-len(self._dog)].rstrip() if pre.endswith(self._dog) else phon(sent[pos:m.start()]))
            out.append(ph[m.group(1)])
            pos = m.end()
        if sent[pos:].strip():
            out.append(phon(sent[pos:]))
        joined = re.sub(r"\s+([.,!?;:])", r"\1", " ".join(x for x in out if x))
        return self.kokoro.create(joined, voice=self.voice, speed=self.speed, lang=self.lang, is_phonemes=True)

    def stream(self, text: str):
        """Yield (audio, sample_rate) piece by piece: play the first sentence while the rest renders."""
        yield from pieces(self.normalize(text), self.render, self.cfg)

    def speak(self, text: str):
        """The whole reply as one (audio float32, sample_rate)."""
        out = list(self.stream(text))
        return np.concatenate([a for a, _ in out]), out[-1][1]

    def save(self, text: str, path: str):
        import soundfile as sf
        a, sr = self.speak(text)
        sf.write(path, a, sr)
        return path


load_settings = load
__all__ = ["Speaker", "normalize", "clean_markdown", "load_settings", "pieces"]
