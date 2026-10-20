"""Text-rule tests (no model needed): python test_kokoro_natural.py   (or pytest)"""
from kokoro_natural import load_settings, normalize

CASES = [
    ("It's $1,249.99", "It's one thousand two hundred forty-nine dollars and ninety-nine cents"),
    ("about $1.5k", "about fifteen hundred dollars"),
    ("$2.3M raised", "two point three million dollars raised"),
    ("$20/mo", "twenty dollars a month"),
    ("at 4:55 pm", "at four fifty five P M"),
    ("at 10:00", "at ten o'clock"),
    ("at 7:05 am", "at seven oh five eigh em"),
    ("Mon-Fri 9am-5pm", "Monday to Friday 9 eigh em to 5 P M"),
    ("on 12/24/2026", "on December twenty-fourth, twenty twenty-six"),
    ("on Dec 5", "on December fifth"),
    ("since 2019", "since twenty nineteen"),
    ("add 1/2 cup", "add half cup"),
    ("1 1/2 tsp salt", "1 and a half teaspoons salt"),
    ("call (212) 555-0123", "call [hold] two one two, five five five, oh one two three"),
    ("at 4804 N Main St.", "at forty-eight oh four North Main Street."),
    ("~12 mins away", "about 12 minutes away"),
    ("1 hrs", "1 hour"),
    ("up 34%", "up thirty-four percent"),
    ("the RTX 4090 at 1080p", "the RTX forty ninety at ten eighty p"),
    ("Python 3.13", "Python three point thirteen"),
    ("the Q4_K_M GGUF", "the Q4 K M G G U F"),
    ("72°F outside", "seventy-two degrees outside"),
    ("visit example.com", "visit example dot com"),
    ("we won 34-30", "we won 34 to 30"),
    ("we live here", "we liv here"),
    ("streaming live on TV", "streaming lyve on T V"),
    ("Ooh, nice.", "Nice."),
    ("Oh no.", "Oh no."),
    ("Nope, not today.", "Nope— not today."),
    ("Honestly? I'd wait.", "Honestly, I'd wait."),
    ("Well... maybe.", "Well. Maybe."),
    ("near the Airbnb", "near the Air B and B"),
    ("a Dr Pepper", "a Dr Pepper"),
    ("at 12 Main St. then left", "at 12 Main Street then left"),
    ("on Oak Blvd, then left", "on Oak Boulevard, then left"),
    ("I'm at 12 Elm Dr.", "I'm at 12 Elm Drive."),
    ("It's on Park Ave. Then left.", "It's on Park Avenue. Then left."),
    ("Take the Dallas Pkwy. It's faster.", "Take the Dallas Parkway. It's faster."),
    ("exit at Oak Blvd. then right", "exit at Oak Boulevard then right"),
    ("Merge onto Central Fwy.", "Merge onto Central Freeway."),
    ("It's on Legacy Wy.", "It's on Legacy Way."),
]


def test_cases():
    bad = [(t, normalize(t), want) for t, want in CASES if normalize(t) != want]
    assert not bad, "\n".join(f"{t!r}\n  got  {g!r}\n  want {w!r}" for t, g, w in bad)


def test_settings_merge():
    assert normalize("Meet at Nguyen's.", {"respell": {"Nguyen": "Win"}}) == "Meet at Win's."
    assert normalize("near the Airbnb", {"respell": {"Airbnb": None}}) == "near the Airbnb"
    assert normalize("at 4:55 pm", {"ampm_letters": False}) == "at four fifty five in the afternoon"
    cfg = load_settings({"respell": {"Nguyen": "Win"}})
    assert cfg["respell"]["Nguyen"] == "Win" and "Airbnb" in cfg["respell"]
    assert load_settings("my_settings.json")["head_pause"]["yep"] == 0.6


if __name__ == "__main__":
    test_cases(); test_settings_merge(); print(f"all {len(CASES)} cases + settings OK")
