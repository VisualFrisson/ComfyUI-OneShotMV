"""Theme / vibe: one free-text line that sets the world of the whole video.

`expand(theme)` asks a local language model for a small "world bible" of the video: who the
characters are, what they wear, where they are, the light, what they do in wide shots and a
colour grade, as lists. Casting (casting.apply_theme) then combines those lists per performer
the way a pack does, so a 60-shot run never repeats a look and every shot stays in the world.

The model is Gemma 3 12B, the LTX-2.3 text encoder every Frisson Vision install already has
(run through ComfyUI's own Generate Text), so a theme costs about a minute and no download.
Without it the theme falls back to templates built around the theme sentence.
Results are cached per theme + seed in output/frissonvision/themes/."""
from __future__ import annotations

import hashlib
import json
import os
import random
import re

# Lab mean / std targets for the finish (see fv/finish.py) plus saturation / contrast tweaks.
GRADES = {
    "neutral": {"grade_mean": [30.0, 1.0, 3.0], "grade_std": [20.0, 9.0, 11.0], "saturation": 1.05},
    "warm": {"grade_mean": [31.0, 6.0, 14.0], "grade_std": [20.0, 10.0, 13.0], "saturation": 1.1},
    "golden": {"grade_mean": [33.0, 5.0, 20.0], "grade_std": [21.0, 9.0, 14.0], "saturation": 1.1},
    "cool": {"grade_mean": [29.0, -2.0, -8.0], "grade_std": [20.0, 8.0, 11.0], "saturation": 1.0},
    "blue night": {"grade_mean": [24.0, 0.0, -16.0], "grade_std": [18.0, 8.0, 13.0], "saturation": 1.0},
    "neon": {"grade_mean": [27.0, 10.0, -10.0], "grade_std": [20.0, 16.0, 18.0], "saturation": 1.25},
    "magenta": {"grade_mean": [28.0, 14.0, -6.0], "grade_std": [19.0, 14.0, 13.0], "saturation": 1.15},
    "red": {"grade_mean": [27.0, 16.0, 9.0], "grade_std": [19.0, 14.0, 12.0], "saturation": 1.15},
    "green": {"grade_mean": [29.0, -10.0, 8.0], "grade_std": [19.0, 11.0, 12.0], "saturation": 1.05},
    "pastel": {"grade_mean": [40.0, 4.0, 2.0], "grade_std": [16.0, 8.0, 9.0], "saturation": 0.85,
               "contrast": 0.2, "lift": 0.06},
    "black and white": {"grade_mean": [30.0, 0.0, 0.0], "grade_std": [22.0, 0.5, 0.5], "saturation": 0.0},
    "faded film": {"grade_mean": [33.0, 3.0, 6.0], "grade_std": [16.0, 7.0, 9.0], "saturation": 0.85,
                   "lift": 0.06},
}

COUNTS = {"who": 14, "wear": 12, "places": 12, "light": 8, "actions": 10, "poses": 10}
OPTIONAL = {"poses"}  # bibles from before poses existed (and the template fallback) have none
N_CHARACTERS = 14

# Characters come whole (who + outfit + place), so a role reads at a glance and an outfit always
# fits the person wearing it. The sample runs of Oct 24 picked who and wear separately: men in lace
# corsets (Y2K Pop Star) and night-shift workers you couldn't place (Night Shift). The prompt also
# carries no example phrases any more: Gemma copied the old ones into every bible (the same
# silver-beehive woman and the same two poses in all three sample casts).
PROMPT = """You design the look of a music video. Theme: "{theme}"
Invent the world of this video. If the theme names jobs, roles, eras or kinds of people, every character is one of them and recognisable at a glance: their outfit, props and place say what they are. Go for specific, era-perfect detail, as if the stylist, the hair and make-up team and the set designer cared about every piece, in the tone the theme asks for: playful and over the top when it is fun, natural and grounded when it says straight, cinematic or serious. Everything must be physically real, things that existed and could be photographed. Never use brand or product names, never real or famous people, nobody holds a microphone, phone or instrument, and nothing carries writing: no signs, posters, banners, menus, screens or scoreboards. Answer with one JSON object only, no other text, with these keys:
"characters": a list of {characters} different characters, each an object with four keys. "who": the role or type first when the theme has them (every role the theme lists comes first, each once, before you invent any of your own, and invented ones are just as specific to this world: never a generic "customer", "passenger", "resident" or "friend"), then age, gender, skin tone, hairdo and one distinctive detail; at most 24 words; all adults, 18 or older; no names. "wear": the outfit that makes this character's role or style obvious, specific in fabric, colour and cut, cut for this character's gender, plain with no text, badges with words or logos; at most 20 words. "place": where this character belongs, starting with "in" or "on"; at most 14 words. "light": the real light source in that place, one you could point at, and its colour; at most 10 words. When the theme names places or a time of day, most characters are in those places at that time. Vary everything across the list: roles, ages within what the theme describes, genders, skin tones, builds, hairdos and accessories; not everyone wears glasses, hats or the same jewellery, and no two outfits are alike.
"places": a list of {places} more specific places in this world, each starting with "in" or "on". At most 16 words each.
"light": a list of {light} lighting setups, each a real source you could point at in this world (window daylight, the sun, a lamp, overhead tubes, a camera's own flash, studio softboxes, streetlights), said plainly with its colour; no coloured gels, washes or backdrops, no mood words, never photographers, crowds or other people. At most 14 words each.
"camera": one plain sentence of at most 22 words starting with "Shot on": the kind of camera, film or video this world would have been photographed with in its own time, and the colour, grain and flash it gives; no brand or model names, no mood words. For a present-day world: an unretouched modern digital photograph.
"actions": a list of {actions} things a person does in this world's places while singing, seen in a wide shot, each starting with a verb ending in -ing, never naming a role or another person, and never generic like "working on a task". They are alone and their hands are free. At most 9 words each.
"poses": a list of {poses} different expressions for a portrait in this world, in the theme's tone, describing the face only, never naming a role or another person. At most 12 words each.
"grade": the colour grade that fits the theme best, one of: {grades}."""

# phrases the old prompt showed as examples; Gemma copied them verbatim into bibles cached before v4
COPIED = {"a woman in her sixties with deep brown skin, a towering silver beehive and cat-eye glasses on a beaded chain",
          "chin resting on a fist with a huge cheesy grin", "pointing at the lens with total swagger"}


def _adults(s: str) -> str:
    """Every character is an adult: 'in her late teens' (Night Shift sample, Oct 24) -> 'in her early twenties'."""
    s = re.sub(r"(?i)\b(in (?:his|her|their)) (?:late |early |mid-?)?teens\b", r"\1 early twenties", s)
    s = re.sub(r"(?i)\bteenage (girl|boy|kid)\b", lambda m: "young " + {"girl": "woman", "boy": "man"}.get(
        m.group(1).lower(), "adult"), s)
    s = re.sub(r"(?i)\bteen(?:ager|age)?s?\b", "young adult", s)
    # "girl" and "boy" on their own (not "boy-band", "girl-group") draw children
    s = re.sub(r"(?i)(?<![\w-])girls(?![\w-])", "women", s)
    s = re.sub(r"(?i)(?<![\w-])girl(?![\w-])", "woman", s)
    s = re.sub(r"(?i)(?<![\w-])boys(?![\w-])", "men", s)
    s = re.sub(r"(?i)(?<![\w-])boy(?![\w-])", "man", s)
    s = re.sub(r"(?i)\b(kids?|child(?:ren)?|minors?|underage|schoolgirls?|schoolboys?)\b", "adults", s)
    return re.sub(r"(?i)\b(a|an) young adult (woman|man)\b", r"a young \2", s)


# things that show text (a tablet, a magazine, record sleeves: stress test st3, st4, st8) count as props too
_PROPS = re.compile(r"(?i)\b(tablets?|laptops?|monitors?|screens?|magazines?|newspapers?|books?|notebooks?|sketchbooks?|"
                    r"vinyl(?: records?)?|records|menus?|clipboards?|maps?|"
                    r"guitars?|drums?|drumsticks|keyboards?|keytars?|pianos?|basses|bass guitar|violins?|"
                    r"saxophones?|trumpets?|instruments?|microphones?|mics?|mic stands?|headsets?|phones?|"
                    r"cell ?phones?|flip ?phones?|smartphones?|turntables?|"
                    # a singer's mouth must be free (st4's barber sipping a drink, st5's pipe)
                    r"sipping|drinking|eating|chewing|smoking|pipes?|cigarettes?|cigars?|straws?)\b")


def _no_props(s: str) -> str:
    """Drop the clauses of a place or action that put an instrument, a microphone or a phone in
    someone's hands (Y2K retest: 'strumming a guitar', 'slamming a drum')."""
    if not _PROPS.search(s):
        return s
    parts = [p for p in re.split(r",\s*", s) if not _PROPS.search(p)]
    return ", ".join(parts).strip(" ,")


# Light and places that bring other people into a one-person shot. Y2K v3 (Oct 24): the bible's
# "intense flashes of white light mimicking paparazzi cameras" put photographers behind three of
# the singers' portraits.
_CROWD = re.compile(r"(?i)\b(paparazzi|photographers?|flashbulbs?|flashing cameras|press pit|crowds?|audiences?|"
                    r"fans|onlookers|spectators|bystanders|passers-?by|backing dancers|dancers|backing band|"
                    r"performers?|players|people|patrons|customers|guests|partygoers|bustling|crowded|packed)\b")
# a character doing something to someone ("a barber ..., trimming hair" in Night Shift v3) brings
# the customer into a one-person portrait
_NO_OTHERS = re.compile(r"(?i),\s*(?:trimming|cutting|styling|shaving|serving|helping|checking on|tending to|"
                        r"talking to|chatting with|waiting on|greeting|handing|pouring for)\b[^,]*")
_TWO = re.compile(r"(?i)\b(couple|pair|duo|twins|and (?:his|her|their) (?:wife|husband|partner|friend|kid|son|daughter)|"
                  r"strollers?|prams?|babies|baby|toddlers?|infants?|with (?:a|her|his) (?:dog|child|kid))\b|\band (?:fe)?male\b")
_LIGHT_SPARE = ["soft daylight from a large window to one side", "warm light from a lamp in the room",
                "bright, even light from large softboxes, like a magazine studio shoot",
                "hard light from a single bare bulb overhead", "the camera's own flash, with dark falloff behind"]
# Coloured washes and gels read as AI, not as a photograph (the user on Y2K v3: "weird unnatural
# blue pink lighting", where the singers lit by plain window or lamp light looked real). A light
# has to be a source you could point at; mood clauses ("creating a dreamy atmosphere", "casting a
# nostalgic glow", "simulating a nighttime sky") go.
_WASH = re.compile(r"(?i)\b(gels?|wash(?:es|ed)?|backdrop|hues|swirl(?:s|ing)?|dreamy|accents|bathed|haze of|"
                   r"reflect(?:s|ing|ions?|ed)?)\b")  # reflections: Y2K v3's chrome light gave a mirror twin
_SOURCE = re.compile(r"(?i)\b(windows?|sun|sunlight|daylight|golden hour|moonlight|lamps?|bulbs?|tubes?|fluorescents?|"
                     r"neon|flash|softbox(?:es)?|street ?lights?|spotlights?|spots|candles?|fire|firelight|screens?|"
                     r"headlights?|fixtures?|lanterns?|torch(?:es)?|chandeliers?|curtains?|skylights?|overhead|"
                     r"ring light|work light|strobes?|oven|console|monitors?)\b")
_MOOD = re.compile(r"(?i),?\s+\b(creating|casting|simulating|mimicking|highlighting|evoking|giving|adding|"
                   r"lending|bathing|illuminating)\b.*$")


def _real_light(v: list[str]) -> list[str]:
    """Lights from real sources only, said plainly; topped up from plain spares."""
    out = []
    for x in v:
        x = _MOOD.sub("", x).strip(" ,")
        if len(x) > 8 and _SOURCE.search(x) and not _WASH.search(x) and x not in out:
            out.append(x)
    return out


def _alone(v: list[str], spare: list[str] | None = None, n_min: int = 3) -> list[str]:
    """Drop the lights or places that put other people in the picture; top up from spares."""
    keep = [x for x in v if not _CROWD.search(x)]
    if spare is not None:
        keep = _real_light(keep)
    for s in spare or []:
        if len(keep) >= n_min:
            break
        if s not in keep:
            keep.append(s)
    return keep or v


_CAM_BRANDS = re.compile(r"(?i)\b(kodak|fuji(?:film)?|ilford|agfa|canon|nikon|leica|olympus|minolta|pentax|"
                         r"panasonic|jvc|sony|handycam|contax|yashica|ricoh|konica|lomo(?:graphy)?|polaroid)\b\s*"
                         r"(?:(?:gold|portra|ektar|ektachrome|kodachrome|superia|velvia|tri-x|hp5|vista|max|"
                         r"ultramax|colorplus|instax|mju|stylus|t4|t2|g2|elph|sure ?shot)\b\s*)?(?:\d{2,4}\b\s*)?")


def _camera(v) -> str:
    """The world's period camera line, without brand or model names."""
    if not isinstance(v, str):
        return ""
    v = _CAM_BRANDS.sub("", " ".join(v.split())).strip(" .,")
    # "The video was captured using X, utilizing Y for a nostalgic feel, resulting in a grainy
    # aesthetic" (the first v4 bible) -> "Shot on X, Y, a grainy look"
    v = re.sub(r"(?i)^(?:the\s+)?(?:video|footage|photos?|photographs?|images?|pictures?|world|scenes?)\s+"
               r"(?:was|were|is|are|would be|would have been)\s+(?:captured|shot|taken|filmed|recorded|photographed)"
               r"\s+(?:using|with|on)\s+", "Shot on ", v)
    v = re.sub(r"(?i),?\s*\b(?:for|giving|lending|creating|evoking)\s+(?:it\s+)?an?\s+[a-z -]*?"
               r"(?:feel|vibe|atmosphere|mood|look)\b", "", v)
    v = re.sub(r"(?i)\butili[sz]ing\b", "on", v)
    v = re.sub(r"(?i),?\s*resulting in\b", ",", v)
    v = re.sub(r"(?i)\baesthetic\b", "look", v)
    v = re.sub(r"\s+,", ",", re.sub(r",\s*,", ",", v)).strip(" .,")
    v = re.sub(r"\b(a|an)\s+(an?)\b", r"\2", v)
    v = re.sub(r"\s+([,.])", r"\1", v)
    if len(v) < 12 or _CROWD.search(v) or re.search(r"(?i)\b(modern|contemporary|present-day|smartphone)\b", v):
        return ""  # present day: the brief's own modern look
    return v


def _role_words(chars: list) -> set:
    """The role nouns of a world's characters ('waitress', 'go-go dancer', 'mother')."""
    out = set()
    for c in chars or []:
        who = str(c.get("who", "")) if isinstance(c, dict) else str(c)
        role = re.sub(r"^(?:an?|the)\s+", "", who.split(",")[0].strip(), flags=re.I).lower()
        if role and not re.fullmatch(r"(?:young |older |old )?(?:wo)?man|person|lady|guy", role):
            out.add(role)
            out.add(role.split()[-1])  # 'dancer' for 'go-go dancer'
    return {r for r in out if len(r) > 3}


def _no_others(v: list, roles: set) -> list:
    """Drop actions and poses that name a character (Pink Pony Club st2b: 'The waitress refills a
    coffee cup' in the mother's portrait drew the waitress next to her, seven portraits with two
    people), or start with 'The ...'; keep the list if that would empty it."""
    pat = re.compile(r"(?i)\b(" + "|".join(re.escape(r) for r in sorted(roles, key=len, reverse=True)) + r")s?\b") \
        if roles else None
    keep = [x for x in v if not re.match(r"(?i)^the\s", x.strip()) and not (pat and pat.search(x))]
    return keep if len(keep) >= 3 else [re.sub(r"(?i)^the\s+[a-z -]+?\s+(?=[a-z]+s\b)", "", x) for x in v]


def _plain(wear: str) -> str:
    """A 'graphic tee' or 'band tee' comes out with garbled print on it."""
    wear = re.sub(r"(?i)\b(graphic|band|logo|printed|slogan)\s+(t-?shirt|tee)\b", r"plain \2", wear)
    # caps come out with a real team or brand logo on the front (v4c: a league logo on a cap)
    if "no logo" not in wear.lower():
        wear = re.sub(r"(?i)\b(baseball caps?|trucker hats?|snapbacks?|ball caps?)\b", r"\1 with no logo", wear, count=1)
    return wear


def _copied(x: str) -> bool:
    return any(c in x.lower() for c in COPIED)


def _characters(v) -> list[dict]:
    """[{who, wear, place}] from the model's 'characters', tolerating strings and odd keys."""
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        if not isinstance(x, dict):
            continue
        low = {str(k).strip().lower(): str(val) for k, val in x.items()}
        who = low.get("who") or low.get("character") or low.get("person") or ""
        wear = low.get("wear") or low.get("outfit") or low.get("clothes") or ""
        place = low.get("place") or low.get("where") or low.get("location") or ""
        light = " ".join((low.get("light") or low.get("lighting") or "").split()).strip(" ,.")
        light = (_real_light([light]) or [""])[0] if light and not _CROWD.search(light) else ""
        who = _adults(_debrand(_NAME.sub("", " ".join(who.split()).strip(" ,.-"))))
        who = _no_props(_NO_OTHERS.sub("", who).strip(" ,"))
        if _TWO.search(who):
            continue  # a couple, or someone with a baby, is two people in a one-person portrait (st6)
        wear = _plain(_debrand(" ".join(wear.split()).strip(" ,.")))
        if re.match(r"^(?:[A-Z][a-z]|A\s)", wear):
            wear = wear[0].lower() + wear[1:]  # "Pale green scrubs" reads "Wearing pale green scrubs"
        place = _no_props(_debrand(" ".join(place.split()).strip(" ,.")))
        # "in the white infinity studio, posing for photos": the place only (actions come separately)
        cut = re.sub(r",\s*[a-z]+ing\b.*$", "", place)
        # "on the floor of a vast, humming factory" is an adjective, not an action
        place = place if re.search(r"(?i)\b(a|an|the|of|in|on|at|vast|huge|large|small|big|old|busy|dark|quiet|empty)$", cut) else cut
        if _CROWD.search(place):
            place = ""  # the cast then takes a place from the world's list
        if len(who) < 4 or len(wear) < 4 or _copied(who):
            continue
        if _WORDS.search(wear) and not _NO_WORDS.search(wear):
            continue  # lettering on clothes comes out as garbled text
        out.append({"who": who, "wear": wear, "place": place, **({"light": light} if light else {})})
    return out


def cache_dir(out_root: str) -> str:
    return os.path.join(out_root, "frissonvision", "themes")


def _key(theme: str, seed: int) -> str:
    return hashlib.sha1(f"{theme.strip()}|{seed}|v5".encode("utf-8")).hexdigest()[:16]  # v5: camera + character lights


# ------------------------------------------------------------------ the language model
def _call(cls, **kw):
    """Run a ComfyUI node class in-process (old-style FUNCTION nodes and new io.ComfyNode)."""
    if hasattr(cls, "FUNCTION"):
        r = getattr(cls(), cls.FUNCTION)(**kw)
    else:
        r = cls.execute(**kw)
    if hasattr(r, "args"):
        return r.args
    if isinstance(r, dict) and "result" in r:
        return r["result"]
    return r


def llm(prompt: str, seed: int = 1, max_length: int = 1600, log=print) -> str | None:
    """Gemma 3 12B (the LTX-2.3 text encoder) through ComfyUI's Generate Text; None when either
    is missing (outside ComfyUI, or an older ComfyUI)."""
    try:
        import nodes as comfy_nodes  # type: ignore
        from . import graphs
    except Exception:
        return None
    loader = comfy_nodes.NODE_CLASS_MAPPINGS.get("LTXAVTextEncoderLoader")
    gen = comfy_nodes.NODE_CLASS_MAPPINGS.get("TextGenerate")
    if loader is None or gen is None:
        return None
    try:
        clip = _call(loader, text_encoder=graphs.M["ltx23_te"], ckpt_name=graphs.M["ltx23_ckpt"],
                     device="default")[0]
        out = _call(gen, clip=clip, prompt=prompt, max_length=max_length,
                    sampling_mode={"sampling_mode": "on", "temperature": 0.8, "top_k": 64, "top_p": 0.95,
                                   "min_p": 0.05, "repetition_penalty": 1.05, "seed": int(seed)},
                    thinking=False, use_default_template=True)
        return str(out[0])
    except Exception as e:  # never let the theme stop a run
        log(f"[FrissonVision] theme: the language model failed ({e}); using templates")
        return None


def _parse(text: str) -> dict | None:
    """The model's JSON object, tolerating code fences, trailing commas and missing commas
    between lines."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return None
    t = t[a:b + 1]
    for fix in (lambda s: s, lambda s: re.sub(r",\s*([\]}])", r"\1", s),
                lambda s: re.sub(r'"\s*\n\s*"', '",\n"', re.sub(r",\s*([\]}])", r"\1", s)),
                lambda s: re.sub(r'(["\]])\s*\n\s*"', r'\1,\n"', re.sub(r",\s*([\]}])", r"\1", s))):
        try:
            d = json.loads(fix(t))
            if isinstance(d, dict):
                return d
        except Exception:
            continue
    return None


_NAME = re.compile(r"^[A-Z][a-z]+(?: [A-Z][a-z]+)+\s*:\s*")  # "Elara Vance: 23, ..." -> "23, ..."
_WORDS = re.compile(r"\b(logos?|lettering|slogans?|name ?tags?|insignia|printed words|text)\b", re.I)
_NO_WORDS = re.compile(r"\b(no|without)\b[^,.]*\b(logos?|lettering|slogans?|text)\b", re.I)


# brand names the model slips in anyway ("oversized Versace sunglasses", n46): a generic word instead,
# so the image models don't draw a logo
BRANDS = re.compile(r"\b(versace|gucci|chanel|prada|louis vuitton|vuitton|dior|fendi|hermes|hermès|burberry|"
                    r"cartier|rolex|ray-?ban|oakley|nike|adidas|reebok|puma|converse|vans|levi'?s|calvin klein|"
                    r"tommy hilfiger|ralph lauren|lacoste|fila|champion|supreme|coca-?cola|pepsi|mcdonald'?s|"
                    r"cadillac|rolls-?royce|ferrari|lamborghini|porsche|mercedes(-benz)?|bmw|polaroid|walkman|"
                    r"sony|gameboy|game boy|nintendo|casio|swatch|members only|timberlands?|dr\.? martens|doc martens|"
                    r"skechers|juicy couture|von dutch|fubu|tommy|abercrombie|hollister|ugg|crocs|heelys)\b", re.I)


_SIGNS = "signs?|banners?|posters?|billboards?|marquees?|placards?|menu boards?"


def _design(s: str) -> str:
    """Signs come out as garbled lettering (n45's "neon signs reflecting off the floor", n46's
    "neon pink and blue signs advertising early-bird specials" and "a bench adorned with
    hand-painted signs"): plain light or plain decoration instead, and never what they say."""
    s = re.sub(rf"(?i)\b({_SIGNS})\s+(advertising|reading|saying|announcing|promoting|offering|for|that say)\b[^,.;:]*",
               r"\1", s)
    s = re.sub(r"(?i)\b(neon(?:\s+[a-z-]+){0,4}?)\s+signs\b(?!-)", r"\1 tube lights", s)
    s = re.sub(r"(?i)\b(neon(?:\s+[a-z-]+){0,4}?)\s+sign\b(?!-)", r"\1 tube light", s)
    s = re.sub(r"(?i)\b(peace|v|victory|thumbs-up|ok|rock|horns|hand|heart)\s+(sign)s?\b", r"\1 gesture", s)
    s = re.sub(rf"(?i)\b(?:hand-painted|painted|handwritten|hand-lettered|wooden|vintage|faded|old)?\s*"
               rf"(?<!gesture )(?:{_SIGNS})\b(?!-)", " pastel bunting", s)
    return re.sub(r"  +", " ", s).strip()


def _debrand(s: str) -> str:
    s = _design(BRANDS.sub("designer", s))
    s = re.sub(r"\bdesigner(\s+designer)+\b", "designer", s)
    # a brand that was the whole item ("fishnet stockings, Doc Martens" in v4b) leaves a bare word
    return re.sub(r"(?:,\s*|\s+(?:and|with)\s+)designer(?=\s*(?:[,;.]|$))", "", s).strip(" ,")


def _clean_list(v, n_min: int = 3) -> list[str]:
    if isinstance(v, str):
        v = [x for x in re.split(r"[\n;]", v) if x.strip()]
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        if isinstance(x, dict):
            x = ", ".join(str(y) for y in x.values())
        x = _debrand(_NAME.sub("", str(x).strip().strip("-•* ").rstrip(".")))
        x = " ".join(x.split()).strip(" ,")
        if _WORDS.search(x) and not _NO_WORDS.search(x):
            continue  # lettering on clothes and walls comes out as garbled text
        if len(x) > 3 and x not in out and not _copied(x):
            out.append(x)
    return out if len(out) >= n_min else []


_DECADE = r"(\d0)s|(twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)"
_GENDER = {"female": ("woman", "her"), "woman": ("woman", "her"), "male": ("man", "his"), "man": ("man", "his"),
           "non-binary": ("person", "their"), "nonbinary": ("person", "their"), "person": ("person", "their")}


def _article(lead: str) -> str:
    if lead[:1].isupper() and len(lead) > 1 and not lead[1].islower():  # "R&B", "DJ"
        return "an" if lead[0] in "AEFHILMNORSX" else "a"
    return "an" if lead[:1].lower() in "aeiou" else "a"


def _role(role: str) -> str:
    """'Firefighter' -> 'firefighter', 'court Jester' -> 'court jester'; 'R&B diva' and 'DJ' stay."""
    words = []
    for w in role.strip().split(" "):
        parts = w.split("-")
        parts = [x if len(x) > 1 and not x[1:].islower() else x[:1].lower() + x[1:] for x in parts]
        words.append("-".join(parts))
    return " ".join(words)


_BARE = re.compile(r"(?i)^(?:an?\s+)?(?:(?:young|older|old|middle-aged|elderly)\s+)?(woman|man|person|lady|gentleman|guy)$")


def _role_first(s: str) -> str:
    """'a Firefighter, 20s, female, light skin, ...' (how Gemma writes whole characters) ->
    'a firefighter, a woman in her twenties, light skin, ...'; 'a Boy-band member, 19, male, ...' ->
    'a boy-band member, a 21-year-old man, ...'. The role stays first so it reads at a glance, and
    an adult noun (man, woman) is always there: role words like "princess" or "boy-band" alone let
    the image model draw someone too young (the user's note of Oct 24: none of them may look like
    kids). Ages under 21 become 21, a margin above the adults rule."""
    s = re.sub(r",\s*(\d)\d\s*s\s*,", lambda mm: f", {mm.group(1)}0s,", s)  # "45s" (st1's blacksmith) -> "40s"
    m = re.match(r"^(?:an?\s+)?([A-Za-z][A-Za-z'&/ -]{1,40}?),\s*(\d{2})\s*,\s*"
                 r"(female|male|woman|man|non-?binary|person)\b,?\s*(.*)$", s.strip(), flags=re.I)
    if m:
        age = max(21, int(m.group(2)))
        noun = _GENDER.get(m.group(3).lower(), ("person", "their"))[0]
        role, rest = _role(m.group(1)), m.group(4)
        adult = f"a {age}-year-old {'adult' if noun == 'person' else noun}"
        if _BARE.match(role):
            return adult + (f", {rest}" if rest else "")
        return f"{_article(role)} {role}, {adult}" + (f", {rest}" if rest else "")
    m = re.match(rf"^(?:an?\s+)?([A-Za-z][A-Za-z'&/ -]{{1,40}}?),\s*(?:(?:early|mid|late)-?\s*)?(?:{_DECADE})\s*,\s*"
                 r"(female|male|woman|man|non-?binary|person)\b,?\s*(.*)$", s.strip(), flags=re.I)
    if not m:
        return s
    role, d1, d2, g, rest = _role(m.group(1)), m.group(2), m.group(3), m.group(4).lower(), m.group(5)
    words = {"2": "twenties", "3": "thirties", "4": "forties", "5": "fifties", "6": "sixties", "7": "seventies",
             "8": "eighties", "9": "nineties"}
    decade = words.get(d1[0], "twenties") if d1 else d2.lower()
    noun, pron = _GENDER.get(g, ("person", "their"))
    adult = f"{'an adult' if noun == 'person' else 'a ' + noun} in {pron} {decade}"
    if _BARE.match(role):
        return adult + (f", {rest}" if rest else "")
    return f"{_article(role)} {role}, {adult}" + (f", {rest}" if rest else "")


def _people(who: list[str]) -> list[str]:
    """'23, female, dark skin, afro, ...' reads badly in a prompt: make it 'a 23-year-old woman ...'."""
    out = []
    for w in who:
        s = _role_first(w)
        m = re.match(r"^(\d{2})\s*(?:-?year-old|yo|years old)?\s*,?\s*(female|male|woman|man|non-?binary|person)\b,?\s*(.*)$",
                     s, flags=re.I)
        if m:
            g = {"female": "woman", "male": "man", "nonbinary": "person", "non-binary": "person"}.get(
                m.group(2).lower(), m.group(2).lower())
            s = f"a {m.group(1)}-year-old {g}" + (f", {m.group(3)}" if m.group(3) else "")
        elif not re.match(r"^(a|an|the)\b", s, flags=re.I):
            s = "a " + s
        # "a woman in her sixty-three" (n46's bible) -> "a woman aged sixty-three"
        s = re.sub(r"(?i)\bin (?:her|his|their) ((?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-"
                   r"(?:one|two|three|four|five|six|seven|eight|nine)|\d{2})\b", r"aged \1", s)
        out.append(s)
    return out


def _places(pl: list[str]) -> list[str]:
    _PREP = (r"behind|outside|around|along|beneath|underneath|across|through|over|onto|against|next to|in front of|"
             r"inside|under|beside|by|near|atop|among|at|on")
    pl = [re.sub(rf"(?i)^in\s+(?=(?:{_PREP})\b)", "", p.strip()) for p in pl]  # "in behind the wheel"
    return [p if re.match(r"^(in|on|at|inside|under|underneath|beside|by|near|atop|among|behind|outside|around|along|beneath|"
                          r"across|through|over|onto|against|next to|in front of)\b", p, flags=re.I) else "in " + p
            for p in pl]


def fallback(theme: str) -> dict:
    """No language model: the theme sentence itself carries the world."""
    t = theme.strip().rstrip(".")
    return {
        "theme": theme, "source": "templates",
        "who": [f"{a} who belongs in {t}" for a in (
            "a young woman", "a young man", "a woman in her thirties", "a man in his forties", "an older woman",
            "an older man", "a woman in her early twenties", "a man in his early twenties", "a woman in her fifties",
            "a man in his twenties")],
        "wear": [f"{a} styled for {t}" for a in (
            "a long coat", "a fitted jacket", "layered streetwear", "a sharp suit", "a flowing dress",
            "practical work clothes", "a bold statement outfit", "a knitted sweater")],
        "places": [f"{a} {t}" for a in (
            "in the heart of", "at the edge of", "in a quiet corner of", "in a narrow passage of",
            "on a high vantage point over", "in an intimate interior of", "in a wide open space of")],
        "light": [f"{a} in the colours of {t}" for a in (
            "soft side light", "hard top light", "glowing practical lights", "backlight with a rim",
            "moody low-key light")],
        "actions": ["walking slowly toward the camera", "leaning against a wall", "sitting and swaying",
                    "walking past the camera", "dancing loosely", "turning toward the camera"],
        "grade": "neutral",
    }


def expand(theme: str, *, seed: int = 1, out_root: str | None = None, log=print) -> dict:
    """The world bible for a theme (cached per theme + seed)."""
    theme = (theme or "").strip()
    if not theme:
        return {}
    path = None
    if out_root:
        path = os.path.join(cache_dir(out_root), _key(theme, seed) + ".json")
        if os.path.exists(path):
            try:
                cached = json.load(open(path, encoding="utf-8"))
                for k in COUNTS:  # bibles cached before the brand and sign scrubs
                    if isinstance(cached.get(k), list):
                        cached[k] = [_debrand(x) for x in cached[k]]
                if isinstance(cached.get("who"), list):
                    cached["who"] = _people([_adults(w) for w in cached["who"] if not _copied(w)])
                if isinstance(cached.get("poses"), list):
                    cached["poses"] = [x for x in cached["poses"] if not _copied(x)]
                for k in ("actions", "places", "poses"):
                    if isinstance(cached.get(k), list):
                        cached[k] = [y for y in (_no_props(x) for x in cached[k]) if len(y) > 3]
                roles = _role_words(cached.get("characters") or [])
                for k in ("actions", "poses"):
                    if isinstance(cached.get(k), list):
                        cached[k] = _no_others(cached[k], roles)
                if isinstance(cached.get("wear"), list):
                    cached["wear"] = [_plain(x) for x in cached["wear"]]
                if isinstance(cached.get("light"), list):
                    cached["light"] = _alone(cached["light"], _LIGHT_SPARE, n_min=4)
                if isinstance(cached.get("places"), list):
                    cached["places"] = _alone(cached["places"])
                if isinstance(cached.get("characters"), list):
                    cached["characters"] = _characters(cached["characters"])
                    for c in cached["characters"]:
                        c["who"] = _people([c["who"]])[0]
                        c["place"] = _places([c["place"]])[0] if c["place"] else ""
                return cached
            except Exception:
                pass
    prompt = PROMPT.format(theme=theme.replace('"', "'"), grades=", ".join(GRADES), characters=N_CHARACTERS,
                           **COUNTS)
    bible = None
    for attempt in range(2):
        text = llm(prompt, seed=seed + attempt, max_length=3200, log=log)  # whole characters run long
        if text is None:
            break
        d = _parse(text)
        if not d:
            log("[FrissonVision] theme: could not read the model's answer, trying again")
            continue
        lists = {k: _clean_list(d.get(k)) for k in COUNTS}
        for k in ("actions", "places", "poses"):
            lists[k] = [y for y in (_no_props(x) for x in lists[k]) if len(y) > 3]
        lists["wear"] = [_plain(x) for x in lists["wear"]]
        roles = _role_words(_characters(d.get("characters")))
        for k in ("actions", "poses"):
            lists[k] = _no_others(lists[k], roles)
        lists["light"] = _alone(lists["light"], _LIGHT_SPARE, n_min=4)
        lists["places"] = _alone(lists["places"])
        chars = _characters(d.get("characters"))
        if len(chars) >= 6:  # who and wear come from the whole characters
            lists["who"] = [c["who"] for c in chars]
            lists["wear"] = [c["wear"] for c in chars]
            lists["places"] = lists["places"] or [c["place"] for c in chars if c["place"]]
        if all(v for k, v in lists.items() if k not in OPTIONAL):
            g = str(d.get("grade", "neutral")).strip().lower()
            bible = {"theme": theme, "source": "gemma-3-12b", **lists,
                     "grade": g if g in GRADES else "neutral", "camera": _camera(d.get("camera")), "raw": text}
            if len(chars) >= 6:
                bible["characters"] = chars
            break
        log("[FrissonVision] theme: the model's answer was missing lists, trying again")
    if bible is None:
        bible = fallback(theme)
    bible["who"] = _people([_adults(w) for w in bible["who"]])
    bible["places"] = _places(bible["places"])
    for c in bible.get("characters") or []:
        c["who"] = _people([c["who"]])[0]
        c["place"] = _places([c["place"]])[0] if c["place"] else ""
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bible, f, indent=1, ensure_ascii=False)
    return bible


def finish_for(bible: dict, finish: dict) -> dict:
    """The pack's finish with the theme's grade (the pack keeps its glow, grain and gate)."""
    g = GRADES.get((bible or {}).get("grade", ""), None)
    if not g:
        return finish
    f = dict(finish)
    f.update({k: v for k, v in g.items()})
    f["grade_amount"] = min(0.6, finish.get("grade_amount", 0.6))
    return f


def summary(bible: dict, n: int = 3) -> str:
    if not bible:
        return ""
    rng = random.Random(0)
    lines = [f"Theme: {bible['theme']} ({bible.get('source', '')}, grade {bible.get('grade', '')})"]
    for k in ("who", "wear", "places"):
        items = bible.get(k, [])
        lines.append(f"  {k}: " + " | ".join(rng.sample(items, min(n, len(items)))))
    return "\n".join(lines)
