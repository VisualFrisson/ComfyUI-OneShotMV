"""Theme / vibe: one free-text line that sets the world of the whole video.

`expand(theme)` asks a local language model for a small "world bible" of the video: who the
characters are, what they wear, where they are, the light, what they do in wide shots and a
colour grade, as lists. Casting (casting.apply_theme) then combines those lists per performer
the way a pack does, so a 60-shot run never repeats a look and every shot stays in the world.

The model is Gemma 3 12B, the LTX-2.3 text encoder every OneShot MV install already has
(run through ComfyUI's own Generate Text), so a theme costs about a minute and no download.
Without it the theme falls back to templates built around the theme sentence.
Results are cached per theme + seed in output/oneshot-mv/themes/."""
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

COUNTS = {"who": 12, "wear": 12, "places": 12, "light": 8, "actions": 10, "poses": 10}
OPTIONAL = {"poses"}  # bibles from before poses existed (and the template fallback) have none
# v7 bibles: a bigger cast, every shot its own character, each with a role, gender, age and look
# the vision check can hold the still to. The first call writes the world and 12 characters;
# extend() asks for more in batches until every shot has someone new.
N_CHARACTERS = 12
EXTRA_BATCH = 8  # batches of 12 come back cut off

# One character, the keys the model fills (shared by the world call and the extra-cast calls).
# Separate keys because, given one "who" field (v6 bibles), Gemma writes just the role ("a Hotel
# Valet"): no gender, age or face, so the image model draws its default man again and again (a
# radio announcer, a businessman and a bellhop with one face) and a "musician" can come out either way.
CHARACTER_KEYS = """each an object with ten keys. "role": the job, role or type, at most 5 words, no names: every role the theme lists comes first, each once, before you invent any of your own, and invented ones are just as specific to this world (never a generic "customer", "passenger", "resident" or "friend"). "gender": "woman" or "man"; about half the characters are women and half men, mixed through the list, except that a role only one gender held in this world's place and time keeps that gender. "age": a number from 18 to 80 that fits the role and the theme. "look": the theme's own words, with every detail, when the theme describes how this person looks; otherwise this person's own face and body, unlike anyone else's in the list: skin tone, hair colour and a hairdo of the period, a beard or moustache for some of the men only, build and one distinctive feature; at most 30 words. "wear": the theme's own words, with every detail, when the theme says what this person wears; otherwise exactly what this person wears for their role in this world, every piece right for its place and period, specific in fabric, colour and cut, cut for their gender; plain, with no writing, badges with words or logos; no glasses, sunglasses or goggles unless the role truly needs them and they existed then; at most 20 words. "place": where this character belongs, starting with "in" or "on"; at most 14 words. "light": the real light source in that place, one you could point at, and its colour; at most 10 words. "doing": what this character does in that place while singing, seen in a wide shot, starting with a verb ending in -ing, something their role would really do there, alone, never with or to another person; at most 9 words. "face": this character's expression in a close-up portrait, in the theme's tone, the face only, head up; at most 10 words."""
STRICT = ("Strict accuracy: every outfit, hairdo, prop and place is exactly right for this world and its "
          "period, as a costume historian would check it, and nothing comes from a later time.")

GENDER_SENTENCE = ('"gender": "woman" or "man"; about half the characters are women and half men, mixed through the '
                   "list, except that a role only one gender held in this world's place and time keeps that gender.")
assert GENDER_SENTENCE in CHARACTER_KEYS
STRICT_NOW = "Strict accuracy: every outfit, hairdo, prop and place is exactly what you would see in this world today."
STRICT_CREATURES = ("Strict accuracy: every creature, place and prop is exactly as this world describes it, and "
                    "everything could be photographed for real.")

# ------------------------------------------------------------------ what the World line itself says
# The defaults (about half women and half men, ages 18 to 80, period hair and clothes, people) only
# fill in what the World line leaves open; what it says about the cast wins. Without this rule,
# "all invented elemental creatures and no people" came back as 60 people with jobs ("a snowdrift
# sculptor, a man aged 62, weathered face, pale skin"), and "No men" as 7 women and 5 men.
BRIEF_FIRST = ("The theme is the brief and it wins: whatever it says about the characters (who or what they are, "
               "their kind, gender, age, period, looks, clothes and places) is a rule, and the instructions below "
               "only fill in what it leaves open. A character the theme describes keeps every detail the theme "
               "gives it, in the theme's own words.")
RULE_LINES = {"woman": "Every character is a woman: the theme says so.",
              "man": "Every character is a man: the theme says so.",
              "now": ("The theme is set today: every character, outfit, hairdo and place is from the present day, "
                      "with no costumes or looks from earlier decades."),
              "creatures": ("The characters are not people: every one is a creature as the theme describes it, never "
                            "a person, a person in costume or a person who works with that thing.")}
_NO_PEOPLE = re.compile(r"(?i)\b(?:no|without)\s+(?:other\s+)?(?:people|humans?|persons|human beings|men or women)\b|"
                        r"\bnon-?humans?\b|\bnot (?:a )?humans?\b")
_CREATURE_CAST = re.compile(r"(?i)\b(?:all|every|only|entirely)\b[^,.;:]{0,40}?\b(?:creatures?|animals?|beasts?|robots?|"
                            r"androids?|machines?|monsters?|elementals?|aliens?|dragons?|spirits?|golems?|puppets?|toys?)\b")
_W, _M = r"(?:women|woman|females?|ladies|lady|girls?)", r"(?:men|man|males?|guys?|gentlemen|boys?)"
# "only sexy females", "all grown men" (a word or two between); not "the only two women", "all day long the women"
_BETWEEN = (r"(?:(?!(?:one|two|three|four|five|six|a|few|of|day|night|long|year|week|around|over|along|at|in|with|"
            r"and|or)\b)[a-z][\w-]*\s+){{0,2}}?")
_ONLY = (r"\bno (?:{other})\b|(?<!the )\b(?:only|all)[- ](?:the )?" + _BETWEEN + r"(?:{one})\b(?!\s+(?:and|or|&)\s)|"
         r"(?<!and )(?<!or )(?<!& )\b(?:{one}) only\b|\ball[- ](?:{adj})\b(?!\s+(?:and|or|&)\s)|"
         r"\b(?:every|each) (?:single )?(?:one|character|singer|person|performer|member)\b[^.;]{{0,30}}?\bis an? "
         r"(?:[\w-]+\s+){{0,4}}?(?:{one})\b|"
         r"\b(?:all|the) (?:the )?(?:characters|singers|performers|cast|people)\b[^.;]{{0,20}}?\bare (?:all )?"
         r"(?:[\w-]+\s+){{0,4}}?(?:{one})\b")
_ONLY_WOMEN = re.compile("(?i)" + _ONLY.format(one=_W, other=_M, adj="female"))
_MIXED = re.compile(rf"(?i)\b(?:{_M}\s+(?:and|or|&)\s+(?:\w+\s+)?{_W}|{_W}\s+(?:and|or|&)\s+(?:\w+\s+)?{_M}|"
                    r"male\s+(?:and|or|&)\s+female|female\s+(?:and|or|&)\s+male)\b")
_ONLY_MEN = re.compile("(?i)" + _ONLY.format(one=_M, other=_W, adj="male"))
_NOW = re.compile(r"(?i)\b(?:modern(?:[- ](?:day|times?))?|present[- ]day|today|contemporary|nowadays|current[- ]day|"
                  r"20[2-3]0s|202\d)\b")
# a role from an earlier decade ("1950s pin-up model", "1957 lifeguard") in a World set today
_PAST_ROLE = re.compile(r"(?i)\b(?:1[5-9]\d0s|1[5-9]\d\d|19\d0s|(?:the )?(?:twenties|thirties|forties|fifties|sixties|"
                        r"seventies|eighties|nineties))\b")


def _says_now(t: str) -> bool:
    """The World line is set today: it says modern, present day, today or 2020s, and names no
    earlier year, decade or era ("a 1965 airline, nothing modern" is not)."""
    from .casting import _ERA
    for m in _ERA.finditer(t):
        y = m.group(1) or m.group(3) or ""
        if not y or int(y[:4]) < 2000:
            return False
    for m in _NOW.finditer(t):
        before = t[max(0, m.start() - 16):m.start()].lower()
        if re.search(r"\b(?:no|not|nothing|without|non|never)[\s-]*(?:\w+\s+)?$", before):
            continue  # "nothing modern", "no modern clothes"
        return True
    return False


def rules(theme: str) -> dict:
    """What the World line says about the cast, which the defaults never override: "cast" people or
    creatures, "gender" woman or man when it asks for one only, "now" when it is set today."""
    t = " ".join((theme or "").split())
    creatures = bool(_NO_PEOPLE.search(t) or _CREATURE_CAST.search(t))
    w, m = bool(_ONLY_WOMEN.search(t)), bool(_ONLY_MEN.search(t))
    if _MIXED.search(t):
        w = m = False  # "all the men and women", "men and women only"
    return {"cast": "creatures" if creatures else "people",
            "gender": "" if creatures or w == m else ("woman" if w else "man"), "now": _says_now(t)}


def brief(r: dict) -> str:
    """The lines at the top of every prompt that writes characters: the World line wins."""
    lines = [BRIEF_FIRST]
    if r.get("cast") == "creatures":
        lines.append(RULE_LINES["creatures"])
    elif r.get("gender"):
        lines.append(RULE_LINES[r["gender"]])
    if r.get("now"):
        lines.append(RULE_LINES["now"])
    return " ".join(lines)


def _strict(r: dict) -> str:
    return STRICT_CREATURES if r.get("cast") == "creatures" else STRICT_NOW if r.get("now") else STRICT


CREATURE_KEYS = """each an object with nine keys. "role": what kind of creature it is, as the theme names it, at most 5 words, no names: every creature the theme lists comes first, each once, before you invent any of your own, and invented ones are just as specific to this world. "kind": "creature". "look": the theme's own words, with every detail, when the theme describes this creature; otherwise what it is made of, its shape and size, its colours and textures, and its face: the eyes and the mouth it sings with; never a person, a person in costume, or a person's skin, hair, hands or clothes unless the theme gives it them; at most 30 words. "wear": only what the theme says it wears or carries, else an empty string. "place": where this creature belongs, starting with "in" or "on"; at most 14 words. "light": the real light source in that place, one you could point at, and its colour; at most 10 words. "doing": what this creature does in that place while singing, seen in a wide shot, the way a creature like it would move, starting with a verb ending in -ing, alone; at most 9 words. "face": this creature's expression in a close-up portrait, in the theme's tone, the face only; at most 10 words."""


def character_keys(r: dict) -> str:
    """The keys each character is written with, under the World line's rules."""
    if r.get("cast") == "creatures":
        return CREATURE_KEYS
    k = CHARACTER_KEYS
    if r.get("gender"):
        k = k.replace(GENDER_SENTENCE, f'"gender": "{r["gender"]}" for every character, because the theme says so.')
        if r["gender"] == "woman":
            k = k.replace(", a beard or moustache for some of the men only", "")
    if r.get("now"):
        k = k.replace("a hairdo of the period", "a present-day hairdo").replace(
            "every piece right for its place and period", "every piece what people really wear there today")
    return k


# Characters come whole (who + outfit + place), so a role reads at a glance and an outfit always
# fits the person wearing it (picking who and wear separately gives men in lace corsets and
# night-shift workers you can't place). The prompt also carries no example phrases: Gemma copies
# them into every bible (the same silver-beehive woman and the same two poses in every cast).
PROMPT = """You design the look of a music video. Theme: "{theme}"
{brief}
Invent the world of this video. If the theme names jobs, roles, eras or kinds of people, every character is one of them and recognisable at a glance: their outfit, props and place say what they are. Go for specific, era-perfect detail, as if the stylist, the hair and make-up team and the set designer cared about every piece, in the tone the theme asks for: playful and over the top when it is fun, natural and grounded when it says straight, cinematic or serious. Everything must be physically real, things that existed and could be photographed. Never use brand or product names, never real or famous people, nobody holds a microphone, phone or instrument, and nothing carries writing: no signs, posters, banners, menus, screens or scoreboards. Answer with one JSON object only, no other text, with these keys:
"characters": a list of {characters} different characters, all adults, {character_keys} When the theme names places or a time of day, most characters are in those places at that time. Every place, prop and moment the theme names belongs to at least one character's place, outfit or "doing". Vary everything across the list: roles, ages within what the theme describes, skin tones, builds, faces and hairdos; no two people look alike and no two outfits are alike. {strict}
"places": a list of {places} more specific places in this world, each starting with "in" or "on". At most 16 words each.
"light": a list of {light} lighting setups, each a real source you could point at in this world (window daylight, the sun, a lamp, overhead tubes, a camera's own flash, studio softboxes, streetlights), said plainly with its colour; no coloured gels, washes or backdrops, no mood words, never photographers, crowds or other people. At most 14 words each.
"camera": one plain sentence of at most 22 words starting with "Shot on": the kind of camera, film or video this world would have been photographed with in its own time, and the colour, grain and flash it gives; no brand or model names, no mood words. For a world from before colour photography (before about 1940): colour motion-picture film like a lavish period film, never a black-and-white process. For a present-day world: an unretouched modern digital photograph.
"actions": a list of {actions} things anyone in this world could do in its places while singing, seen in a wide shot, each starting with a verb ending in -ing, never tied to one role, never naming a role or another person, and never generic like "working on a task". They are alone and their hands are free. At most 9 words each.
"poses": a list of {poses} different expressions for a portrait that anyone in this world could have, in the theme's tone, describing the face only, never tied to one role, place or object, never naming a role or another person. At most 12 words each.
"grade": the colour grade that fits the theme best, one of: {grades}."""

# A creature World (the World line says no people, or all creatures, animals, robots...): the same
# world, places, light, camera and grade, with creatures in place of people
_P = PROMPT.split("\n")
PROMPT_CREATURES = "\n".join([_P[0], _P[1],
    "Invent the world of this video. Its characters are the creatures the theme describes, never people: each one "
    "recognisable at a glance from what it is made of and how it looks, as if a creature designer and the set "
    "designer cared about every detail, in the tone the theme asks for. Everything must look physically real, as if it "
    "could be photographed. Never use brand or product names, never real or famous people or characters, nothing "
    "holds a microphone, phone or instrument, and nothing carries writing: no signs, posters, banners, menus, screens "
    "or scoreboards. Answer with one JSON object only, no other text, with these keys:",
    '"characters": a list of {characters} different creatures, {character_keys} When the theme names places or a time '
    "of day, most creatures are in those places at that time. Every place and moment the theme names belongs to at "
    'least one creature\'s place or "doing". Vary everything across the list: kinds, materials, sizes, colours and '
    "faces; no two creatures look alike. {strict}",
    _P[4], _P[5].replace("never photographers, crowds or other people", "never people"), _P[6],
    '"actions": a list of {actions} things any creature in this world could do in its places while singing, seen in a '
    "wide shot, each starting with a verb ending in -ing, never naming one kind of creature or a person, and never "
    "generic. At most 9 words each.",
    '"poses": a list of {poses} different expressions for a close portrait that any creature in this world could have, '
    "in the theme's tone, describing the face only, never naming one kind of creature or a person. At most 12 words "
    "each.", _P[9]])
assert len(_P) == 10 and _P[4].startswith('"places"') and _P[9].startswith('"grade"')

ROLES_CREATURES = """You design the look of a music video. Theme: "{theme}"
{brief}
Creatures already in the video: {taken}.
List {n} more kinds of creature that belong in this theme's own places ({places}), described the way the theme describes its creatures, each one obviously part of this world at a glance and different from the creatures already in the video and from each other, never a person, at most 4 words each, no names. Answer with one JSON object only, no other text: {{"roles": ["...", ...]}}"""


# Shown the cast so far and asked for more, Gemma writes the cast so far out again and adds one
# or two. Two steps instead: new role names first, then those roles written out, each with the
# gender the cast needs.
ROLES_PROMPT = """You design the look of a music video. Theme: "{theme}"
{brief}
Roles already in the video: {taken}.
List {n} more roles, jobs or kinds of people who belong in this theme's own places at its own moment (when the theme is one day, one event or one night, the people who are really there then and part of it). Most of them belong in these places of the theme, as part of what happens there: {places}. Each one is obviously tied to the theme at a glance: the people at the heart of what happens there, who perform, compete, rule, host or watch it (the stars, the rich and powerful, eye-catching guests and spectators), never people who only work behind the scenes or keep the place running (cleaners, sweepers, clerks, scribes, mechanics, porters, vendors, cooks, labourers) unless the theme itself names that kind of person. Every one different from the roles already in the video and from each other, each as specific to this world as those and standing on its own (never someone's assistant, servant or relative, never a generic "customer", "passenger", "resident", "local" or "friend"), at most 4 words each, no names. Give each the gender of the person who would really hold that role in this world's place and time; about half of the new roles are held by women, so pick roles women really held there. {strict} Answer with one JSON object only, no other text: {{"roles": [{{"role": "...", "gender": "woman" or "man"}}, ...]}}"""

# A role only one gender held there and then keeps that gender (strict period accuracy): checked
# over the whole cast once it is written (the first world call can give a woman chariot racer)
GENDER_PROMPT = """A music video is set in this world: "{theme}"
For each role below, say who really held it in this world's place and time: "women" if only women did, "men" if only men did, "both" if both did. Roles: {roles}.
Answer with one JSON object only, no other text, each role exactly as written with its answer: {{"roles": {{"<role>": "women" or "men" or "both", ...}}}}"""

FILL_PROMPT = """You design the look of a music video. Theme: "{theme}"
{brief}
Write these {n} characters for it, in this order, one each: {wanted}.
Everything must be physically real, things that existed and could be photographed. Never use brand or product names, never real or famous people, nobody holds a microphone, phone or instrument, and nothing carries writing. {strict} Answer with one JSON object only, no other text: {{"characters": [...]}}, a list of these {n} {who_kind} {character_keys}"""

# phrases the old prompt showed as examples; Gemma copied them verbatim into bibles cached before v4
COPIED = {"a woman in her sixties with deep brown skin, a towering silver beehive and cat-eye glasses on a beaded chain",
          "chin resting on a fist with a huge cheesy grin", "pointing at the lens with total swagger"}


def _adults(s: str) -> str:
    """Every character is an adult: 'in her late teens' -> 'in her early twenties'."""
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


# things that show text (a tablet, a magazine, record sleeves) count as props too
_PROPS = re.compile(r"(?i)\b(tablets?|laptops?|monitors?|screens?|magazines?|newspapers?|books?|notebooks?|sketchbooks?|"
                    r"vinyl(?: records?)?|records|menus?|clipboards?|maps?|hymnals?|hymn ?books?|prayer ?books?|"
                    r"guitars?|drums?|drumsticks|keyboards?|keytars?|pianos?|basses|bass guitar|violins?|"
                    r"saxophones?|trumpets?|instruments?|microphones?|mics?|mic stands?|headsets?|phones?|"
                    r"cell ?phones?|flip ?phones?|smartphones?|turntables?|"
                    # a singer's mouth must be free (no sipping a drink, no pipe)
                    r"sipping|drinking|eating|chewing|smoking|pipes?|cigarettes?|cigars?|straws?)\b")


def _no_props(s: str) -> str:
    """Drop the clauses of a place or action that put an instrument, a microphone or a phone in
    someone's hands ('strumming a guitar', 'slamming a drum')."""
    if not _PROPS.search(s):
        return s
    parts = [p for p in re.split(r",\s*", s) if not _PROPS.search(p)]
    return ", ".join(parts).strip(" ,")


# Light and places that bring other people into a one-person shot: "intense flashes of white
# light mimicking paparazzi cameras" puts photographers behind the singer.
_CROWD = re.compile(r"(?i)\b(paparazzi|photographers?|flashbulbs?|flashing cameras|press pit|crowds?|audiences?|"
                    r"fans|onlookers|spectators|bystanders|passers-?by|backing dancers|dancers|backing band|"
                    r"performers?|players|people|patrons|customers|guests|partygoers|bustling|crowded|packed)\b")
# a character doing something to someone ("a barber ..., trimming hair") brings
# the customer into a one-person portrait
_NO_OTHERS = re.compile(r"(?i),\s*(?:trimming|cutting|styling|shaving|serving|helping|checking on|tending to|"
                        r"talking to|chatting with|waiting on|greeting|handing|pouring for)\b[^,]*")
_TWO = re.compile(r"(?i)\b(couple|pair|duo|twins|and (?:his|her|their) (?:wife|husband|partner|friend|kid|son|daughter)|"
                  r"strollers?|prams?|babies|baby|toddlers?|infants?|with (?:a|her|his) (?:dog|child|kid))\b|\band (?:fe)?male\b")
_LIGHT_SPARE = ["soft daylight from a large window to one side", "warm light from a lamp in the room",
                "bright, even light from large softboxes, like a magazine studio shoot",
                "hard light from a single bare bulb overhead", "the camera's own flash, with dark falloff behind"]
# Coloured washes and gels read as AI, not as a photograph (unnatural blue and pink light), while
# plain window or lamp light looks real. A light has to be a source you could point at; mood
# clauses ("creating a dreamy atmosphere", "casting a
# nostalgic glow", "simulating a nighttime sky") go.
_WASH = re.compile(r"(?i)\b(gels?|wash(?:es|ed)?|backdrop|hues|swirl(?:s|ing)?|dreamy|accents|bathed|haze of|"
                   r"reflect(?:s|ing|ions?|ed)?)\b")  # reflections: light off chrome can give a mirror twin
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


# black-and-white processes: given "a large format glass plate camera using wet collodion
# process" (a bible for ancient Rome), Krea draws some portraits in pure black and white on every take
_MONO_CAMERA = re.compile(r"(?i)\b(?:wet[- ]?collodion|collodion|glass[- ]plate|daguerreotype|tintype|ambrotype|"
                          r"calotype|albumen|cyanotype|silver[- ]gelatin|orthochromatic|black[- ]and[- ]white|"
                          r"b\s*&\s*w|monochrome|monochromatic|sepia|silent[- ]film|hand[- ]crank(?:ed)?)\b")
COLOUR_PERIOD_CAMERA = ("Shot on colour motion-picture film like a lavish period film, fine grain, "
                        "rich natural colour and warm practical light")


def _camera(v) -> str:
    """The world's period camera line, without brand or model names, always in colour."""
    if not isinstance(v, str):
        return ""
    if _MONO_CAMERA.search(v):
        return COLOUR_PERIOD_CAMERA
    v = _CAM_BRANDS.sub("", " ".join(v.split())).strip(" .,")
    # "The video was captured using X, utilizing Y for a nostalgic feel, resulting in a grainy
    # aesthetic" -> "Shot on X, Y, a grainy look"
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
    """Drop actions and poses that name a character ('The waitress refills a coffee cup' in the
    mother's portrait draws the waitress next to her), or start with 'The ...'; keep the list if
    that would empty it."""
    pat = re.compile(r"(?i)\b(" + "|".join(re.escape(r) for r in sorted(roles, key=len, reverse=True)) + r")s?\b") \
        if roles else None
    keep = [x for x in v if not re.match(r"(?i)^the\s", x.strip()) and not (pat and pat.search(x))]
    return keep if len(keep) >= 3 else [re.sub(r"(?i)^the\s+[a-z -]+?\s+(?=[a-z]+s\b)", "", x) for x in v]


def _plain(wear: str) -> str:
    """A 'graphic tee' or 'band tee' comes out with garbled print on it."""
    # quoted words on clothes are drawn as garbled lettering ("satin sash with 'Miss County Fair'
    # embroidered")
    wear = re.sub(r"\s*(?:with|reading|saying|that says|bearing)?\s*(?<![\w])[\"'\u2018\u201c][A-Z][^\"'\u2019\u201d]{1,40}"
                  r"[\"'\u2019\u201d](?:\s*(?:embroidered|printed|written|stitched|on it|on the front|across it))?", "", wear)
    wear = re.sub(r"(?i)\b(graphic|band|logo|printed|slogan)\s+(t-?shirt|tee)\b", r"plain \2", wear)
    if "lettering" not in wear.lower():
        wear = re.sub(r"(?i)\b((?:satin |pageant |silk )?sash|bowling shirt|jersey|varsity jacket|letterman jacket)\b",
                      r"plain \1 with no lettering", wear, count=1)
    # caps come out with a real team or brand logo on the front
    if "no logo" not in wear.lower():
        wear = re.sub(r"(?i)\b(baseball caps?|trucker hats?|snapbacks?|ball caps?)\b", r"\1 with no logo", wear, count=1)
    return wear


def _copied(x: str) -> bool:
    return any(c in x.lower() for c in COPIED)


# glasses in the bible draw glasses on every other face: eyewear only
# when the World line itself asks for it
EYEWEAR = re.compile(r"(?i)\b(glasses|eyeglasses|spectacles|sunglasses|shades|sunnies|goggles|monocles?|pince-nez|"
                     r"lorgnettes?|eyewear|cat-eye|horn-rimmed|wire-rimmed|aviators|bifocals|frames)\b")
FACIAL_HAIR = re.compile(r"(?i)\b(beards?|bearded|moustaches?|mustaches?|stubble|goatees?|sideburns|whiskers|"
                         r"facial hair|five o'clock shadow|mutton ?chops)\b")
_WOMAN_ROLE = re.compile(r"(?i)\b(woman|women|lady|ladies|girl|girlfriend|waitress|actress|hostess|stewardess|seamstress|"
                         r"priestess|empress|queen|princess|duchess|countess|baroness|nun|maid|matron|mother|wife|"
                         r"widow|bride|cowgirl|showgirl|ballerina|songstress|heiress|governess|daughter|sister|aunt|"
                         r"niece|grandmother|granddaughter|grandma|gal|lass|madam|mrs|miss|she|her)\b")
_MAN_ROLE = re.compile(r"(?i)\b(man|men|gentleman|guy|boyfriend|waiter|actor|host|steward|priest|emperor|king|prince|duke|"
                       r"count|baron|monk|father|husband|widower|groom|cowboy|showman|son|brother|uncle|nephew|"
                       r"grandfather|grandson|grandpa|fella|lad|mister|sir|he|his|him)\b")


def _drop_clauses(s: str, pat: re.Pattern) -> str:
    """s without the comma / 'and' / 'with' clauses that match pat."""
    if not pat.search(s or ""):
        return s
    parts = [p for p in re.split(r",\s*|;\s*|\s+and\s+(?=[a-z])|\s+with\s+(?=[a-z])", s) if p.strip() and not pat.search(p)]
    return ", ".join(parts).strip(" ,")


def _gender_of(val: str, role: str, look: str, wear: str) -> str:
    g = (val or "").strip().lower()
    if g in ("woman", "female", "f", "women", "girl", "lady"):
        return "woman"
    if g in ("man", "male", "m", "men", "boy", "gentleman"):
        return "man"
    for s in (role, look, wear):
        w, m = bool(_WOMAN_ROLE.search(s or "")), bool(_MAN_ROLE.search(s or ""))
        if w != m:
            return "woman" if w else "man"
    return ""


def _who_line(role: str, gender: str, age: str, look: str) -> str:
    """'a gladiator, a man aged 34, deep olive skin, short black curls and a broken nose': the role
    first so it reads at a glance, then who this person is."""
    role = _role(role.strip(" ,.-"))
    role = re.sub(r"(?i)^(?:an?|the)\s+", "", role)
    person = f"a {gender or 'person'}" + (f" aged {age}" if age else "")
    return f"{_article(role)} {role}, {person}" + (f", {look}" if look else "")


def _characters(v, allow_eyewear: bool = False, rules: dict | None = None) -> list[dict]:
    """[{who, wear, place, ...}] from the model's 'characters', tolerating strings and odd keys.
    v7 characters also carry role, gender, age and look (who is built from them). rules (the World
    line's, rules()): a creature cast has no gender, age, skin or hair; a cast of one gender drops
    the others; a World set today drops roles from earlier decades."""
    if not isinstance(v, list):
        return []
    rules = rules or {}
    creatures = rules.get("cast") == "creatures"
    out = []
    for x in v:
        if not isinstance(x, dict):
            continue
        low = {str(k).strip().lower(): str(val) for k, val in x.items()}
        who = low.get("who") or low.get("character") or low.get("person") or ""
        wear = low.get("wear") or low.get("outfit") or low.get("clothes") or ""
        extra = {}
        role = " ".join((low.get("role") or "").split())
        if role:
            role = _adults(_debrand(_NAME.sub("", role)))
            # the role alone: "rodeo clown entertaining the crowd" puts a crowd in the portrait,
            # "cowgirl in pink fringe" repeats the outfit; what they do and wear has its own key
            role = re.sub(r"\s*\((?:a\s+)?(?:woman|man|female|male)\)", "", role)  # the gender asked for, echoed
            short = re.sub(r"\s+(?:(?!(?:king|viking|darling|wedding|building|evening|morning|earring|string|swing|"
                           r"wing|ring|spring|ceiling|clothing|pudding|sterling|duckling|lightning)\s)[a-z]+ing\s+"
                           r"(?:the|a|an|his|her|their|its|some|[a-z]+s)\b.*|"
                           r"(?:in|on|at|with|from|during|behind|near)\s.*)$", "", role).strip(" ,")
            role = short if len(short) >= 3 else role
            if rules.get("now") and _PAST_ROLE.search(role):
                continue  # a 1950s pin-up model in a World set today
            look = _adults(_debrand(" ".join((low.get("look") or low.get("appearance") or "").split()).strip(" ,.")))
            look = look[:1].lower() + look[1:] if look[:2].istitle() or look[:1].isupper() and look[1:2].islower() else look
            if creatures:  # no gender, age, skin or hair: what it is made of and its face, as written
                look = ", ".join(x for x in re.split(r",\s*|;\s*", look)  # a person's skin tone or hairdo,
                                 if x.strip() and not _HUMAN_LOOK.search(x)       # written anyway: the whole clause
                                 and not _PERSON_SIZE.search(x))                  # "about the height of a man"
                rname = _role(re.sub(r"(?i)^(?:an?|the)\s+", "", role))
                who = _creature_who(rname, look)
                extra = {"role": rname, "kind": "creature", "gender": "",
                         "age": "", "look": look}
            else:
                look = _no_props(_drop_clauses(look, EYEWEAR))  # a face never comes with glasses unasked
                role = _one_person(role)  # "medical staff", "cheerleaders": one person to a portrait
                gender = _gender_of(low.get("gender", ""), role, look, wear)
                if rules.get("gender") and gender and gender != rules["gender"]:
                    continue  # "No men": a man is left out (more women are written to take his place)
                gender = gender or rules.get("gender", "")
                if gender == "woman":
                    look = _drop_clauses(look, FACIAL_HAIR)
                m = re.search(r"\d{2}", low.get("age", ""))
                age = str(min(80, max(21, int(m.group(0))))) if m else ""  # 21: a margin above the adults rule
                who = _who_line(role, gender, age, look)
                extra = {"role": _role(re.sub(r"(?i)^(?:an?|the)\s+", "", role)), "gender": gender, "age": age,
                         "look": look}
        if wear and not allow_eyewear:
            wear = _drop_clauses(wear, EYEWEAR)
        if extra:
            extra["eyewear"] = bool(EYEWEAR.search(wear or ""))
        if creatures and not extra:
            continue  # a creature cast is written with roles
        place = low.get("place") or low.get("where") or low.get("location") or ""
        light = " ".join((low.get("light") or low.get("lighting") or "").split()).strip(" ,.")
        # "neon sign, red" draws garbled lettering: the sign scrub, as on the world's lists
        light = (_real_light([_debrand(light)]) or [""])[0] if light and not _CROWD.search(light) else ""
        who = _adults(_debrand(_NAME.sub("", " ".join(who.split()).strip(" ,.-"))))
        who = _no_props(_NO_OTHERS.sub("", who).strip(" ,"))
        if _TWO.search(who):
            continue  # a couple, or someone with a baby, is two people in a one-person portrait
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
        if len(who) < 4 or (len(wear) < 4 and not creatures) or _copied(who):
            continue  # a creature may wear nothing
        if _WORDS.search(wear) and not _NO_WORDS.search(wear):
            continue  # lettering on clothes comes out as garbled text
        doing = _own(low.get("doing") or low.get("action") or low.get("activity") or "", ing=True)
        face = _own(low.get("face") or low.get("expression") or low.get("pose") or "")
        face = "" if _HEAD_DOWN.search(face) else face
        out.append({"who": who, "wear": wear, "place": place, **({"light": light} if light else {}),
                    **({"doing": doing} if doing else {}), **({"face": face} if face else {}), **extra})
    # a character's own action never names another one ("spinning the DJ's record")
    roles = _role_words(out)
    for c in out:
        others = roles - _role_words([c])
        pat = re.compile(r"(?i)\b(" + "|".join(re.escape(r) for r in sorted(others, key=len, reverse=True)) + r")s?\b") \
            if others else None
        for k in ("doing", "face"):
            if pat and k in c and pat.search(c[k]):
                del c[k]
    return out


def _ing(s: str) -> str:
    """'twirls her lasso' -> 'twirling her lasso' (Gemma 3 sometimes writes actions this way)."""
    m = re.match(r"([A-Za-z]+)(\b.*)$", s)
    if not m or m.group(1).lower().endswith("ing") or not m.group(1).lower().endswith("s"):
        return s
    w, rest = m.group(1).lower(), m.group(2)
    if w in ("ties", "lies", "dies"):
        return w[0] + "ying" + rest
    if w.endswith("ies"):
        base = w[:-3] + "y"
    elif re.search(r"(sh|ch|x|ss|z)es$", w):
        base = w[:-2]
    elif w.endswith("ss"):
        return s  # not a verb in the third person
    else:
        base = w[:-1]
    if base.endswith("ie"):
        word = base[:-2] + "ying"
    elif base.endswith("e") and not re.search(r"(ee|ye|oe)$", base):
        word = base[:-1] + "ing"
    elif re.search(r"[^aeiou][aeiou][bcdfghjklmnpqrstvz]$", base) and len(re.findall(r"[aeiou]+", base)) == 1:
        word = base + base[-1] + "ing"  # spin -> spinning
    else:
        word = base + "ing"
    return word + rest


# a portrait has to see the mouth (the choir singer's 'head bowed in prayer')
_HEAD_DOWN = re.compile(r"(?i)\b(bow(?:s|ed|ing)?|eyes (?:closed|shut)|looking down|downcast|hidden|covering)\b")


def _own(s: str, ing: bool = False) -> str:
    """A character's own action or expression, scrubbed like the shared lists: no props in the hands
    or at the mouth, nobody else, no crowd. Gemma 3 writes one action per character even when asked
    for a shared list ('The cowgirl twirls her lasso', 'The radio DJ spins a record'); shuffled
    into a pool, those would have a gas-station attendant spinning the DJ's record."""
    s = " ".join(str(s or "").split()).strip(" ,.")
    s = re.sub(r"(?i)^(?:is|are|while|she is|he is|they are|she|he|they)\s+", "", s)
    if ing:
        s = _ing(s)
    s = _no_props(_debrand(s))
    s = _NO_OTHERS.sub("", ", " + s).strip(" ,")
    if len(s) < 6 or _CROWD.search(s) or _TWO.search(s) or _copied(s) or re.match(r"(?i)^the\s", s):
        return ""
    if ing and not re.match(r"(?i)^[a-z]+ing\b", s):
        return ""
    return s[0].lower() + s[1:]


def cache_dir(out_root: str) -> str:
    """output/oneshot-mv/themes: the cached Worlds and casts."""
    return os.path.join(out_root, "oneshot-mv", "themes")


def _key(theme: str, seed: int) -> str:
    return hashlib.sha1(f"{theme.strip()}|{seed}|v11".encode("utf-8")).hexdigest()[:16]  # v11: the bible format version; a new one stops older cached bibles from being reused


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
        log(f"[OneShotMV] theme: the language model failed ({e}); using templates")
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
    # curly quotes as the JSON's own quotes (a list can switch to them halfway through)
    if "\u201c" in t or "\u201d" in t:
        t2 = t.replace("\u201c", '"').replace("\u201d", '"')
        for fix in (lambda s: s, lambda s: re.sub(r",\s*([\]}])", r"\1", s)):
            try:
                d = json.loads(fix(t2))
                if isinstance(d, dict):
                    return d
            except Exception:
                continue
        t = t2
    return _salvage(t)


def _salvage(t: str) -> dict | None:
    """A cut-off answer (the model ran out of length mid-list): the complete character objects."""
    i = t.find('"characters"')
    if i < 0:
        return None
    out, depth, start, in_str, esc = [], 0, None, False, False
    for k in range(t.find("[", i) + 1, len(t)):
        ch = t[k]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = k
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    out.append(json.loads(re.sub(r",\s*}", "}", t[start:k + 1])))
                except Exception:
                    pass
                start = None
        elif ch == "]" and depth == 0:
            break
    return {"characters": out} if out else None


_NAME = re.compile(r"^[A-Z][a-z]+(?: [A-Z][a-z]+)+\s*:\s*")  # "Elara Vance: 23, ..." -> "23, ..."
_WORDS = re.compile(r"\b(logos?|lettering|slogans?|name ?tags?|insignia|printed words|text)\b", re.I)
_NO_WORDS = re.compile(r"\b(no|without)\b[^,.]*\b(logos?|lettering|slogans?|text)\b", re.I)


# brand names the model slips in anyway ("oversized Versace sunglasses"): a generic word instead,
# so the image models don't draw a logo
BRANDS = re.compile(r"\b(versace|gucci|chanel|prada|louis vuitton|vuitton|dior|fendi|hermes|hermès|burberry|"
                    r"cartier|rolex|ray-?ban|oakley|nike|adidas|reebok|levi'?s|calvin klein|"
                    r"tommy hilfiger|ralph lauren|lacoste|fila|coca-?cola|pepsi|mcdonald'?s|"
                    r"cadillac|rolls-?royce|ferrari|lamborghini|porsche|mercedes(-benz)?|bmw|polaroid|walkman|"
                    r"sony|gameboy|game boy|nintendo|casio|swatch|members only|timberlands?|dr\.? martens|doc martens|"
                    r"skechers|juicy couture|von dutch|fubu|abercrombie|hollister|ugg|heelys)\b", re.I)
# brand names that are also plain words (a champion gladiator, a supreme court, a puma, to converse,
# delivery vans, crocodiles): only before a garment (otherwise "champion" turns into "a designer
# charioteer" or "a designer murmillo gladiator")
BRANDS_IN_CONTEXT = re.compile(r"(?i)\b(champion|supreme|puma|converse|vans|crocs|tommy)(?=\s+(?:hoodies?|sweat(?:shirt|er)s?|"
                               r"t-?shirts?|tees?|sneakers|trainers|shoes|slip-ons|jackets?|caps?|shirts?|logos?|"
                               r"tracksuits?|clogs|sandals|jeans|bags?)\b)")


_SIGNS = "signs?|banners?|posters?|billboards?|marquees?|placards?|menu boards?"


def _design(s: str) -> str:
    """Signs come out as garbled lettering ("neon signs reflecting off the floor", "neon pink and
    blue signs advertising early-bird specials", "a bench adorned with hand-painted signs"): plain
    light or plain decoration instead, and never what they say."""
    s = re.sub(rf"(?i)\b({_SIGNS})\s+(advertising|reading|saying|announcing|promoting|offering|for|that say)\b[^,.;:]*",
               r"\1", s)
    s = re.sub(r"(?i)\b(neon(?:\s+[a-z-]+){0,4}?)\s+signs\b(?!-)", r"\1 tube lights", s)
    s = re.sub(r"(?i)\b(neon(?:\s+[a-z-]+){0,4}?)\s+sign\b(?!-)", r"\1 tube light", s)
    s = re.sub(r"(?i)\b(peace|v|victory|thumbs-up|ok|rock|horns|hand|heart)\s+(sign)s?\b", r"\1 gesture", s)
    s = re.sub(rf"(?i)\b(?:hand-painted|painted|handwritten|hand-lettered|wooden|vintage|faded|old)?\s*"
               rf"(?<!gesture )(?:{_SIGNS})\b(?!-)", " pastel bunting", s)
    return re.sub(r"  +", " ", s).strip()


def _debrand(s: str) -> str:
    s = _design(BRANDS_IN_CONTEXT.sub("designer", BRANDS.sub("designer", s)))
    s = re.sub(r"\bdesigner(\s+designer)+\b", "designer", s)
    # a brand that was the whole item ("fishnet stockings, Doc Martens") leaves a bare word
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


# one person to a portrait: "the blockers" is a blocker, and Gemma's "medical staff" a staff member
_GROUP_WORD = {"staff": "staff member", "crew": "crew member", "personnel": "staff member", "team": "team member",
               "maintenance": "maintenance worker", "catering": "caterer", "security": "security guard",
               "management": "manager", "medical": "medic"}


def _one_person(role: str) -> str:
    words = role.split()
    if not words:
        return role
    last = words[-1]
    if last.lower() in _GROUP_WORD:
        words[-1] = _GROUP_WORD[last.lower()]
    elif re.search(r"(?i)men$", last) and len(last) > 4:
        words[-1] = last[:-3] + "man"  # fishermen
    elif re.search(r"(?i)[^aeiou]ies$", last):
        words[-1] = last[:-3] + "y"  # ladies, fireflies
    elif re.search(r"(?i)esses$", last):
        words[-1] = last[:-2]  # actresses
    elif re.search(r"(?i)[a-z]{3}[^su']s$", last) and not re.search(r"(?i)(?:ics|ous|ss|us|is|as|os)$", last):
        words[-1] = last[:-1]  # blockers, referees, gladiators
    return " ".join(words)


_BARE = re.compile(r"(?i)^(?:an?\s+)?(?:(?:young|older|old|middle-aged|elderly)\s+)?(woman|man|person|lady|gentleman|guy)$")


def _role_first(s: str) -> str:
    """'a Firefighter, 20s, female, light skin, ...' (how Gemma writes whole characters) ->
    'a firefighter, a woman in her twenties, light skin, ...'; 'a Boy-band member, 19, male, ...' ->
    'a boy-band member, a 21-year-old man, ...'. The role stays first so it reads at a glance, and
    an adult noun (man, woman) is always there: role words like "princess" or "boy-band" alone let
    the image model draw someone too young, and nobody may look like a child. Ages under 21 become
    21, a margin above the adults rule."""
    s = re.sub(r",\s*(\d)\d\s*s\s*,", lambda mm: f", {mm.group(1)}0s,", s)  # "45s" -> "40s"
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
        # "a woman in her sixty-three" -> "a woman aged sixty-three"
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
                if cached.get("camera"):
                    cached["camera"] = _camera(cached["camera"])
                if isinstance(cached.get("characters"), list):
                    kept = {(c.get("role"), c.get("place")): c.get("given") for c in cached["characters"]}
                    cached["characters"] = _characters(cached["characters"], allow_eyewear=eyewear_ok(theme),
                                                       rules=cached.get("rules") or rules(theme))
                    for c in cached["characters"]:
                        c["who"] = _people([c["who"]])[0]
                        c["place"] = _places([c["place"]])[0] if c["place"] else ""
                        if kept.get((c.get("role"), c.get("place"))):
                            _with_given(c, kept[(c.get("role"), c.get("place"))])
                    _attach_given(cached["characters"], theme)  # as this version reads the World line
                return cached
            except Exception:
                pass
    r = rules(theme)
    log(f"[OneShotMV] theme: the World line asks for {'creatures, no people' if r['cast'] == 'creatures' else 'people'}"
        + (f", {r['gender']}en only" if r["gender"] else "") + (", set today" if r["now"] else ""))
    prompt = (PROMPT_CREATURES if r["cast"] == "creatures" else PROMPT).format(
        theme=theme.replace('"', "'"), brief=brief(r), grades=", ".join(GRADES), characters=N_CHARACTERS,
        character_keys=character_keys(r), strict=_strict(r), **COUNTS)
    bible = None
    for attempt in range(2):
        text = llm(prompt, seed=seed + attempt, max_length=3200, log=log)  # whole characters run long
        if text is None:
            break
        d = _parse(text)
        if not d:
            log("[OneShotMV] theme: could not read the model's answer, trying again")
            continue
        lists = {k: _clean_list(d.get(k)) for k in COUNTS}
        for k in ("actions", "places", "poses"):
            lists[k] = [y for y in (_no_props(x) for x in lists[k]) if len(y) > 3]
        lists["wear"] = [_plain(x) for x in lists["wear"]]
        roles = _role_words(_characters(d.get("characters"), allow_eyewear=eyewear_ok(theme), rules=r))
        for k in ("actions", "poses"):
            lists[k] = _no_others(lists[k], roles)
        lists["light"] = _alone(lists["light"], _LIGHT_SPARE, n_min=4)
        lists["places"] = _alone(lists["places"])
        chars = _characters(d.get("characters"), allow_eyewear=eyewear_ok(theme), rules=r)
        # the World line's rules can leave out half an answer ("No men"): a few characters are enough,
        # extend() writes the rest under the same rules
        min_chars = 6 if r == {"cast": "people", "gender": "", "now": False} else 2
        if len(chars) >= min_chars:  # who and wear come from the whole characters
            lists["who"] = [c["who"] for c in chars]
            lists["wear"] = [c["wear"] for c in chars]
            lists["places"] = lists["places"] or [c["place"] for c in chars if c["place"]]
        if all(v for k, v in lists.items() if k not in OPTIONAL):
            g = str(d.get("grade", "neutral")).strip().lower()
            bible = {"theme": theme, "source": "gemma-3-12b", **lists,
                     "grade": g if g in GRADES else "neutral", "camera": _camera(d.get("camera")), "raw": text}
            if len(chars) >= min_chars:
                bible["characters"] = chars
                bible["n_main"] = len(chars)  # the World line's own characters come first in the cast
            break
        log("[OneShotMV] theme: the model's answer was missing lists, trying again")
    if bible is None:
        bible = fallback(theme)
    bible["rules"] = r
    bible["who"] = _people([_adults(w) for w in bible["who"]])
    bible["places"] = _places(bible["places"])
    for c in bible.get("characters") or []:
        c["who"] = _people([c["who"]])[0]
        c["place"] = _places([c["place"]])[0] if c["place"] else ""
    if bible.get("characters"):
        given = _attach_given(bible["characters"], theme)
        if given:
            log(f"[OneShotMV] theme: {given} characters keep the World line's own words")
        bible["who"] = [c["who"] for c in bible["characters"]]
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bible, f, indent=1, ensure_ascii=False)
    return bible


def eyewear_ok(theme: str) -> bool:
    """Glasses only when the World line itself puts them in the world."""
    return bool(EYEWEAR.search(theme or ""))


def _cast_line(chars: list[dict]) -> str:
    out = []
    for c in chars:
        if c.get("role"):
            out.append(f"{c['role']} ({c.get('gender') or '?'}, {c.get('age') or '?'})")
        else:
            out.append(c.get("who", "").split(",")[0])
    return "; ".join(out)


def _same(a: dict, b: dict) -> bool:
    """Two characters that would read as the same person: same role, gender and age band."""
    if not a.get("role") or not b.get("role"):
        return a.get("who") == b.get("who")
    try:
        close = abs(int(a.get("age") or 0) - int(b.get("age") or 0)) < 6
    except ValueError:
        close = True
    return a["role"].lower() == b["role"].lower() and a.get("gender") == b.get("gender") and close


def _theme_places(bible: dict) -> str:
    """The places the World line itself names (the part after the colon), else the bible's places:
    without them the extra cast drifts off the World (a Roman market day, with a vinegar merchant,
    a tax collector and a herbalist, in a World set at the games)."""
    th = bible.get("theme", "")
    # World lines read "headline: who; where; light" (as the sample World lines do)
    parts = [x.strip(" .") for x in re.split(r"[;:]", th)[1:] if x.strip()]
    named = [parts[1]] if len(parts) >= 3 else []
    if not named:
        named = [p_.replace("in ", "", 1) for p_ in (bible.get("places") or [])[:6]]
    return "; ".join(named)[:600] or "the places of the theme"


def _role_list(text: str | None) -> list[tuple[str, str]]:
    d = _parse(text or "") or {}
    v = d.get("roles")
    if not isinstance(v, list) and text:
        # a cut-off or malformed answer: the role / gender pairs one by one
        t = (text or "").replace("\u201c", '"').replace("\u201d", '"')
        v = [{"role": m.group(1), "gender": m.group(2)} for m in
             re.finditer(r'"role"\s*:\s*"([^"\n]{2,60})"\s*,\s*"gender"\s*:\s*"(\w+)"', t)]
    if not isinstance(v, list):  # a bare list, or one role per line
        m = re.search(r"\[[^\]]*\]", text or "")
        try:
            v = json.loads(m.group(0)) if m else []
        except Exception:
            v = []
    out, seen = [], set()
    for x in v:
        g = ""
        if isinstance(x, dict):
            g = _gender_of(str(x.get("gender", "")), "", "", "") if str(x.get("gender", "")).strip() else ""
            x = x.get("role") or x.get("name") or ""
        x = re.sub(r"\s*\((?:a\s+)?(?:woman|man|female|male)\)\s*$", "", " ".join(str(x).split()), flags=re.I)
        x = x.strip(" ,.-\"'")
        x = re.sub(r"(?i)^(?:an?|the)\s+", "", _debrand(_NAME.sub("", x)))
        # groups are more than one person ("rodeo clown's family"), and a customer is generic
        if re.search(r"(?i)\b(family|families|kids|children|friends|crew|band|gang|group|team|couple|parents|"
                     r"customers?|passengers?|residents?|locals|tourists?|visitors?)\b", x):
            continue
        if re.search(r"(?i)\w['\u2019]s\s", x):
            continue  # someone's relative or helper is a role made from another role ("the diner
            # waitress's boyfriend", "the lion keeper's guard", "the scribe's papyrus supplier")
        if 2 < len(x) <= 40 and not _CROWD.search(x) and not _TWO.search(x) and x.lower() not in seen:
            seen.add(x.lower())
            out.append((x, g))
    return out


def _true_genders(chars: list[dict], theme: str, seed: int, dump, log=print) -> int:
    """Characters whose gender their role never had in this world's place and time are written
    again with the right one (or left out); returns how many changed."""
    roles = []
    for c in chars:
        r = c.get("role") or c["who"].split(",")[0]
        if r not in roles:
            roles.append(r)
    if not roles:
        return 0
    text = llm(GENDER_PROMPT.format(theme=theme.replace('"', "'"), roles="; ".join(roles)),
               seed=seed + 307, max_length=14 * len(roles) + 120, log=log)
    dump("g", text)
    d = _parse(text or "") or {}
    ans = d.get("roles") if isinstance(d.get("roles"), dict) else d
    held = {}
    for k, v in (ans or {}).items():
        v = str(v).strip().lower()
        if v in ("women", "woman", "female"):
            held[str(k).strip().lower()] = "woman"
        elif v in ("men", "man", "male"):
            held[str(k).strip().lower()] = "man"
    wrong = [i for i, c in enumerate(chars)
             if c.get("gender") and held.get((c.get("role") or c["who"].split(",")[0]).lower(), c["gender"]) != c["gender"]]
    if not wrong:
        log("[OneShotMV] theme: every role has its real gender")
        return 0
    want = [(chars[i].get("role") or chars[i]["who"].split(",")[0], held[(chars[i].get("role") or chars[i]["who"].split(",")[0]).lower()])
            for i in wrong]
    log("[OneShotMV] theme: wrong gender for " + ", ".join(f"{r} (should be {g})" for r, g in want))
    redo = {}
    for b in range(0, len(want), EXTRA_BATCH):
        batch = want[b:b + EXTRA_BATCH]
        rl = rules(theme)
        text = llm(FILL_PROMPT.format(theme=theme.replace('"', "'"), brief=brief(rl), n=len(batch),
                                      wanted="; ".join(f"{_article(r)} {r} ({g})" for r, g in batch),
                                      character_keys=character_keys(rl), strict=_strict(rl),
                                      who_kind="characters, all adults,"),
                   seed=seed + 401 + b, max_length=320 * len(batch) + 600, log=log)
        dump(f"gx{b}", text)
        d2 = _parse(text or "")
        for c in (_characters((d2 or {}).get("characters"), allow_eyewear=eyewear_ok(theme), rules=rl) if d2 else []):
            key = (c.get("role") or "").lower()
            if c.get("gender") and key and key not in redo:
                c["who"] = _people([c["who"]])[0]
                c["place"] = _places([c["place"]])[0] if c["place"] else ""
                redo[key] = c
    out, changed = [], 0
    for i, c in enumerate(chars):
        if i not in wrong:
            out.append(c)
            continue
        new = redo.get((c.get("role") or c["who"].split(",")[0]).lower())
        if new is not None and new.get("gender") == held.get((c.get("role") or "").lower()):
            out.append(new)
            changed += 1
        else:
            changed += 1  # left out: a wrong person is worse than one fewer
    chars[:] = out
    return changed


# Roles that only work behind the scenes or keep a place running: a World built on an event is cast
# from who performs, rules or watches it, so every character is obviously tied to the event (left
# alone, Gemma adds a sand-raker, an armourer, a fortune teller, a silk merchant, a jewel vendor
# and a flute player to gladiator games). Kept when the World line names that kind of job.
_BACKSTAGE = re.compile(r"(?i)\b(vendors?|sellers?|merchants?|traders?|hawkers?|stall ?holders?|cleaners?|sweepers?|"
                        r"janitors?|clerks?|scribes?|secretar(?:y|ies)|accountants?|mechanics?|porters?|cooks?|chefs?|"
                        r"bakers?|labou?rers?|workers?|makers?|artists?|painters?|sculptors?|carpenters?|smiths?|"
                        r"armou?rers?|tailors?|fortune[- ]tellers?|medics?|doctors?|nurses?|physicians?|rakers?|"
                        r"carriers?|measurers?|criers?|players?|musicians?|pipers?|flautists?|harpists?|drummers?|"
                        r"trumpeters?|lyrists?|astrologers?|soothsayers?|beggars?|slaves?|medicus|medici|augurs?|"
                        r"haruspex|operators?|technicians?|attendants?|keepers?|handlers?|grooms?|stewards?|"
                        r"servants?|assistants?|apprentices?|engineers?|builders?)\b")
_ADJ = re.compile(r"(?i)^(?:(?:beautiful|young|old|elderly|rich|wealthy|champion|famous|elegant|glamorous|grizzled|"
                  r"veteran|proud|powerful|noble|handsome|stunning)\s+)+")


def _offstage(role: str, theme: str) -> bool:
    """An invented role that only works behind the scenes, unless the World line names that job."""
    t = theme.lower()
    if re.search(r"(?i)\b(workers?|staff|shift|crew|behind the scenes|backstage)\b", t) and "backstage at" not in t:
        return False
    return any(m.group(0).lower().rstrip("s") not in t for m in _BACKSTAGE.finditer(role))


_ROLE_SPLIT = re.compile(r"(?i)\s+(?:in|with|who|whose|beside|at|on|from|wearing|holding|cheering|betting|"
                         r"fanning|carrying|by|near|under|and|alone|standing|sitting|waiting|leaning|waving|outside|"
                         r"inside|behind|beneath|up|dressed|(?!(?:king|viking)\b)[a-z]{2,}ing(?=\s+(?:(?:up|out|down|off|over|back|away)\s+)?"
                         r"(?:the|a|an|his|her|their|its|some)\b))\b")
# "the owner, a man in his sixties with ...": what the owner looks like, not a second character
_APPOSITIVE = re.compile(r"(?i)^(?:an?\s+)?(?:(?:young|old|older|elderly|tall|short|stout|slim)\s+)?(?:man|woman|person|"
                         r"lady|gentleman|guy|fellow|figure)(?![\w'-])")
# "the announcer and the superfan in face paint": two characters ("a goat with horns and a bell": one)
_AND_SPLIT = re.compile(r"(?i)^((?:an?|the)\s+[\w'-]+(?:\s+[\w'-]+){0,2})\s+and\s+((?:an?|the)\s.+)$")
# a list of places after the cast ("...; the showroom floor; humming light")
_PLACE_NOUN = re.compile(r"(?i)\b(?:room|floor|hall|stage|kitchen|street|alley|beach|field|yard|garden|market|shop|"
                         r"store|stockroom|display|window|deck|dock|pier|harbou?r|arena|rink|track|court|pool|booth|"
                         r"counter|office|lobby|station|platform|cave|forest|valley|river|lake|sea|shore|cliff|ridge|"
                         r"meadow|square|plaza|terrace|rooftop|balcony|corridor|backstage|tent|chapel|church|bar|"
                         r"diner|lot|ring|gym|stands|bleachers|showroom|warehouse|alleyway|sky|water|hallway|"
                         r"cafeteria|clubhouse|gymnasium|tunnel|entrance|buffet)s?$")


def _is_place(desc: str) -> bool:
    """'the hallway of lockers', 'the showroom floor': a place in the cast list, not a character."""
    role = _ROLE_SPLIT.split(re.sub(r"(?i)^\s*(?:an?|the)\s+", "", desc))[0]
    return bool(_PLACE_NOUN.search(role.split(" of ")[0].strip(" ,.")))
# a long creature description: the kind before what it is made of ("snowdrift creature of packed powder")
_ROLE_CUT = re.compile(r"(?i)\s+(?:of|built|made|carved|covered|formed|shaped|drifting|like)\b")


def named_characters(theme: str) -> list[tuple[str, str]]:
    """(role, the World line's own words for that character) for each character it names ("headline,
    all invented adults: a murmillo gladiator in a crested helmet, the emperor in a purple toga, ...;
    places; light"), in its order."""
    segs = [x.strip() for x in theme.split(";")]
    if ":" not in segs[0]:
        return []
    # the cast follows the last colon a character follows: "No people, all animals: a farm at sunrise:
    # a rooster, ..." (not "residents in their eighties ...: early-bird dinner, water aerobics")
    parts = segs[0].split(":")
    k = max((i for i in range(1, len(parts)) if re.match(r"(?i)\s*(?:an?|the)\s", parts[i])), default=1)
    stretches = [":".join(parts[k:])]
    if len(_cast_items(stretches[0], merge=True)) == 1:  # one character to a ";": "the owner, a man in his
        for x in segs[1:]:                                 # sixties with ...; the champion, ...; ..."
            items = _cast_items(x, merge=True)
            if not re.match(r"(?i)(?:an?|the)\s", x) or len(items) != 1 or _is_place(items[0]):
                break  # the places, or the light
            stretches.append(x)
    merge = len(stretches) > 1
    out = []
    for item in (i for x in stretches for i in _cast_items(x, merge)):
        if _is_place(item):
            continue
        desc = re.sub(r"(?i)^\s*(?:an?|the)\s+", "", item).strip(" .,")
        role = _ROLE_SPLIT.split(desc)[0].strip(" .,")
        role = role.split(",")[0].strip()  # "champion murmillo gladiator, scarred and oiled"
        if len(role.split()) > 4 or re.search(r"(?i)\s(?:built|made|carved|covered|formed|shaped)$", role):
            role = _ROLE_CUT.split(role)[0].strip()  # "glacier creature of clear blue ice" -> "glacier creature"
        if _GEAR.search(role.split()[-1] if role.split() else ""):
            continue  # "..., a short sword and a tall shield": the last character's kit, not a character
        one = _one_person(role)
        if one != role and desc.startswith(role):
            desc, role = one + desc[len(role):], one  # "the blockers": a blocker
        if 2 <= len(role) <= 40 and len(role.split()) <= 4 and role.lower() not in [r.lower() for r, _ in out]:  # the DJ
            out.append((role, desc))
    return out


def _cast_items(text: str, merge: bool = False) -> list[str]:
    """The characters in one stretch of a World line's cast, each with its own words. merge: a cast
    written one character to a ";", where "the owner, a man in his sixties" is one character (in a
    list with commas, "a barber, a woman under a hood dryer" are two)."""
    items: list[str] = []
    for item in re.split(r",\s+(?=(?:an?|the)\s)", text):
        item = " ".join(item.split()).strip(" .,")
        m = _AND_SPLIT.match(item)
        parts = [m.group(1), m.group(2)] if m and not _ROLE_SPLIT.search(m.group(1)) else [item]
        for x in parts:
            if merge and items and _APPOSITIVE.match(x):
                items[-1] += ", " + x  # "the owner, a man in his sixties"
            elif x:
                items.append(x)
    return items


def _named_roles(theme: str) -> list[str]:
    """The characters a World line names, as role names (named_characters)."""
    return [r for r, _ in named_characters(theme)]


_AGE_WORDS = re.compile(r"(?i)\b(?:in (?:his|her|their) (?:early |mid-?|late )?(?:twenties|thirties|forties|fifties|"
                        r"sixties|seventies|eighties|nineties)|aged \d{2}|\d{2}-year-old|young|old|elderly|older|"
                        r"middle-aged|seniors?|retired)\b")
_LOOK_WORDS = re.compile(r"(?i)\b(?:skin|skinned|hair|haired|beard|bearded|moustache|mustache|eyes|eyed|face|faced|"
                         r"freckles|freckled|tattoos?|tattooed|scars?|scarred|bald|braids?|curls|bob|bun|ponytail|"
                         r"build|tall|stout|slim|muscular|wrinkled|lashes|cheekbones|made of|of (?:\w+\s){0,3}?(?:ice|"
                         r"snow|stone|granite|clay|moss|bark|leaves|water|sand|fire|smoke|metal|steel|glass|light))\b")


# what only a person has, in a look the model wrote for a creature anyway ("pale skin freckled with
# green, long auburn hair woven with moss"): the clause goes, the moss stays in the next one
_HUMAN_LOOK = re.compile(r"(?i)\b(?:(?:pale|fair|olive|tan(?:ned)?|dark|deep|light|brown|black|ruddy|rosy|sun-darkened|"
                         r"weathered)\s+(?:skin|complexion)|complexion|hair(?:do|cut|line)?|beards?|bearded|moustaches?|"
                         r"mustaches?|stubble|eyebrows|cheekbones|jawline|weathered face|wrinkled face|build)\b")
# a creature's size held to a person's: the person comes out in the portrait
_PERSON_SIZE = re.compile(r"(?i)\b(?:height|tall|size|big|large|small)\b.*\b(?:man|woman|person|human|adults?|child|"
                          r"people|men|women)\b")
# a look that starts with a creature's name, "hulking boulder creature of ...": the name, the rest
_LOOK_HEAD = re.compile(r"(?i)^((?:[\w'-]+\s+){0,4}?creature)\b(.*)$")


def _lower_first(x: str) -> str:
    return x[0].lower() + x[1:] if x[:1].isupper() and x[1:2].islower() else x  # "Heather blossom creature"


def _creature_who(rname: str, look: str) -> str:
    """A creature's who line, named once. The model's look often starts with a name of its own: the
    role again ("an ice shard creature formed from ice"), the role with an adjective ("hulking
    boulder creature of granite", kept), or another creature ("a hulking boulder creature" for the
    granite lichen creature, which then reads as two)."""
    rname = _lower_first(rname)
    m = _LOOK_HEAD.match(re.sub(r"(?i)^(?:an?|the)\s+", "", look))
    if not m:
        return f"{_article(rname)} {rname}" + (f", {look}" if look else "")
    head, rest = m.group(1), m.group(2)
    if rname.lower() not in head.lower():
        head = rname
    if re.match(r"(?i)^\s*,\s*(?:an?|the)\s", rest):  # "a granite lichen creature, a hulking boulder creature of"
        m2 = _LOOK_HEAD.match(re.sub(r"(?i)^\s*,\s*(?:an?|the)\s+", "", rest))
        rest = m2.group(2) if m2 else rest
    head = _lower_first(head)
    rest = rest.rstrip(" ,")
    rest = rest if rest[:1] in ",;" or not rest else " " + rest.lstrip()
    return f"{_article(head)} {head}{rest}"


_DRESSED = re.compile(r"(?i)\b(?:wears?|wearing|dressed|clothes|clothing|carr(?:y|ies|ying)|holding|cloak|cape|robe|"
                      r"shawl|crown|hat|hood|coat|gown|armou?r|helmet|mask)\b")


def _with_given(c: dict, desc: str) -> None:
    """A character the World line describes keeps its words: they lead its who line, and they are
    what the vision check holds the still to. What the line leaves open (an age, a gender, a face)
    comes from the language model as before."""
    c["given"] = desc
    head = f"{_article(desc)} {desc}"
    if c.get("kind") == "creature":
        detailed = len(desc.split()) > len((c.get("role") or "").split()) + 1
        c["who"] = head if detailed or not c.get("look") else f"{head}, {c['look']}"
        if detailed:
            c["look"] = desc
            if not _DRESSED.search(desc):
                c["wear"] = ""  # the model's fur hat on the line's snowdrift creature
        return
    g, age = c.get("gender", ""), c.get("age", "")
    if not _AGE_WORDS.search(desc):
        person = f"a {g or 'person'}" + (f" aged {age}" if age else "")
    else:
        person = f"a {g}" if g and not (_WOMAN_ROLE.search(desc) or _MAN_ROLE.search(desc)) else ""
    looks = bool(_LOOK_WORDS.search(desc))
    c["who"] = ", ".join(x for x in (head, person, "" if looks else c.get("look", "")) if x)
    if looks:
        c["look"] = desc


def _attach_given(chars: list[dict], theme: str) -> int:
    """Give each character the World line describes its own words (story_positions matches them up).
    Returns how many."""
    named = named_characters(theme)
    if not named:
        return 0
    k = 0
    for c, pos in zip(chars, story_positions(chars, theme)):
        if pos is not None and c.get("given") != named[pos][1]:
            _with_given(c, named[pos][1])
            k += 1
    return k


_GEAR = re.compile(r"(?i)^(?:swords?|sica|shields?|nets?|tridents?|helmets?|spears?|lances?|guards?|wraps?|greaves|harness|"
                   r"breastbands?|breastplates?|caps?|hats?|togas?|tunics?|robes?|silks?|pearls?|wreaths?|pelts?|"
                   r"branch(?:es)?|cloaks?|capes?|bags?|cases?|cups?|trays?|aprons?|dress(?:es)?|suits?|coats?|"
                   r"jackets?|shirts?|boots?|shoes?|gloves?|scarf|scarves|belts?|sashes?|necklaces?|earrings?|"
                   r"bracelets?|bangles?|rings?|fans?|whips?|ropes?|chains?|daggers?|knives|knife|axes?|bows?|"
                   r"arrows?|clubs?|staffs?|sticks?|torch(?:es)?|lamps?|flowers?|bouquets?|baskets?|jugs?|"
                   r"bottles?|glasses?|goblets?|coins?|purses?|scrolls?|tablets?|horses?|chariots?|lions?|dogs?)$")


# A World built on a spectacle in front of a crowd (without the arena behind them, gladiators don't
# read as an event): the crowd stays behind the singer as a soft blur
_CROWD_WORLD = re.compile(r"(?i)\b(?:packed|roaring|cheering|screaming|crowded|full)\s+(?:\w+\s+){0,2}?"
                          r"(?:stands|crowd|arena|stadium|colosseum|grandstands?|tiers|audience|amphitheatre|bleachers)\b")


def crowd_world(theme: str) -> bool:
    return bool(_CROWD_WORLD.search(theme or ""))


CROWD_DRESS_PROMPT = """A music video is set in this world: "{theme}"
Behind the singers there is a big crowd of spectators. Describe what that crowd wears, as a film's costume designer would brief extras: the real garments of that time and place and their colours, concrete nouns only, at most 18 words, no numbers of people, nothing from a later time. Answer with the description only, starting with "a sea of"."""


def crowd_dress(bible: dict, seed: int = 1, out_root: str | None = None, log=print) -> str:
    """What a spectacle World's crowd wears, in concrete garments and colours ("a sea of white wool
    togas and red and ochre tunics"): "dressed for ancient Rome" is too abstract, and Krea fills
    the stands with modern spectators. Written once by Gemma and kept with the bible."""
    if bible.get("crowd_dress"):
        return bible["crowd_dress"]
    theme = bible.get("theme", "")
    text = llm(CROWD_DRESS_PROMPT.format(theme=theme.replace('"', "'")), seed=seed + 307, max_length=120, log=log)
    line = " ".join((text or "").replace("\n", " ").split()).strip(' "\'.')
    m = re.search(r"(?i)\ba sea of\b.*", line)
    line = m.group(0) if m else ""
    line = re.sub(r"(?i)\b(t-?shirts?|jeans|caps?|hoodies?|sneakers|trainers|sunglasses|modern)\b[^,;.]*", "", line)
    line = " ".join(line.split()).strip(" ,;.")
    if len(line) < 16:
        return ""
    bible["crowd_dress"] = line
    if out_root:
        path = os.path.join(cache_dir(out_root), _key(theme, seed) + ".json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(bible, f, indent=1, ensure_ascii=False)
        except Exception:
            pass
    log(f"[OneShotMV] theme: the crowd wears {line}")
    return line


def story_positions(chars: list[dict], theme: str) -> list[int | None]:
    """Each character's place in the World line's own list of characters (None for the ones it
    doesn't name): the World line is cast in its own order, so a gladiator World opens on the
    gladiators singing and then cuts to the people in the crowd."""
    named = _named_roles(theme)
    out = []
    for c in chars:
        role = (c.get("role") or (c.get("who") or "").split(",")[0]).strip()
        role = re.sub(r"(?i)^(?:an?|the)\s+", "", role)
        base = _ADJ.sub("", role.lower()).strip()
        # the closest name first: "second gladiatrix" is not "gladiatrix"
        pos = next((k for k, r in enumerate(named) if r.lower() == role.lower()), None)
        if pos is None:
            pos = next((k for k, r in enumerate(named) if _ADJ.sub("", r.lower()).strip() == base), None)
        if pos is None:
            pos = next((k for k, r in enumerate(named) if _has_role(r, [role])), None)
        out.append(pos)
    # one character per named role: a second "gladiator" match keeps no position
    seen, res = set(), []
    for pos in out:
        res.append(pos if pos is not None and pos not in seen else None)
        if pos is not None:
            seen.add(pos)
    return res


def _has_role(role: str, cast: list[str]) -> bool:
    r = _ADJ.sub("", role.lower()).strip()
    return any(r and (r in c.lower() or _ADJ.sub("", c.lower()).strip() in r) for c in cast if c)


def extend(bible: dict, n: int, *, seed: int = 1, out_root: str | None = None, log=print) -> dict:
    """At least n characters in the bible (a big cast, no repeats): new role
    names first, then those roles written out in batches of EXTRA_BATCH, each with the gender
    that keeps the cast about half women and half men (a role only one gender holds keeps it).
    Cached with the bible."""
    chars = bible.get("characters") or []
    theme = bible.get("theme", "")
    if not theme or bible.get("source") != "gemma-3-12b" or (len(chars) >= n and bible.get("genders_checked")):
        return bible

    def dump(tag, text):
        if out_root and text:  # kept for debugging the cast
            try:
                os.makedirs(cache_dir(out_root), exist_ok=True)
                with open(os.path.join(cache_dir(out_root), f"{_key(theme, seed)}_{tag}.txt"), "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass

    r = bible.get("rules") or rules(theme)
    creatures = r.get("cast") == "creatures"
    words = {role.lower(): desc for role, desc in named_characters(theme)}
    taken = [c.get("role") or c["who"].split(",")[0] for c in chars]
    # the World line's own characters the first call left out come first (it can return 12 of 16
    # named roles, leaving out the emperor, the patrician matron and the praetorian guard)
    roles: list = [(x, "" if creatures else _gender_of("", x, "", "") or "") for x in _named_roles(theme)
                   if not _has_role(x, taken)]
    if roles:
        log(f"[OneShotMV] theme: the World line's own characters still to cast: {', '.join(x for x, _ in roles)}")
    for attempt in range(3):
        need = n - len(chars) - len(roles) + 4
        if need <= 4:
            break
        text = llm((ROLES_CREATURES if creatures else ROLES_PROMPT).format(
                       theme=theme.replace('"', "'"), brief=brief(r), taken=", ".join(taken + [x for x, _ in roles]),
                       n=need, strict=_strict(r), places=_theme_places(bible)),
                   seed=seed + 211 + attempt, max_length=24 * need + 160, log=log)
        dump(f"r{attempt}", text)
        if text is None:
            break
        known = {x.lower() for x in taken} | {x.lower() for x, _ in roles}
        got = _role_list(text)
        if r.get("now"):
            got = [(x, g) for x, g in got if not _PAST_ROLE.search(x)]
        off = [] if creatures else [x for x, _ in got if _offstage(x, theme)]
        if off:
            log(f"[OneShotMV] theme: left out (behind the scenes): {', '.join(off)}")
        roles += [(x, g) for x, g in got if x.lower() not in known and x not in off]
    log(f"[OneShotMV] theme: {len(roles)} new roles: {', '.join(f'{x} ({g or chr(63)})' for x, g in roles)}")
    women = sum(1 for c in chars if c.get("gender") == "woman")
    men = sum(1 for c in chars if c.get("gender") == "man")
    queue = []
    # the gender the role really had (the model's, or the role word's); only an open role is
    # given the one the cast is short of (balancing every role would make a woman the senator's bodyguard)
    for x, given in roles:
        if creatures:
            g = ""
        elif r.get("gender"):
            g = r["gender"]  # "No men": every new role is a woman's
        else:
            g = _gender_of("", x, "", "") or given or ("man" if women > men else "woman")
        women, men = women + (g == "woman"), men + (g == "man")
        queue.append((x, g))

    def want(x, g):  # "a snowdrift creature, as the theme describes it: snowdrift creature of packed powder ..."
        desc = words.get(x.lower())
        return f"{_article(x)} {x}" + (f" ({g})" if g else "") + (
            f", as the theme describes it: {desc}" if desc and desc.lower() != x.lower() else "")
    attempt = 0
    while queue and len(chars) < n and attempt < 8:
        batch, queue = queue[:EXTRA_BATCH], queue[EXTRA_BATCH:]
        wanted = "; ".join(want(x, g) for x, g in batch)
        text = llm(FILL_PROMPT.format(theme=theme.replace('"', "'"), brief=brief(r), n=len(batch), wanted=wanted,
                                      character_keys=character_keys(r), strict=_strict(r),
                                      who_kind="creatures, never people," if creatures else "characters, all adults,"),
                   seed=seed + 101 + attempt, max_length=320 * len(batch) + 600, log=log)
        dump(f"x{attempt}", text)
        attempt += 1
        if text is None:
            break
        d = _parse(text)
        new = _characters((d or {}).get("characters"), allow_eyewear=eyewear_ok(theme), rules=r) if d else []
        added = 0
        for c in new:
            c["who"] = _people([c["who"]])[0]
            c["place"] = _places([c["place"]])[0] if c["place"] else ""
            if any(_same(c, o) for o in chars):
                continue
            chars.append(c)
            added += 1
        got = {(c.get("role") or "").lower() for c in new}
        queue += [(x, g) for x, g in batch if x.lower() not in got and _role(x).lower() not in got][:2]  # a dropped one, once more
        log(f"[OneShotMV] theme: cast {len(chars)} characters (+{added} of {len(batch)} asked)")
    if not creatures and not r.get("gender"):  # a creature has none; a one-gender World set them all
        _true_genders(chars, theme, seed, dump, log)
    bible["genders_checked"] = True
    bible["rules"] = r
    if _attach_given(chars, theme):
        bible["who"] = [c["who"] for c in chars]
    if chars:  # saved with the gender check done (and any new characters)
        # a role's own action never names another character (as in _characters, over the whole cast)
        roles_ = _role_words(chars)
        for c in chars:
            others = roles_ - _role_words([c])
            pat = re.compile(r"(?i)\b(" + "|".join(re.escape(r) for r in sorted(others, key=len, reverse=True)) +
                             r")s?\b") if others else None
            for key in ("doing", "face"):
                if pat and key in c and pat.search(c[key]):
                    del c[key]
        bible["characters"] = chars
        bible["who"] = [c["who"] for c in chars]
        bible["wear"] = [c["wear"] for c in chars]
        if out_root:
            path = os.path.join(cache_dir(out_root), _key(theme, seed) + ".json")
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(bible, f, indent=1, ensure_ascii=False)
            except Exception:
                pass
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
