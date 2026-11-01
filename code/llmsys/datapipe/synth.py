"""A synthetic 'web crawl' with ground-truth labels, so every pipeline stage can be tested
and measured offline.

The generator produces HTML pages of several kinds that mimic what a real crawl contains
(articles wrapped in site boilerplate, foreign-language pages, keyword spam, login walls,
product listings, near-duplicate licence templates, exact mirrors) and plants contact
details for the PII stage. Everything is generated from small word lists written for this
book; no real web content is included. Contact details use reserved ranges
(``example.com``-style domains, 555-01xx phone numbers, documentation IP blocks).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

LANG_WORDS: Dict[str, List[str]] = {
    "en": ("the of and to in is that for it with as was on be by this are from at or an "
           "which have not but they their we can has more one all would there been other "
           "into its these also when than many some most such between through during").split(),
    "es": ("el la de que y en un una los las del se por con para es su al lo como mas pero "
           "sus le ya o este porque esta entre cuando muy sin sobre tambien hasta hay donde "
           "desde todo nos durante todos uno les contra otros ese eso ante ellos").split(),
    "de": ("der die und in den von zu das mit sich des auf fur ist im dem nicht ein eine als "
           "auch es an werden aus er hat dass sie nach wird bei einer um am sind noch wie "
           "einem uber einen so zum war haben nur oder aber vor zur bis mehr durch").split(),
    "fr": ("le de la et les des en un du une est que pour dans qui par sur pas plus au avec "
           "ce il sont ne se mais ou comme on sa leur nous elle tout deux aussi cette "
           "entre fait sans peut ses lui dont alors meme donc").split(),
    "it": ("il di che la e un a per in una sono mi ho lo ma ti le si con cosa non se del "
           "della gli anche come questo nel alla essere tutto quando molto degli dei "
           "ancora sempre dove dopo prima tra fra cui").split(),
}

CONTENT = {
    "en": ("kernel memory bandwidth cache processor network compiler model dataset river "
           "forest climate engine bridge library garden museum orchestra theorem protein "
           "satellite harbor market council student teacher village").split(),
    "es": ("ciudad gobierno historia mundo agua tiempo trabajo empresa escuela musica "
           "ciencia pueblo cultura libro casa familia camino ventana").split(),
    "de": ("stadt regierung geschichte welt wasser zeit arbeit unternehmen schule musik "
           "wissenschaft dorf kultur buch haus familie strasse fenster").split(),
    "fr": ("ville gouvernement histoire monde eau temps travail entreprise ecole musique "
           "science village culture livre maison famille chemin fenetre").split(),
    "it": ("citta governo storia mondo acqua tempo lavoro azienda scuola musica scienza "
           "paese cultura libro casa famiglia strada finestra").split(),
}
EN_ADJ = ("efficient large small modern ancient careful detailed reliable public local "
          "remarkable quiet complex simple northern").split()
EN_VERB = ("improves describes measures connects supports reduces explains protects "
           "transforms studies follows").split()
SPAM = ("buy cheap best deal click here free download casino offer discount win now "
        "hot sale prize bonus limited").split()

SITES = ["news.example.org", "wiki.example.net", "blog.example.com", "shop.example.com",
         "forum.example.org", "docs.example.net"]


@dataclass
class SynthDoc:
    url: str
    html: str
    kind: str                     # article | foreign | spam | login | listing | template | mirror
    lang: str
    good: bool                    # ground-truth "keep for pretraining"
    group: Optional[int] = None   # near/exact duplicate group id
    charset: str = "utf-8"
    declare_charset: bool = True
    pii: Dict[str, int] = field(default_factory=dict)


def _sentence(rng: random.Random, lang: str) -> str:
    if lang == "en":
        n1, n2 = rng.sample(CONTENT["en"], 2)
        s = (f"The {rng.choice(EN_ADJ)} {n1} {rng.choice(EN_VERB)} the {n2} "
             f"{rng.choice(['in', 'for', 'with', 'during', 'through'])} "
             f"{rng.choice(['the', 'a', 'every'])} {rng.choice(EN_ADJ)} "
             f"{rng.choice(CONTENT['en'])}")
        extra = " ".join(rng.choices(LANG_WORDS["en"] + CONTENT["en"], k=rng.randint(2, 7)))
        return f"{s} {extra}."
    words = rng.choices(LANG_WORDS[lang], k=rng.randint(6, 10)) + \
        rng.choices(CONTENT[lang], k=rng.randint(2, 4))
    rng.shuffle(words)
    return " ".join(words).capitalize() + "."


def _paragraphs(rng, lang, n_par, n_sent=(3, 7)):
    return ["<p>" + " ".join(_sentence(rng, lang) for _ in range(rng.randint(*n_sent))) + "</p>"
            for _ in range(n_par)]


def _boilerplate(site: str):
    nav = (f'<nav><ul><li><a href="/">Home</a></li><li><a href="/about">About {site}</a></li>'
           '<li><a href="/contact">Contact</a></li><li><a href="/login">Sign in</a></li></ul></nav>')
    footer = (f"<footer><p>Copyright {site}. All rights reserved.</p>"
              "<p>Privacy policy | Terms of use | Cookie settings</p></footer>")
    return nav, footer


def _page(title, body_parts, site, scripts=True):
    nav, footer = _boilerplate(site)
    script = ("<script>var tracker = {id: 42}; function load(){return 1;}</script>"
              "<style>body{font-family:serif}</style>") if scripts else ""
    return (f"<!DOCTYPE html><html><head><title>{title}</title>{script}</head><body>"
            f"<header>{site}</header>{nav}<main><h1>{title}</h1>{''.join(body_parts)}</main>"
            f"{footer}</body></html>")


def _pii_block(rng: random.Random):
    user = rng.choice(["alice", "bob.smith", "carol_j", "dev-team", "info"])
    email = f"{user}@{rng.choice(['example.com', 'example.org', 'mail.example.net'])}"
    area = rng.choice(["212", "415", "512", "617"])
    phone = rng.choice([f"({area}) 555-01{rng.randint(10, 99)}",
                        f"{area}-555-01{rng.randint(10, 99)}",
                        f"+1 {area} 555 01{rng.randint(10, 99)}"])
    ip = rng.choice(["192.0.2.", "198.51.100.", "203.0.113."]) + str(rng.randint(1, 254))
    text = (f"<p>Questions? Write to {email} or call {phone}. "
            f"Our status server is reachable at {ip}.</p>")
    return text, {"email": 1, "phone": 1, "ip": 1}


LICENCE_TEMPLATE = (
    "The Lantern Public Licence. Copyright {year} {holder}. You may use, study, change and "
    "share this work, in source or compiled form, provided that this notice and the name of "
    "the holder are kept with every copy. The work is offered as it is, with no promise of "
    "fitness for any purpose, and the holder accepts no liability for any loss that comes "
    "from using it. Changed versions must say clearly that they were changed and by whom.")


def generate_corpus(n: int = 200, seed: int = 0, mix: Optional[Dict[str, float]] = None) -> List[SynthDoc]:
    """Generate ``n`` labelled pages. ``mix`` gives the fraction of each kind."""
    rng = random.Random(seed)
    mix = mix or {"article": 0.40, "foreign": 0.12, "spam": 0.12, "login": 0.06,
                  "listing": 0.08, "template": 0.12, "mirror": 0.10}
    kinds = list(mix)
    docs: List[SynthDoc] = []
    articles: List[SynthDoc] = []
    next_group = 0          # ids for duplicate groups (templates and mirrors share the counter)
    tmpl_group = None
    for i in range(n):
        kind = rng.choices(kinds, weights=[mix[k] for k in kinds])[0]
        if kind == "mirror" and not articles:
            kind = "article"
        site = rng.choice(SITES)
        url = f"https://{site}/page/{i}"
        if kind == "article":
            body = _paragraphs(rng, "en", rng.randint(3, 6))
            pii = {}
            if rng.random() < 0.4:
                block, pii = _pii_block(rng)
                body.append(block)
            charset = rng.choice(["utf-8", "utf-8", "utf-8", "cp1252"])
            if charset == "cp1252":
                body.append("<p>The café résumé was naïve but well received.</p>")
            d = SynthDoc(url, _page(f"On the {rng.choice(CONTENT['en'])}", body, site), kind,
                         "en", True, None, charset, rng.random() < 0.8, pii)
            articles.append(d)
        elif kind == "foreign":
            lang = rng.choice(["es", "de", "fr", "it"])
            d = SynthDoc(url, _page(rng.choice(CONTENT[lang]).title(),
                                    _paragraphs(rng, lang, rng.randint(3, 5)), site), kind, lang, False)
        elif kind == "spam":
            lines = [" ".join(rng.choices(SPAM, k=rng.randint(3, 6))) + "..." for _ in range(rng.randint(20, 40))]
            body = ["<div>" + "<br>".join(f"#{j} | {ln}" for j, ln in enumerate(lines)) + "</div>"]
            d = SynthDoc(url, _page("BEST DEALS", body, site), kind, "en", False)
        elif kind == "login":
            body = ['<form><label>Email</label><input name="u"><label>Password</label>'
                    '<input type="password"><button>Sign in</button></form><p>Forgot password?</p>']
            d = SynthDoc(url, _page("Sign in", body, site), kind, "en", False)
        elif kind == "listing":
            rows = [f"<li>SKU-{rng.randint(1000, 9999)} ${rng.randint(1, 999)}.{rng.randint(10, 99)} "
                    f"{rng.randint(1, 50)}x{rng.randint(1, 50)}cm #{rng.randint(1, 99)}</li>" for _ in range(40)]
            d = SynthDoc(url, _page("Catalogue", ["<ul>" + "".join(rows) + "</ul>"], site), kind, "en", False)
        elif kind == "template":
            if tmpl_group is None:          # all filled-in templates form one near-dup family
                next_group += 1
                tmpl_group = next_group
            holder = rng.choice(["Ada Lovelace", "Grace Hopper", "the Lantern Project", "J. Doe",
                                 "Example Labs", "Turing Systems"])
            text = LICENCE_TEMPLATE.format(year=rng.randint(1999, 2026), holder=holder)
            d = SynthDoc(url, _page("LICENCE", [f"<p>{text}</p>"], site), kind, "en", True, tmpl_group)
        else:  # mirror: identical article body under another URL
            src = rng.choice(articles)
            if src.group is None:
                next_group += 1
                src.group = next_group
            d = SynthDoc(url, src.html, kind, "en", True, src.group, src.charset, src.declare_charset,
                         dict(src.pii))
        docs.append(d)
    return docs


def to_warc_records(docs: List[SynthDoc]):
    from .records import make_response
    return [make_response(d.url, d.html, d.charset, d.declare_charset) for d in docs]


def language_training_set(n_per_lang: int = 200, seed: int = 1):
    """(text, lang) pairs for training the language identifier."""
    rng = random.Random(seed)
    data = []
    for lang in LANG_WORDS:
        for _ in range(n_per_lang):
            data.append((" ".join(_sentence(rng, lang) for _ in range(rng.randint(1, 3))), lang))
    rng.shuffle(data)
    return data


def quality_training_set(n: int = 400, seed: int = 2):
    """(text, label) pairs: 1 = reference-like prose, 0 = random-crawl-like text.

    Positives mimic "pages linked from a trusted source": well-formed prose.
    Negatives mimic a random crawl sample: spam, menus, listings, forms, fragments.
    """
    rng = random.Random(seed)
    data = []
    for _ in range(n // 2):
        data.append((" ".join(_sentence(rng, "en") for _ in range(rng.randint(3, 8))), 1))
        kind = rng.choice(["spam", "menu", "listing", "fragment"])
        if kind == "spam":
            t = " ".join(" ".join(rng.choices(SPAM, k=5)) + "..." for _ in range(rng.randint(3, 8)))
        elif kind == "menu":
            t = " | ".join(rng.choices(["Home", "About", "Contact", "Sign in", "Cart", "Help", "Blog"], k=8))
        elif kind == "listing":
            t = " ".join(f"SKU-{rng.randint(1000, 9999)} ${rng.randint(1, 99)}.99" for _ in range(8))
        else:
            t = " ".join(rng.choices(LANG_WORDS["en"], k=rng.randint(3, 12)))
        data.append((t, 0))
    rng.shuffle(data)
    return data
