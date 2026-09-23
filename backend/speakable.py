"""Turn an answer into what the TTS should say.

The display text keeps its citations and symbols (the UI shows citation chips);
the spoken text drops citations, reads maths as words, and strips markdown and
LaTeX punctuation that Piper would otherwise read out literally."""
from __future__ import annotations

import re

CITATION = re.compile(r"\s*\([^()]*\bpp?\.\s*\d[^()]*\)")   # (doc, p. 6), (doc, pp. 12–17)
MATH = re.compile(r"\$\$?(.+?)\$\$?")

LATEX = [
    (r"\\bar\{(\w+)\}", r"\1 bar"),
    (r"\\hat\{(\w+)\}", r"\1 hat"),
    (r"\\frac\{([^}]*)\}\{([^}]*)\}", r"\1 over \2"),
    (r"\\sqrt\{([^}]*)\}", r"square root of \1"),
    (r"\^\{?2\}?", " squared"),
    (r"\^\{?(\w+)\}?", r" to the power \1"),
    (r"_\{?(\w+)\}?", r" \1"),
    (r"\\(sigma|mu|sum|alpha|beta|lambda|theta|pi)", r" \1 "),
]
OPS = {"-": " minus ", "+": " plus ", "=": " equals ", "/": " over ",
       "<": " less than ", ">": " greater than "}
SYMBOLS = {"σ": " sigma", "μ": " mu", "Σ": " sum of ", "√": " square root of ",
           "²": " squared", "≈": " approximately ", "≤": " at most ",
           "≥": " at least ", "−": " minus ", "×": " times ", "÷": " divided by "}


def _math_to_words(m: re.Match) -> str:
    s = m.group(1)
    for pat, rep in LATEX:
        s = re.sub(pat, rep, s)
    for op, word in OPS.items():
        s = s.replace(op, word)
    return re.sub(r"[\\{}]", "", s)


def speakable(text: str) -> str:
    text = CITATION.sub("", text)
    text = MATH.sub(_math_to_words, text)
    text = re.sub(r"(\w)\u0304", r"\1 bar", text)          # x̄ (combining macron)
    for sym, word in SYMBOLS.items():
        text = text.replace(sym, word)
    text = re.sub(r"[*_#`$\\{}]", " ", text)                # leftover markdown/LaTeX
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", text))
    return re.sub(r"\s+([.,;:!?])", r"\1", text).strip()
