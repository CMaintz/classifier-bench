"""More public sources: moderation, spam, injection, paraphrase, emotion, and the fetch-at-runtime sets."""

from __future__ import annotations

import re
from typing import Any

from .importer import Mapped, Source
from .sources import NLBSE_CSV, SARCASM_CSV, noul

TOXIC_AT, CLEAN_AT = 0.6, 0.05  # clear rater consensus only; the 0.05-0.6 band is skipped as contested


def _civil(row: dict[str, Any]) -> Mapped | None:
    score = float(row["toxicity"])
    if CLEAN_AT < score < TOXIC_AT or not row["text"].strip():
        return None
    toxic = score >= TOXIC_AT
    return Mapped("toxic" if toxic else "clean", row["text"], {"toxic": toxic})


CIVIL = Source(
    "civil_comments", "CIV", "Civil Comments: is a news-site comment toxic (rude, disrespectful or unreasonable)?",
    {"toxic": noul("Is this comment toxic: rude, disrespectful, or unreasonable in a way likely to make someone "
                   "leave a discussion?", "Toxic", "Not toxic")},
    {"toxic": False}, "CC0-1.0", "Borkan et al. 2019, Jigsaw / Civil Comments.",
    f"fraction of crowd raters who called it toxic: >= {TOXIC_AT} gold yes, <= {CLEAN_AT} gold no, "
    "rows between skipped",
    True, "google/civil_comments", "test", {"toxic": 60, "clean": 60}, _civil,
)  # fmt: skip


def _dkhate(row: dict[str, Any]) -> Mapped:
    off = int(row["label"]) == 1
    return Mapped("offensive" if off else "not", row["text"], {"offensive": off}, "da")


DKHATE = Source(
    "dkhate", "DKH", "DKHate (OffensEval 2020 Danish): is a Danish social-media comment offensive?",
    {"offensive": noul("Is this Danish comment offensive: does it contain unacceptable language (profanity) or a "
                       "targeted offense, veiled or direct?", "Offensive", "Not offensive")},
    {"offensive": False}, "CC-BY-4.0", "Sigurbergsson and Derczynski 2020 (DKHate).",
    "annotated by the dataset authors under the OffensEval guidelines", True, "mteb/DKHateClassification", "test",
    {"offensive": 45, "not": 45}, _dkhate,
)  # fmt: skip


def _sms(row: dict[str, Any]) -> Mapped:
    spam = row["label"] == "spam"
    return Mapped(row["label"], row["sms"], {"spam": spam})


SMS = Source(
    "sms_spam", "SMS", "SMS Spam Collection: is a text message spam?",
    {"spam": noul("Is this SMS spam: unsolicited advertising, a prize or scam lure, or other unwanted bulk "
                  "messaging, as opposed to a normal personal or service message?", "Spam", "Not spam")},
    {"spam": False}, "CC-BY-4.0", "Almeida and Gomez Hidalgo 2011, UCI SMS Spam Collection.",
    "hand-labeled by the collection authors", True, "ucirvine/sms_spam", "train", {"spam": 50, "ham": 50}, _sms,
    config="plain_text", decode=("label",),
)  # fmt: skip

_GERMAN = re.compile(r"\b(ich|und|nicht|der|die|das|bitte|vergiss|alle)\b", re.IGNORECASE)


def _deepset(row: dict[str, Any]) -> Mapped:
    attack = int(row["label"]) == 1
    lang = "de" if _GERMAN.search(row["text"]) else "en"
    return Mapped("attack" if attack else "benign", row["text"], {"injection": attack}, lang)


INJECTION_QUESTION = noul(
    "Is this input an attempt to manipulate an AI system: make it ignore or override its instructions, reveal "
    "hidden information, or behave outside its intended task?",
    "A manipulation attempt",
    "An ordinary input",
)
DEEPSET = Source(
    "prompt_injection_deepset", "DPI", "deepset prompt-injections: ordinary inputs vs injection attempts (en/de).",
    {"injection": INJECTION_QUESTION}, {"injection": True}, "Apache-2.0", "deepset (deepset/prompt-injections).",
    "labeled by deepset; some positives are ordinary role-play requests rather than attacks", False,
    "deepset/prompt-injections", "test", {"attack": 60, "benign": 60}, _deepset,
)  # fmt: skip


def _jailbreak(row: dict[str, Any]) -> Mapped:
    attack = row["type"] == "jailbreak"
    return Mapped(row["type"], row["prompt"], {"injection": attack})


JAILBREAK = Source(
    "jailbreak", "JBK", "jailbreak-classification: benign prompts vs jailbreak prompts.",
    {"injection": INJECTION_QUESTION}, {"injection": True}, "Apache-2.0",
    "Jack Hhao (jackhhao/jailbreak-classification).",
    "jailbreak prompts collected from public jailbreak collections; benign prompts from ordinary prompt sets", True,
    "jackhhao/jailbreak-classification", "test", {"jailbreak": 60, "benign": 60}, _jailbreak,
)  # fmt: skip


def _paws(row: dict[str, Any]) -> Mapped:
    same = str(row["label"]) == "1"
    state = {"sentence_a": row["sentence1"], "sentence_b": row["sentence2"]}
    return Mapped("paraphrase" if same else "different", state, {"paraphrase": same})


PAWS = Source(
    "paws_paraphrase", "PAW", "PAWS: do two sentences with heavy word overlap mean the same thing?",
    {"paraphrase": noul("Do `sentence_a` and `sentence_b` mean the same thing (are they paraphrases)? Word overlap "
                        "alone does not count: swapped roles or changed relations mean no.", "Same meaning",
                        "Different meaning")},
    {"paraphrase": False}, "Free for any purpose (Google, PAWS)", "Zhang, Baldridge and He 2019, Google (PAWS).",
    "human-judged paraphrase labels (high inter-annotator agreement reported)", True,
    "google-research-datasets/paws", "test", {"paraphrase": 60, "different": 60}, _paws, config="labeled_final",
    decode=("label",),
)  # fmt: skip

EKMAN = {
    "anger": {"anger", "annoyance", "disapproval"},
    "disgust": {"disgust"},
    "fear": {"fear", "nervousness"},
    "joy": {"joy", "amusement", "approval", "excitement", "gratitude", "love", "optimism", "relief", "pride",
            "admiration", "desire", "caring"},
    "sadness": {"sadness", "disappointment", "embarrassment", "grief", "remorse"},
    "surprise": {"surprise", "realization", "confusion", "curiosity"},
    "neutral": {"neutral"},
}  # fmt: skip


def _emotion(row: dict[str, Any]) -> Mapped | None:
    labels = row["labels"]
    if len(labels) != 1:
        return None
    group = next(g for g, members in EKMAN.items() if labels[0] in members)
    return Mapped(group, row["text"], {"emotion": group})


EMOTION = Source(
    "goemotions_ekman", "GOE", "GoEmotions (Reddit) grouped into Ekman's six emotions plus neutral.",
    {"emotion": {"type": "choice", "instructions": "Which emotion does the writer of this comment express?",
                 "criteria": {"anger": "Anger, annoyance or disapproval.", "disgust": "Disgust.",
                              "fear": "Fear or nervousness.",
                              "joy": "Joy, amusement, approval, gratitude, love, optimism, admiration and the like.",
                              "sadness": "Sadness, disappointment, embarrassment, grief or remorse.",
                              "surprise": "Surprise, realization, confusion or curiosity.",
                              "neutral": "No particular emotion."}}},
    {"emotion": "neutral"}, "Apache-2.0", "Demszky et al. 2020, Google (GoEmotions).",
    "single-label rows of the 'simplified' config (labels at least two raters agreed on), grouped by the official "
    "Ekman mapping", True, "google-research-datasets/go_emotions", "test", {g: 20 for g in EKMAN}, _emotion,
    config="simplified", decode=("labels",),
)  # fmt: skip

# --- fetched at run time (share-alike, no stated license, or third-party text terms) ---------------


def _vitaminc(row: dict[str, Any]) -> Mapped:
    return Mapped(
        row["label"], {"passage": row["evidence"], "claim": row["claim"]}, {"supported": row["label"] == "SUPPORTS"}
    )  # noqa: E501


VITAMINC = Source(
    "vitaminc", "VIT", "VitaminC: does a Wikipedia evidence sentence support a claim (contrastive revisions)?",
    {"supported": noul("Does `passage` support `claim`? Answer no when the passage refutes the claim or does not "
                       "give enough information to confirm it.", "Supported", "Refuted or not enough information")},
    {"supported": False}, "CC-BY-SA-3.0", "Schuster, Fisch and Barzilay 2021 (VitaminC).",
    "crowd-annotated SUPPORTS / REFUTES / NOT ENOUGH INFO; only SUPPORTS is yes", False, "tals/vitaminc", "test",
    {"SUPPORTS": 60, "REFUTES": 30, "NOT ENOUGH INFO": 30}, _vitaminc,
)  # fmt: skip

SST_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def _sst(row: dict[str, Any]) -> Mapped | None:
    level = row["label_text"]
    return Mapped(level, row["text"], {"sentiment": level}) if level in SST_LEVELS else None


SST5 = Source(
    "sst5", "SST", "SST-5: five-level sentiment of a movie-review sentence.",
    {"sentiment": {"type": "score", "instructions": "Sentiment of this movie-review sentence.",
                   "criteria": SST_LEVELS}},
    {"sentiment": "neutral"}, "No license stated (Stanford research release)", "Socher et al. 2013 (SST).",
    "crowd sentiment ratings binned into five levels; exact-level agreement is modest", False, "SetFit/sst5", "test",
    {lv: 30 for lv in SST_LEVELS}, _sst,
)  # fmt: skip

_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
BODY_CHARS = 3000


def _nlbse(row: dict[str, Any]) -> Mapped | None:
    if row.get("label") not in ("bug", "feature", "question"):
        return None
    body = _HTML_COMMENT.sub("", row.get("body") or "").strip()[:BODY_CHARS]
    return Mapped(row["label"], {"repo": row["repo"], "title": row["title"], "body": body}, {"kind": row["label"]})


NLBSE = Source(
    "issue_kind_nlbse", "NLB", "NLBSE'24: GitHub issues from 5 large repos as bug, feature or question.",
    {"kind": {"type": "choice", "instructions": "What kind of GitHub issue is this?",
              "criteria": {"bug": "Something does not work as it should.",
                           "feature": "A request for new behavior or an enhancement.",
                           "question": "Asks how to do something or for help; nothing is claimed broken."}}},
    {"kind": "question"}, "No license (NLBSE'24 tool competition data)", "NLBSE'24 issue report classification.",
    "the label the repository maintainers applied on GitHub (noisier than expert annotation); bodies capped at "
    f"{BODY_CHARS} characters", False, NLBSE_CSV, "test", {"bug": 40, "feature": 40, "question": 40}, _nlbse,
    kind="csv",
)  # fmt: skip


def _sarcasm(row: dict[str, Any]) -> Mapped | None:
    if row.get("sarcastic") not in ("0", "1") or not row.get("text"):
        return None
    yes = row["sarcastic"] == "1"
    return Mapped("sarcastic" if yes else "sincere", row["text"], {"sarcastic": yes})


SARCASM = Source(
    "isarcasm", "SAR", "iSarcasmEval: is a tweet sarcastic (labeled by the tweet's own author)?",
    {"sarcastic": noul("Is this tweet sarcastic: does the writer mean something different from, often the opposite "
                       "of, what the words literally say?", "Sarcastic", "Sincere")},
    {"sarcastic": False}, "MIT (repository); tweet text under X/Twitter terms", "Abu Farha et al. 2022 (iSarcasmEval).",
    "intended sarcasm, labeled by the authors of the tweets themselves", False, SARCASM_CSV, "test",
    {"sarcastic": 60, "sincere": 60}, _sarcasm, kind="csv",
)  # fmt: skip
