"""Public datasets the bench can import, each asked with its own original annotation definition.

The question for an imported task restates the definition the dataset's annotators labeled
against, so the published gold actually answers the question the classifiers are asked.
Sources marked redistributable=False are fetched at run time and never committed.
"""

from __future__ import annotations

import random
from typing import Any

from .hub import PAGE, Hub
from .importer import Mapped, Picked, Source

HF = "huggingface.co/datasets/"
NLBSE_CSV = "https://raw.githubusercontent.com/nlbse2024/issue-report-classification/2927bc67eb42db8affd16eaf3e5a6d74f3063961/data/issues_test.csv"  # noqa: E501
SARCASM_CSV = "https://raw.githubusercontent.com/iabufarha/iSarcasmEval/dfc708b53bde1bb571abfb5692f63231c2232195/test/task_A_En_test.csv"  # noqa: E501


def noul(instructions: str, yes: str, no: str) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions, "criteria": {"true": yes, "false": no}}


def yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


# --- BANKING77 -------------------------------------------------------------------------------


def _banking77(row: dict[str, Any]) -> Mapped:
    return Mapped(row["label"], row["text"], {"intent": row["label"]})


def banking77(names: list[str]) -> Source:
    question = {
        "type": "choice",
        "instructions": "Which banking customer-service intent does this customer message express?",
        "criteria": {n: n.replace("_", " ") for n in names},
    }
    return Source(
        "banking77", "B77", "BANKING77: 77 fine-grained intents from an online-banking support domain.",
        {"intent": question}, {"intent": names[0]}, "CC-BY-4.0", "Casanueva et al. 2020, PolyAI (BANKING77).",
        "intent annotated by the dataset authors", True, "legacy-datasets/banking77", "test",
        {n: 2 for n in names}, _banking77, decode=("label",),
    )  # fmt: skip


# --- CLINC150: in scope for a bank assistant ----------------------------------------------------

BANKING_INTENTS = frozenset(
    [
        "transfer",
        "transactions",
        "balance",
        "freeze_account",
        "pay_bill",
        "bill_balance",
        "bill_due",
        "interest_rate",
        "routing",
        "min_payment",
        "order_checks",
        "pin_change",
        "report_fraud",
        "account_blocked",
        "spending_history",
        "credit_score",
        "report_lost_card",
        "credit_limit",
        "rewards_balance",
        "new_card",
        "application_status",
        "card_declined",
        "international_fees",
        "apr",
        "redeem_rewards",
        "credit_limit_change",
        "damaged_card",
        "replacement_card_duration",
        "improve_credit_score",
        "expiration_date",
    ]
)


def _clinc(row: dict[str, Any]) -> Mapped:
    intent = row["intent"]
    stratum = "banking" if intent in BANKING_INTENTS else ("oos" if intent == "oos" else "other_domain")
    return Mapped(stratum, row["text"], {"in_scope": intent in BANKING_INTENTS})


CLINC = Source(
    "clinc_bank_scope", "CLS", "CLINC150: is a request in the banking or credit-card domain, or out of scope?",
    {"in_scope": noul(
        "Is this request about banking or credit cards (accounts, transfers, bills, cards, credit, rewards), "
        "so that a bank's assistant should handle it?",
        "A banking or credit-card request", "Anything else, including requests no assistant domain covers")},
    {"in_scope": False}, "CC-BY-3.0", "Larson et al. 2019, CLINC150 (clinc/oos-eval).",
    "intent and domain annotated by the dataset authors; 'oos' rows are written to be out of scope",
    True, "clinc/clinc_oos", "test", {"banking": 60, "oos": 30, "other_domain": 30}, _clinc, config="plus",
    decode=("intent",),
)  # fmt: skip

# --- MASSIVE: Danish scenario ---------------------------------------------------------------------

SCENARIOS = {
    "alarm": "Set, query or remove alarms.",
    "audio": "Device volume: mute, louder, quieter.",
    "calendar": "Set, query or remove calendar events and reminders.",
    "cooking": "Recipes and cooking questions.",
    "datetime": "The date or time, time zones and conversions between them.",
    "email": "Send, query or manage email and contacts.",
    "general": "Chit-chat or commands to the assistant itself: jokes, repeat, confirm, quirky remarks.",
    "iot": "Smart-home devices: lights, plugs, vacuum, coffee machine.",
    "lists": "To-do and shopping lists.",
    "music": "Music preferences, settings and questions about music.",
    "news": "News queries.",
    "play": "Play music, radio, podcasts, audiobooks or games.",
    "qa": "Factual questions: definitions, facts, currency, maths, stocks.",
    "recommendation": "Recommendations for events, places or movies.",
    "social": "Social media posts and queries.",
    "takeaway": "Order or ask about takeaway food.",
    "transport": "Tickets, taxis, traffic and train or bus queries.",
    "weather": "Weather questions.",
}


def _massive_scenario(row: dict[str, Any]) -> Mapped | None:
    label = row["label_text"]
    return Mapped(label, row["text"], {"scenario": label}, "da") if label in SCENARIOS else None


MASSIVE_DA = Source(
    "massive_scenario_da", "MSD", "MASSIVE (Danish): which voice-assistant domain a request belongs to.",
    {"scenario": {"type": "choice", "instructions": "Which voice-assistant domain does this Danish request belong to?",
                  "criteria": SCENARIOS}},
    {"scenario": "general"}, "CC-BY-4.0", "FitzGerald et al. 2022, Amazon (MASSIVE).",
    "intent annotated in English and carried over by professional translators", True,
    "mteb/amazon_massive_scenario", "test", {s: 7 for s in SCENARIOS}, _massive_scenario, config="da",
)  # fmt: skip

# --- MASSIVE: language identification (gold = the locale) -----------------------------------------

LANGUAGE_QUESTION = {
    "type": "choice",
    "instructions": "Which language is this text written in?",
    "criteria": {"en": "English", "da": "Danish", "no": "Norwegian (Bokmal)", "sv": "Swedish", "de": "German",
                 "nl": "Dutch", "other": "Any other language"},
}  # fmt: skip
LOCALES = {
    "en": "en",
    "da": "da",
    "nb": "no",
    "sv": "sv",
    "de": "de",
    "nl": "nl",
    "fi": "other",
    "is": "other",
    "fr": "other",
    "pl": "other",
}  # noqa: E501
SCANDINAVIAN = ("da", "nb", "sv")
MIN_WORDS = 4


def _decidable(locale: str, texts: dict[str, str]) -> bool:
    """At least 4 words, and for da/nb/sv a text that differs from the parallel sentence in the other two."""
    text = texts[locale].strip().lower()
    if len(text.split()) < MIN_WORDS:
        return False
    return locale not in SCANDINAVIAN or all(texts[o].strip().lower() != text for o in SCANDINAVIAN if o != locale)


def _language_rows(hub: Hub, offset: int) -> list[dict[str, dict[str, Any]]]:
    pages = {loc: hub.page("mteb/amazon_massive_intent", loc, "test", offset)[0] for loc in LOCALES}
    n = min(len(rows) for rows in pages.values())
    return [
        {loc: pages[loc][i] for loc in LOCALES} for i in range(n) if len({pages[loc][i]["id"] for loc in LOCALES}) == 1
    ]


def _assign(group: dict[str, dict[str, Any]], need: dict[str, int], rng: random.Random) -> tuple[str, str] | None:
    texts = {loc: row["text"] for loc, row in group.items()}
    options = [loc for loc in LOCALES if need[LOCALES[loc]] > 0 and _decidable(loc, texts)]
    if not options:
        return None
    loc = rng.choice(options)
    need[LOCALES[loc]] -= 1
    return loc, texts[loc]


def sample_languages(hub: Hub, seed: int, per_label: int = 25) -> list[Picked]:
    """Each MASSIVE utterance id is used once, in one language, so no sentence repeats across languages."""
    rng = random.Random(f"language_id_public:{seed}")
    total = hub.page("mteb/amazon_massive_intent", "en", "test", 0)[1]
    pages = list(range(-(-total // PAGE)))
    rng.shuffle(pages)
    need = {code: per_label for code in set(LOCALES.values())}
    picked: list[Picked] = []
    for page in pages:
        for group in _language_rows(hub, page * PAGE):
            hit = _assign(group, need, rng)
            if hit:
                gold = LOCALES[hit[0]]
                picked.append(Picked(int(group["en"]["id"]), Mapped(gold, hit[1], {"language": gold}, hit[0])))
        if not any(need.values()):
            break
    return picked


LANGUAGES = Source(
    "language_id_public", "LIP", "MASSIVE utterances in 10 locales: identify the language (gold is the locale).",
    {"language": LANGUAGE_QUESTION}, {"language": "other"}, "CC-BY-4.0", "FitzGerald et al. 2022, Amazon (MASSIVE).",
    "the locale the professional translation was written for; Danish, Norwegian and Swedish texts identical to "
    "their parallel sentence were excluded", True, "mteb/amazon_massive_intent", "test",
    {code: 25 for code in set(LOCALES.values())}, lambda row: None, sampler=sample_languages,
)  # fmt: skip
