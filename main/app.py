# app.py — Life Minus Work (Streamlit)

from __future__ import annotations
import os, json, re, hashlib, unicodedata, time, ssl, smtplib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from email.message import EmailMessage
from email.utils import formataddr

import streamlit as st
from fpdf import FPDF
from PIL import Image

# =============================================================================
# App Config & Constants
# =============================================================================

THEMES = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]

def _resolve_model(raw: Optional[str]) -> str:
    if not raw:
        return "gpt-5-mini"
    r = raw.strip().lower()
    alias_map = {
        "chatgpt 5 mini": "gpt-5-mini",
        "chatgpt-5-mini": "gpt-5-mini",
        "gpt 5 mini": "gpt-5-mini",
        "gpt5mini": "gpt-5-mini",
        "gpt-5 mini": "gpt-5-mini",
    }
    return alias_map.get(r, raw)

# Force GPT-5 mini, as requested
AI_MODEL = _resolve_model(st.secrets.get("LW_MODEL", os.getenv("LW_MODEL", "gpt-5-mini")))

# Output token caps to MATCH your working app
AI_MAX_TOKENS_CAP = 7000            # primary attempt for GPT-5 mini (via max_completion_tokens)
AI_MAX_TOKENS_FALLBACK = 6000       # retry budget if the first attempt errors

# Fixed Future Snapshot horizon (~1 month)
FUTURE_WEEKS_DEFAULT = 4

# Safe Mode: OFF by default so AI runs if key present
SAFE_MODE = os.getenv("LW_SAFE_MODE", st.secrets.get("LW_SAFE_MODE", "0")) == "1"

# Email / SMTP (Gmail App Password)
GMAIL_USER = st.secrets.get("GMAIL_USER", "whatisyourminus@gmail.com")
GMAIL_APP_PASSWORD = st.secrets.get("GMAIL_APP_PASSWORD", "")
SENDER_NAME = st.secrets.get("SENDER_NAME", "Life Minus Work")
REPLY_TO = st.secrets.get("REPLY_TO", "whatisyourminus@gmail.com")

# Dev helper: show codes onscreen if email not configured
DEV_SHOW_CODES = os.getenv("DEV_SHOW_CODES", st.secrets.get("DEV_SHOW_CODES", "0")) == "1"

# ---- Logo auto-detect (png/webp); optional override via LW_LOGO ----------------
LOGO_CANDIDATES = [
    os.getenv("LW_LOGO", "").strip(),
    "logo.png",
    "Life-Minus-Work-Logo.png",
    "Life-Minus-Work-Logo.webp",
]

def here() -> Path:
    return Path(__file__).parent

def find_logo_path() -> Optional[Path]:
    for name in LOGO_CANDIDATES:
        if not name:
            continue
        p = here() / name
        if p.exists():
            return p
    return None

# =============================================================================
# Questions Loading & Utilities
# =============================================================================

def load_questions(filename="questions.json") -> Tuple[List[dict], List[str]]:
    p = here() / filename
    if not p.exists():
        st.warning(f"{filename} not found at {p}. Using built-in fallback questions.")
        fallback = [
            {
                "id": "q1",
                "text": "I feel connected to a supportive community.",
                "choices": [
                    {"label": "Strongly disagree", "weights": {"Connection": 0}},
                    {"label": "Disagree", "weights": {"Connection": 1}},
                    {"label": "Neutral", "weights": {"Connection": 2}},
                    {"label": "Agree", "weights": {"Connection": 3}},
                    {"label": "Strongly agree", "weights": {"Connection": 4}},
                ],
            },
            {
                "id": "q2",
                "text": "I’m actively exploring new interests or skills.",
                "choices": [
                    {"label": "Strongly disagree", "weights": {"Growth": 0}},
                    {"label": "Disagree", "weights": {"Growth": 1}},
                    {"label": "Neutral", "weights": {"Growth": 2}},
                    {"label": "Agree", "weights": {"Growth": 3}},
                    {"label": "Strongly agree", "weights": {"Growth": 4}},
                ],
            },
        ]
        return fallback, THEMES
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "questions" in data:
            return data["questions"], data.get("themes", THEMES)
        elif isinstance(data, list):
            return data, THEMES
        else:
            raise ValueError("Unexpected questions format")
    except Exception as e:
        st.error(f"Could not parse {filename}: {e}")
        return [], THEMES

def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")
    return s.lower()

class PDF(FPDF):
    def header(self):
        pass

def setf(pdf: "PDF", style="", size=11):
    pdf.set_font("Helvetica", style=style, size=size)

# =============================================================================
# FPDF Latin-1 Safety
# =============================================================================

LATIN1_MAP = {
    "—": "-", "–": "-", "―": "-",
    "“": '"', "”": '"', "„": '"',
    "’": "'", "‘": "'", "‚": "'",
    "•": "-", "·": "-", "∙": "-",
    "…": "...",
    "□": "[ ]", "✓": "v", "✔": "v", "✗": "x", "✘": "x",
    "★": "*", "☆": "*", "█": "#", "■": "#", "▪": "-",
    "\u00a0": " ", "\u200b": "",
}

def to_latin1(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    t = unicodedata.normalize("NFKD", text)
    for k, v in LATIN1_MAP.items():
        t = t.replace(k, v)
    try:
        t = t.encode("latin-1", errors="ignore").decode("latin-1")
    except Exception:
        t = t.encode("ascii", errors="ignore").decode("ascii")
    t = re.sub(r"(\S{80})\S+", r"\1", t)
    return t

def mc(pdf: "PDF", text: str, h=6):
    pdf.multi_cell(0, h, to_latin1(text))

def sc(pdf: "PDF", w, h, text: str, ln=0):
    pdf.cell(w, h, to_latin1(text), ln=ln)

# =============================================================================
# PDF Drawing Helpers
# =============================================================================

def hr(pdf: "PDF", y_offset: float = 2.0, gray: int = 220):
    x1, x2 = 10, 200
    y = pdf.get_y() + y_offset
    pdf.set_draw_color(gray, gray, gray)
    pdf.line(x1, y, x2, y)
    pdf.ln(4)
    pdf.set_draw_color(0, 0, 0)

def checkbox_line(pdf: "PDF", text: str, line_h: float = 7.5):
    x = pdf.get_x()
    y = pdf.get_y()
    box = 4.6
    pdf.rect(x, y + 1.6, box, box)
    pdf.set_xy(x + box + 3, y)
    mc(pdf, text, h=line_h)

def draw_scores_barchart(pdf: "PDF", scores: Dict[str, int]):
    setf(pdf, "B", 14); mc(pdf, "Your Theme Snapshot", h=7); setf(pdf, "", 12)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    positive = [v for _, v in ordered if v > 0]
    max_pos = max(positive) if positive else 1
    left_x = pdf.get_x(); y = pdf.get_y()
    label_w = 40; bar_w_max = 118; bar_h = 5.0
    pdf.set_fill_color(33, 150, 243)
    for theme, val in ordered:
        pdf.set_xy(left_x, y); sc(pdf, label_w, 6, theme)
        bar_x = left_x + label_w + 2
        if val > 0:
            bar_w = (val / float(max_pos)) * bar_w_max
            pdf.rect(bar_x, y + 1.0, bar_w, bar_h, "F")
            num_x = bar_x + bar_w + 2.5
        else:
            num_x = bar_x + 2.5
        pdf.set_xy(num_x, y); sc(pdf, 0, 6, str(val))
        y += 7
    pdf.set_y(y + 2); hr(pdf, y_offset=0)

# =============================================================================
# Scoring
# =============================================================================

def compute_scores(questions: List[dict], answers_by_qid: Dict[str, str]) -> Dict[str, int]:
    scores = {t: 0 for t in THEMES}
    for q in questions:
        sel = answers_by_qid.get(q["id"])
        if not sel:
            continue
        for c in q["choices"]:
            if c["label"] == sel:
                for k, w in (c.get("weights") or {}).items():
                    scores[k] = scores.get(k, 0) + int(w or 0)
    return scores

def top_n_themes(scores: Dict[str, int], n=3) -> List[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]]

def choice_key(qid: str) -> str:
    return f"choice_{qid}"

def free_key(qid: str) -> str:
    return f"free_{qid}"

def q_version_hash(questions: List[dict]) -> str:
    s = json.dumps(questions, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

# =============================================================================
# Session State
# =============================================================================

def ensure_state(questions: List[dict]):
    ver = q_version_hash(questions)
    if "answers_by_qid" not in st.session_state:
        st.session_state["answers_by_qid"] = {}
    if "free_by_qid" not in st.session_state:
        st.session_state["free_by_qid"] = {}
    if st.session_state.get("q_version") != ver:
        old_a = st.session_state.get("answers_by_qid", {})
        old_f = st.session_state.get("free_by_qid", {})
        st.session_state["answers_by_qid"] = {q["id"]: old_a.get(q["id"], "") for q in questions}
        st.session_state["free_by_qid"] = {q["id"]: old_f.get(q["id"], "") for q in questions}
        st.session_state["q_version"] = ver

# =============================================================================
# OpenAI (lazy import) & AI helpers
# =============================================================================

def get_openai_client():
    key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    if not key or SAFE_MODE:
        return None
    try:
        from openai import OpenAI
        return OpenAI()
    except Exception as e:
        st.warning(f"OpenAI SDK unavailable: {e}")
        return None

def ai_enabled() -> bool:
    return get_openai_client() is not None

def _extract_json_blob(txt: str) -> str:
    try:
        start = txt.index("{"); end = txt.rindex("}")
        return txt[start:end+1]
    except Exception:
        return "{}"

AI_SCHEMA_FIELDS = (
    "{archetype, core_need, signature_metaphor, signature_sentence, "
    "top_themes:[], insights, why_now, future_snapshot, "
    "from_your_words:{summary, keepers:[]}, one_liners_to_keep:[], personal_pledge, "
    "what_this_really_says, signature_strengths:[], "
    "energy_map:{energizers:[], drainers:[]}, hidden_tensions:[], "
    "watch_out, actions_7d:[], impl_if_then:[], plan_1_week:[], "
    'balancing_opportunity:[], keep_in_view:[], tiny_progress:[]}'
)

def _fallback_ai(scores: Dict[str, int]) -> Dict[str, Any]:
    # Rich fallback shaped like target schema
    return {
        "archetype": "Bold Explorer",
        "core_need": "space to try new selves",
        "signature_metaphor": "a tango with possibility",
        "signature_sentence": "You learn by stepping forward and listening to the rhythm around you.",
        "top_themes": [t for t, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]],
        "insights": ("You move like someone who prefers the step to the map. Adventure leads; Growth and "
                     "Contribution need simple scaffolding so experiences become learning and trust."),
        "why_now": ("Early momentum is easier to shape than late habits. A few tiny anchors now will turn "
                    "spark into clarity and steady contribution."),
        "future_snapshot": ("One month from now you feel a quieter confidence. Small rituals after each adventure "
                            "leave a trail of learning and stronger bonds."),
        "from_your_words": {
            "summary": "Your notes point to rhythm, partnership, and small acts that make moments matter.",
            "keepers": ["try new steps", "listen before leading", "turn play into learning"],
        },
        "one_liners_to_keep": ["listen as much as lead", "small commitments build trust", "one step clarifies identity"],
        "personal_pledge": "I will try one new adventure and call someone afterward.",
        "what_this_really_says": ("You're energized by novelty and expression; add tiny routines that capture insight "
                                  "and follow-up so energy compounds into identity and contribution."),
        "signature_strengths": ["seeks novelty", "social rhythm", "authentic expression", "decisive action"],
        "energy_map": {
            "energizers": ["new experiences", "shared adventures", "creative risks", "spontaneous plans", "storytelling"],
            "drainers": ["endless routine", "vague obligations", "over-analysis", "long solitary stretches"],
        },
        "hidden_tensions": ["loves variety but avoids follow-through", "wants clarity yet resists structure", "connects intensely then withdraws"],
        "watch_out": "Don't let the thrill of novelty replace small acts of responsibility that build trust and growth.",
        "actions_7d": ["Plan one mini-adventure", "Invite someone to join", "Journal one identity insight"],
        "impl_if_then": [
            "If I crave novelty, then I will schedule a short experiment.",
            "If I avoid feedback, then I will ask one trusted person.",
            "If I feel restless, then I will reflect for five minutes.",
        ],
        "plan_1_week": [
            "Choose one mini-adventure this week",
            "Invite a friend to participate",
            "Do a five-minute reflection afterward",
            "Send a follow-up message within 48 hours",
            "Note one learning in a simple log",
            "Offer one small helpful action",
        ],
        "balancing_opportunity": ["Turn adventures into learning", "Translate experiences into small acts of contribution"],
        "keep_in_view": ["listen as much as lead", "small commitments build trust", "one step clarifies identity"],
        "tiny_progress": ["Tried one new thing", "Called someone after event", "Wrote one reflection"],
    }

# =============================================================================
# List normalizer to prevent per-character bullets
# =============================================================================

def as_list(v) -> List[str]:
    """Coerce any AI field into a clean list of bullet strings."""
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str):
        s = v.strip()
        parts = re.split(r"(?:\r?\n|\r|^\s*[-•–]\s+|[\u2022\u2023]\s+)", s)
        parts = [p.strip(" .;") for p in parts if p and p.strip()]
        return parts if len(parts) > 1 else [s]
    return []

# =============================================================================
# AI Runner — GPT-5 mini only, with 7000→6000 retry (max_completion_tokens)
# =============================================================================

def run_ai(first_name: str, horizon_weeks: int, scores: Dict[str, int], scores_free: Dict[str, str] | None = None):
    """
    Returns (ai_sections_dict, usage_dict, raw_text_head).
    Records st.session_state['effective_model'] and ['model_attempts'].
    """
    usage = {}
    prompt_ctx = {
        "first_name": first_name or "",
        "horizon_weeks": horizon_weeks,
        "scores": scores,
        "free": scores_free or {},
    }
    raw_text = json.dumps(prompt_ctx)[:800]

    client = get_openai_client()
    if client is None:
        st.session_state["effective_model"] = "(no key / safe mode)"
        st.session_state["model_attempts"] = [("none", "no-client")]
        return _fallback_ai(scores), usage, raw_text

    system_msg = (
        "You are a precise reflection coach.\n"
        f"Return ONLY a compact JSON object with fields {AI_SCHEMA_FIELDS}.\n"
        "Depth requirements:\n"
        "- Write substantial, concrete content (no fluff).\n"
        "- 'insights' and 'why_now': 2–3 dense paragraphs each (5–8 sentences per paragraph).\n"
        "- 'future_snapshot': 8–12 vivid sentences as a short postcard from ~1 month ahead.\n"
        "- Lists (actions_7d, impl_if_then, plan_1_week, strengths, energizers/drainers): 6–10 items each, specific and actionable.\n"
        "- All list fields MUST be JSON arrays of short strings (never a paragraph).\n"
        "- Keep language simple, humane, and encouraging.\n"
    )
    user_msg = (
        "Name: {name}\n"
        "Horizon: about {weeks} weeks (~1 month)\n"
        "Theme scores (higher = stronger energy): {scores}\n"
        "From user's notes (optional): {notes}\n"
        "Top themes should reflect the score order. Fill every field with specific, helpful content."
    ).format(
        name=first_name or "friend",
        weeks=horizon_weeks,
        scores=json.dumps(scores),
        notes=json.dumps(scores_free or {}, ensure_ascii=False),
    )

    # GPT-5 mini only; two passes with 7000 then 6000 completion tokens.
    model = AI_MODEL  # "gpt-5-mini"
    attempts: List[Tuple[str, str]] = []
    for max_out in (AI_MAX_TOKENS_CAP, AI_MAX_TOKENS_FALLBACK):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                # IMPORTANT: GPT-5 mini expects max_completion_tokens; don't send temperature.
                "max_completion_tokens": int(max_out),
            }
            st.session_state["param_used"] = "max_completion_tokens"
            resp = client.chat.completions.create(**kwargs)

            txt = (resp.choices[0].message.content or "").strip()
            blob = _extract_json_blob(txt)
            data = json.loads(blob)

            usage_obj = getattr(resp, "usage", None)
            if usage_obj:
                inp = getattr(usage_obj, "prompt_tokens", None)
                out = getattr(usage_obj, "completion_tokens", None) or getattr(usage_obj, "output_tokens", None)
                tot = getattr(usage_obj, "total_tokens", None)
                usage = {
                    "input": inp or 0,
                    "output": out or 0,
                    "total": tot or ((inp or 0) + (out or 0)),
                }

            if "top_themes" not in data or not isinstance(data["top_themes"], list):
                data["top_themes"] = [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]]
            for key in ["future_snapshot", "insights", "why_now", "what_this_really_says"]:
                if key not in data:
                    raise ValueError("Missing key: " + key)

            st.session_state["effective_model"] = model
            attempts.append((f"{model} @ {max_out}", "ok"))
            st.session_state["model_attempts"] = attempts
            return (data, usage, txt[:800])
        except Exception as e:
            attempts.append((f"{model} @ {max_out}", f"error: {e}"))
            continue

    st.warning("AI call fell back to safe content.")
    st.session_state["effective_model"] = "(fallback)"
    st.session_state["model_attempts"] = attempts
    return _fallback_ai(scores), usage, raw_text

    system_msg = (
        "You are a precise reflection coach.\n"
        f"Return ONLY a compact JSON object with fields {AI_SCHEMA_FIELDS}.\n"
        "Depth requirements:\n"
        "- Write substantial, concrete content (no fluff).\n"
        "- 'insights' and 'why_now': 2–3 dense paragraphs each (5–8 sentences per paragraph).\n"
        "- 'future_snapshot': 8–12 vivid sentences as a short postcard from ~1 month ahead.\n"
        "- Lists (actions_7d, impl_if_then, plan_1_week, strengths, energizers/drainers): 6–10 items each, specific and actionable.\n"
        "- All list fields MUST be JSON arrays of short strings (never a paragraph).\n"
        "- Keep language simple, humane, and encouraging.\n"
    )
    user_msg = (
        "Name: {name}\n"
        "Horizon: about {weeks} weeks (~1 month)\n"
        "Theme scores (higher = stronger energy): {scores}\n"
        "From user's notes (optional): {notes}\n"
        "Top themes should reflect the score order. Fill every field with specific, helpful content."
    ).format(
        name=first_name or "friend",
        weeks=horizon_weeks,
        scores=json.dumps(scores),
        notes=json.dumps(scores_free or {}, ensure_ascii=False),
    )

    model = AI_MODEL  # gpt-5-mini
    attempts: List[Tuple[str, str]] = []
    for max_out in (AI_MAX_TOKENS_CAP, AI_MAX_TOKENS_FALLBACK):
        try:
            kwargs = {
                "model": model,
                "temperature": 0.7,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                # GPT-5 family uses max_completion_tokens
                "max_completion_tokens": int(max_out),
            }
            st.session_state["param_used"] = "max_completion_tokens"
            resp = client.chat.completions.create(**kwargs)

            txt = (resp.choices[0].message.content or "").strip()
            blob = _extract_json_blob(txt)
            data = json.loads(blob)

            usage_obj = getattr(resp, "usage", None)
            if usage_obj:
                inp = getattr(usage_obj, "prompt_tokens", None)
                out = getattr(usage_obj, "completion_tokens", None) or getattr(usage_obj, "output_tokens", None)
                tot = getattr(usage_obj, "total_tokens", None)
                usage = {
                    "input": inp or 0,
                    "output": out or 0,
                    "total": tot or ((inp or 0) + (out or 0)),
                }

            if "top_themes" not in data or not isinstance(data["top_themes"], list):
                data["top_themes"] = [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]]
            for key in ["future_snapshot", "insights", "why_now", "what_this_really_says"]:
                if key not in data:
                    raise ValueError("Missing key: " + key)

            st.session_state["effective_model"] = model
            attempts.append((f"{model} @ {max_out}", "ok"))
            st.session_state["model_attempts"] = attempts
            return (data, usage, txt[:800])
        except Exception as e:
            attempts.append((f"{model} @ {max_out}", f"error: {e}"))
            continue

    st.warning("AI call fell back to safe content.")
    st.session_state["effective_model"] = "(fallback)"
    st.session_state["model_attempts"] = attempts
    return _fallback_ai(scores), usage, raw_text

# =============================================================================
# Email Helpers (SMTP via Gmail)
# =============================================================================

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def valid_email(e: str) -> bool:
    return bool(EMAIL_RE.match((e or "").strip()))

def send_email(to_addr: str, subject: str, text_body: str, html_body: str | None = None,
               attachments: list[tuple[str, bytes, str]] | None = None):
    if not (GMAIL_USER and GMAIL_APP_PASSWORD):
        raise RuntimeError("Email not configured: set GMAIL_USER and GMAIL_APP_PASSWORD in Streamlit secrets.")
    msg = EmailMessage()
    msg["From"] = formataddr((SENDER_NAME, GMAIL_USER))
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg["Reply-To"] = REPLY_TO
    msg.set_content(text_body or "")
    if html_body:
        msg.add_alternative(html_body, subtype="html")
    for (filename, data, mime_type) in (attachments or []):
        maintype, subtype = mime_type.split("/", 1)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
    context = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
        server.starttls(context=context)
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.send_message(msg)

def generate_code() -> str:
    return f"{int.from_bytes(os.urandom(3), 'big') % 1_000_000:06d}"

# =============================================================================
# PDF Builder (full layout like your “33” PDF) — using as_list() everywhere
# =============================================================================

def section_title(pdf: "PDF", title: str, size=14, top_margin=2):
    pdf.ln(top_margin)
    setf(pdf, "B", size); mc(pdf, title); setf(pdf, "", 11)

def bullets(pdf: "PDF", items: List[str]):
    for it in items:
        mc(pdf, f"- {it}")

def make_pdf_bytes(
    first_name: str,
    email: str,
    scores: Dict[str, int],
    ai: Dict[str, Any],
    logo_path: Optional[Path] = None
) -> bytes:
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    # Logo → title (auto-detect png/webp)
    y_after_logo = 18
    logo = logo_path or find_logo_path()
    logo_found = False
    if logo and Path(logo).exists():
        try:
            img = Image.open(logo).convert("RGBA")   # WEBP/PNG both OK
            tmp = here() / "_logo_tmp.png"
            img.save(tmp, format="PNG")
            pdf.image(str(tmp), x=10, y=10, w=28)
            y_after_logo = 26
            logo_found = True
        except Exception:
            y_after_logo = 20
    pdf.set_y(y_after_logo)

    # Header
    setf(pdf, "B", 18); mc(pdf, "Life Minus Work - Reflection Report", h=9)
    setf(pdf, "", 12); mc(pdf, f"Hi {first_name or 'there'},")
    hr(pdf)

    # Archetype cluster
    section_title(pdf, "Archetype"); mc(pdf, "A simple lens for your pattern.")
    setf(pdf, "B", 12); mc(pdf, ai.get("archetype", "-")); setf(pdf, "", 11)

    section_title(pdf, "Core Need"); mc(pdf, "The fuel that keeps your effort meaningful.")
    mc(pdf, ai.get("core_need", "-"))

    section_title(pdf, "Signature Metaphor"); mc(pdf, "A mental image to remember your mode.")
    mc(pdf, ai.get("signature_metaphor", "-"))

    section_title(pdf, "Signature Sentence"); mc(pdf, "One clean line to orient your week.")
    mc(pdf, ai.get("signature_sentence", "-"))
    hr(pdf)

    # Top Themes + chart
    section_title(pdf, "Top Themes")
    mc(pdf, "Where your energy is strongest right now.")
    top3 = ai.get("top_themes") or [k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]]
    if top3:
        setf(pdf, "B", 12); mc(pdf, ", ".join(top3)); setf(pdf, "", 11)
    draw_scores_barchart(pdf, scores)

    # From your words
    fyw = ai.get("from_your_words") or {}
    if fyw.get("summary"):
        section_title(pdf, "From your words")
        mc(pdf, "We pulled a few cues from what you typed.")
        mc(pdf, fyw["summary"])
        if fyw.get("keepers"):
            section_title(pdf, "One-liners to keep"); mc(pdf, "Tiny reminders that punch above their weight.")
            bullets(pdf, as_list(fyw["keepers"]))
        hr(pdf)

    # Personal pledge
    if ai.get("personal_pledge"):
        section_title(pdf, "Personal pledge"); mc(pdf, "Your simple promise to yourself.")
        mc(pdf, ai["personal_pledge"]); hr(pdf)

    # What this really says + Insights + Why Now + Future Snapshot
    if ai.get("what_this_really_says"):
        section_title(pdf, "What this really says about you")
        mc(pdf, ai["what_this_really_says"]); hr(pdf)

    if ai.get("insights"):
        section_title(pdf, "Insights"); mc(pdf, "A practical, encouraging synthesis of your answers.")
        mc(pdf, ai["insights"]); hr(pdf)

    if ai.get("why_now"):
        section_title(pdf, "Why Now"); mc(pdf, "Why these themes may be active in this season.")
        mc(pdf, ai["why_now"]); hr(pdf)

    if ai.get("future_snapshot"):
        section_title(pdf, "Future Snapshot"); mc(pdf, "A short postcard from 1 month ahead.")
        mc(pdf, ai["future_snapshot"]); hr(pdf)

    # Strengths & Energy map
    if ai.get("signature_strengths"):
        section_title(pdf, "Signature Strengths"); mc(pdf, "Traits to lean on when momentum matters.")
        bullets(pdf, as_list(ai.get("signature_strengths"))); hr(pdf)

    emap = ai.get("energy_map") or {}
    if emap.get("energizers") or emap.get("drainers"):
        section_title(pdf, "Energy Map"); mc(pdf, "Name what fuels you, and what quietly drains you.")
        if emap.get("energizers"):
            setf(pdf, "B", 11); mc(pdf, "Energizers"); setf(pdf, "", 11)
            bullets(pdf, as_list(emap.get("energizers")))
        if emap.get("drainers"):
            setf(pdf, "B", 11); mc(pdf, "Drainers"); setf(pdf, "", 11)
            bullets(pdf, as_list(emap.get("drainers")))
        hr(pdf)

    # Tensions & Watch-out
    if ai.get("hidden_tensions"):
        section_title(pdf, "Hidden Tensions"); mc(pdf, "Small frictions to watch with kindness.")
        bullets(pdf, as_list(ai.get("hidden_tensions"))); hr(pdf)

    if ai.get("watch_out"):
        section_title(pdf, "Watch-out (gentle blind spot)")
        mc(pdf, "A nudge to keep you steady.")
        mc(pdf, ai["watch_out"]); hr(pdf)

    # Actions & Plans
    if ai.get("actions_7d"):
        section_title(pdf, "3 Next-step Actions (7 days)"); mc(pdf, "Tiny moves that compound quickly.")
        bullets(pdf, as_list(ai.get("actions_7d"))); hr(pdf)

    if ai.get("impl_if_then"):
        section_title(pdf, "Implementation Intentions (If-Then)")
        mc(pdf, "Pre-decide responses to common bumps.")
        bullets(pdf, as_list(ai.get("impl_if_then"))); hr(pdf)

    if ai.get("plan_1_week"):
        section_title(pdf, "1-Week Gentle Plan"); mc(pdf, "A light structure you can actually follow.")
        bullets(pdf, as_list(ai.get("plan_1_week"))); hr(pdf)

    if ai.get("balancing_opportunity"):
        section_title(pdf, "Balancing Opportunity"); mc(pdf, "Low themes to tenderly rebalance.")
        bullets(pdf, as_list(ai.get("balancing_opportunity"))); hr(pdf)

    if ai.get("keep_in_view"):
        section_title(pdf, "Keep This In View"); mc(pdf, "Tiny reminders that protect progress.")
        bullets(pdf, as_list(ai.get("keep_in_view")))

    # Page 2: Signature Week + Tiny Progress
    pdf.add_page()
    setf(pdf, "B", 16); mc(pdf, "Signature Week - At a glance")
    setf(pdf, "", 11); mc(pdf, "A simple plan you can print or screenshot. Check items off as you go.")
    pdf.ln(2)
    for line in as_list(ai.get("plan_1_week")):
        checkbox_line(pdf, str(line))
    hr(pdf)

    setf(pdf, "B", 14); mc(pdf, "Tiny Progress Tracker")
    setf(pdf, "", 11); mc(pdf, "Three tiny milestones you can celebrate this week.")
    for t in as_list(ai.get("tiny_progress") or ["Tried one new thing", "Called someone after event", "Wrote one reflection"]):
        checkbox_line(pdf, str(t))

    pdf.ln(6); setf(pdf, "", 10); mc(pdf, f"Requested for: {email or '-'}")
    pdf.ln(6); setf(pdf, "", 9)
    mc(pdf, "Life Minus Work * This report is a starting point for reflection. Nothing here is medical or financial advice.")

    out = pdf.output(dest="S")
    if isinstance(out, str):
        out = out.encode("latin-1", errors="ignore")

    # Debug: record logo presence
    st.session_state["logo_found"] = bool(logo_found)
    st.session_state["logo_path"] = str(logo) if logo else "(none)"
    return out

# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="Life Minus Work — Questionnaire", page_icon="🧭", layout="centered")
st.title("Life Minus Work — Questionnaire")
st.caption("✅ App booted. If you see this, imports & first render succeeded.")

try:
    questions, _themes = load_questions("questions.json")
    ensure_state(questions)
    st.write(
        "Answer the questions, add your own reflections, and unlock a personalized PDF summary. "
        "**Desktop:** press Ctrl+Enter in text boxes to apply. **Mobile:** tap outside the box to save."
    )
    horizon_weeks = FUTURE_WEEKS_DEFAULT  # fixed 1-month snapshot
except Exception as e:
    st.error("The app hit an error while starting up.")
    st.exception(e)
    st.stop()

# Questionnaire
for i, q in enumerate(questions, start=1):
    st.subheader(f"Q{i}. {q['text']}")
    labels = [c["label"] for c in q["choices"]]
    WRITE_IN = "✍️ I'll write my own answer"
    labels_plus = labels + [WRITE_IN]
    prev = st.session_state["answers_by_qid"].get(q["id"])
    idx = labels_plus.index(prev) if prev in labels_plus else 0
    sel = st.radio("Pick one", labels_plus, index=idx, key=choice_key(q["id"]), label_visibility="collapsed")
    st.session_state["answers_by_qid"][q["id"]] = sel
    if sel == WRITE_IN:
        ta_key = free_key(q["id"])
        default_text = st.session_state["free_by_qid"].get(q["id"], "")
        new_text = st.text_area(
            "Your words (a sentence or two)",
            value=default_text,
            key=ta_key,
            placeholder="Type here... (on mobile, tap outside to save)",
            height=90,
        )
        st.session_state["free_by_qid"][q["id"]] = new_text or ""
    else:
        st.session_state["free_by_qid"].pop(q["id"], None)

st.divider()
# (Future Snapshot explainer removed earlier)

# Mini report trigger
with st.form("mini_form"):
    first_name = st.text_input("Your first name (for the report greeting)", key="first_name_input", placeholder="First name")
    submit_preview = st.form_submit_button("Show My Mini Report")
if submit_preview:
    st.session_state["preview_ready"] = True

# Mini Report + Gate
if st.session_state.get("preview_ready"):
    scores = compute_scores(questions, st.session_state["answers_by_qid"])
    top3 = top_n_themes(scores, 3)

    # keepers from free text
    free_texts = [txt.strip() for txt in (st.session_state.get("free_by_qid") or {}).values() if txt and txt.strip()]
    keepers = []
    for t in free_texts:
        for line in t.splitlines():
            s = line.strip()
            if 3 <= len(s) <= 80:
                keepers.append(s)
                if len(keepers) >= 3:
                    break
        if len(keepers) >= 3:
            break

    with st.container():
        st.subheader("Your Mini Report (Preview)")
        st.write(f"**Top themes:** {', '.join(top3) if top3 else '-'}")
        if scores:
            try:
                st.bar_chart({k: v for k, v in sorted(scores.items(), key=lambda kv: kv[0])})
            except Exception:
                items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
                st.table({"Theme": [k for k, _ in items], "Score": [v for _, v in items]})
        if keepers:
            st.markdown("**From your words:**")
            for k in keepers:
                st.markdown(f"- {k}")
        recs = []
        if "Connection" in top3:   recs.append("Invite someone for a 20-minute walk this week.")
        if "Growth" in top3:       recs.append("Schedule one 20-minute skill rep on your calendar.")
        if "Peace" in top3:        recs.append("Block two 15-minute quiet blocks—phone away.")
        if "Identity" in top3:     recs.append("Draft a 3-line purpose that feels true today.")
        if "Adventure" in top3:    recs.append("Plan one micro-adventure within 30 minutes from home.")
        if "Contribution" in top3: recs.append("Offer a 30-minute help session to someone this week.")
        if recs:
            st.markdown("**Tiny actions to try this week:**")
            for r in recs[:3]:
                st.markdown(f"- {r}")
        st.markdown("**Your next 7 days (teaser):**")
        for p in [
            "Mon: choose one lever and block 10 minutes",
            "Tue: do one 20-minute skill rep",
            "Wed: invite one person to join a quick activity",
        ]:
            st.markdown(f"- {p}")
        st.markdown("**What you’ll unlock with the full report:**")
        st.markdown(
            "- Your *postcard from 1 month ahead* (Future Snapshot)\n"
            "- Insights & Why Now (personalized narrative)\n"
            "- Signature strengths, Energy map, Hidden tensions & Watch-out\n"
            "- 3 actions + If-Then plan + 1-week gentle plan\n"
            "- Printable 'Signature Week' checklist page + Tiny progress tracker"
        )
    st.caption("Unlock your complete Reflection Report to see your postcard from 1 month ahead, insights, plan & checklist.")

    # Gate state
    if "verify_state" not in st.session_state:
        st.session_state.verify_state = "collect"  # collect -> sent -> verified
    if "pending_email" not in st.session_state:
        st.session_state.pending_email = ""
    if "pending_code" not in st.session_state:
        st.session_state.pending_code = ""
    if "code_issued_at" not in st.session_state:
        st.session_state.code_issued_at = 0.0
    if "last_send_ts" not in st.session_state:
        st.session_state.last_send_ts = 0.0

    st.divider()
    st.subheader("Unlock your complete Reflection Report")
    st.write("We’ll email a 6-digit code to verify it’s really you. No spam—ever.")
    st.caption("Heads up: generating your full report is intensive and may take up to a minute.")

    if st.session_state.verify_state == "collect":
        user_email = st.text_input("Your email", placeholder="you@example.com", key="gate_email")
        c1, c2 = st.columns([1, 1])
        with c1:
            send_code_btn = st.button("Email me a 6-digit code")
        with c2:
            st.caption("You’ll enter it here to unlock your full report (PDF included).")
        if send_code_btn:
            if not valid_email(user_email):
                st.error("Please enter a valid email address.")
            else:
                now = time.time()
                if now - st.session_state.last_send_ts < 25:
                    st.warning("Please wait a moment before requesting another code.")
                else:
                    code = generate_code()
                    st.session_state.pending_email = user_email.strip()
                    st.session_state.pending_code = code
                    st.session_state.code_issued_at = now
                    st.session_state.last_send_ts = now
                    try:
                        plain = f"Your Life Minus Work verification code is: {code}\nThis code expires in 10 minutes."
                        html = f"""
                        <p>Your Life Minus Work verification code is:</p>
                        <h2 style="letter-spacing:2px">{code}</h2>
                        <p>This code expires in 10 minutes.</p>
                        """
                        send_email(
                            to_addr=st.session_state.pending_email,
                            subject="Your Life Minus Work verification code",
                            text_body=plain,
                            html_body=html
                        )
                        st.success(f"We’ve emailed a code to {st.session_state.pending_email}.")
                        st.session_state.verify_state = "sent"
                        st.rerun()
                    except Exception as e:
                        if DEV_SHOW_CODES:
                            st.warning(f"(Dev Mode) Email not configured; on-screen code: **{code}**")
                            st.session_state.verify_state = "sent"
                            st.rerun()
                        else:
                            st.error(f"Couldn’t send the code. {e}")

    elif st.session_state.verify_state == "sent":
        st.info(f"Enter the 6-digit code we emailed to **{st.session_state.pending_email}**.")
        v = st.text_input("Verification code", max_chars=6)
        c1, c2 = st.columns([1, 1])
        with c1:
            verify_btn = st.button("Verify")
        with c2:
            resend = st.button("Resend code")
        if verify_btn:
            if time.time() - st.session_state.code_issued_at > 600:
                st.error("This code has expired. Please request a new one.")
            elif v.strip() == st.session_state.pending_code:
                st.success("Verified! Your full report is unlocked.")
                st.session_state.verify_state = "verified"
            else:
                st.error("That code didn’t match. Please try again.")
        if resend:
            now = time.time()
            if now - st.session_state.last_send_ts < 25:
                st.warning("Please wait a moment before requesting another code.")
            else:
                st.session_state.pending_code = generate_code()
                st.session_state.code_issued_at = now
                st.session_state.last_send_ts = now
                try:
                    send_email(
                        to_addr=st.session_state.pending_email,
                        subject="Your Life Minus Work verification code",
                        text_body=f"Your code is: {st.session_state.pending_code}\nThis code expires in 10 minutes.",
                        html_body=f"<p>Your code is:</p><h2>{st.session_state.pending_code}</h2><p>This code expires in 10 minutes.</p>"
                    )
                    st.success("We’ve sent a new code.")
                except Exception as e:
                    if DEV_SHOW_CODES:
                        st.warning(f"(Dev Mode) Email not configured; new on-screen code: **{st.session_state.pending_code}**")
                    else:
                        st.error(f"Couldn’t resend the code. {e}")

# Verified → generate once, then reuse
if st.session_state.get("verify_state") == "verified":
    answers_by_qid = st.session_state.get("answers_by_qid", {})
    free_by_qid = st.session_state.get("free_by_qid", {})
    key_payload = {
        "email": st.session_state.pending_email,
        "first_name": st.session_state.get("first_name_input", ""),
        "answers": answers_by_qid,
        "free": free_by_qid,
        "model": AI_MODEL,
    }
    report_key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
    need_build = (
        st.session_state.get("report_key") != report_key
        or st.session_state.get("pdf_bytes") is None
        or st.session_state.get("ai_sections") is None
    )
    if need_build:
        with st.status("Generating your report…", expanded=False) as s:
            scores_build = compute_scores(questions, answers_by_qid)
            free_responses = {qid: txt for qid, txt in (free_by_qid or {}).items() if txt and txt.strip()}
            ai_sections, usage, raw_head = run_ai(
                first_name=st.session_state.get("first_name_input", ""),
                horizon_weeks=FUTURE_WEEKS_DEFAULT,
                scores=scores_build,
                scores_free=free_responses,
            )
            s.update(label="Building PDF…")
            pdf_bytes = make_pdf_bytes(
                first_name=st.session_state.get("first_name_input", ""),
                email=st.session_state.pending_email,
                scores=scores_build,
                ai=ai_sections,
                logo_path=find_logo_path(),
            )
            st.session_state["report_key"] = report_key
            st.session_state["pdf_bytes"] = pdf_bytes
            st.session_state["ai_sections"] = ai_sections
            st.session_state["scores_final"] = scores_build
            st.session_state["ai_usage"] = usage
            st.session_state["raw_head"] = raw_head
            s.update(label="Report ready.", state="complete")

    # Render from cache
    st.success("Your email is verified.")
    st.subheader("Your Complete Reflection Report")
    st.write("Includes your postcard from **1 month ahead**, insights, plan & printable checklist.")
    pdf_bytes = st.session_state.get("pdf_bytes", b"")
    if pdf_bytes:
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="LifeMinusWork_Reflection_Report.pdf",
            mime="application/pdf",
        )
        with st.expander("Email me the PDF", expanded=False):
            if st.button("Send report to my email"):
                try:
                    send_email(
                        to_addr=st.session_state.pending_email,
                        subject="Your Life Minus Work — Reflection Report",
                        text_body="Your report is attached. Be kind to your future self. —Life Minus Work",
                        html_body="<p>Your report is attached. Be kind to your future self.<br>—Life Minus Work</p>",
                        attachments=[("LifeMinusWork_Reflection_Report.pdf", pdf_bytes, "application/pdf")],
                    )
                    st.success("We’ve emailed your report.")
                except Exception as e:
                    st.error(f"Could not email the PDF. {e}")

    # Debug
    with st.expander("AI status (debug)", expanded=False):
        st.write(f"AI enabled: {ai_enabled()}")
        st.write(f"Model: {AI_MODEL}")
        st.write(f"Max tokens: {AI_MAX_TOKENS_CAP} (fallback {AI_MAX_TOKENS_FALLBACK})")
        usage = st.session_state.get("ai_usage") or {}
        if usage:
            st.write(f"Token usage — input: {usage.get('input', 0)}, output: {usage.get('output', 0)}, total: {usage.get('total', 0)}")
        else:
            st.write("No usage reported by SDK.")
        st.write(f"Logo found: {st.session_state.get('logo_found', False)} at {st.session_state.get('logo_path', '(none)')}")
        att = st.session_state.get("model_attempts") or []
        if att:
            st.write("Model attempts:")
            for m, status in att:
                st.write(f"    {m}: {status}")
        st.text("Raw head (first 800 chars)"); st.code(st.session_state.get("raw_head") or "(empty)")
