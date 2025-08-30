# app.py — Life Minus Work (Streamlit, long-form version)
# -----------------------------------------------------------------------------
# This version is intentionally verbose and extensively commented.
# It implements:
#   - Future Snapshot fixed to "1 month ahead" (no horizon slider anywhere)
#   - A rich "Mini Report" preview that appears immediately
#   - A gated "Full Report" unlocked by verifying a real email (6-digit code)
#   - Email delivery via Gmail SMTP with App Password (or Dev mode that shows codes)
#   - Safe Mode for AI so Streamlit Cloud won't hang if outbound calls fail
#   - PDF generation with FPDF (2 pages, bar chart, coaching blocks)
#   - Resilient question loading so the app always renders
#
# Setup notes:
#   1) requirements.txt (you already have a good one)
#      streamlit==1.36.0
#      fpdf==1.7.2
#      Pillow>=10.3.0
#      openai>=1.60.0
#
#   2) Streamlit Secrets (App → Settings → Secrets):
#      GMAIL_USER = "whatisyourminus@gmail.com"        # or lifeminuswork@gmail.com
#      GMAIL_APP_PASSWORD = "xxxxxxxxxxxxxxxx"         # 16-char App Password
#      SENDER_NAME = "Life Minus Work"
#      REPLY_TO = "whatisyourminus@gmail.com"
#      LW_SAFE_MODE = "1"                               # default ON (no AI calls)
#      OPENAI_API_KEY = "sk-..."                        # needed when LW_SAFE_MODE="0"
#      DEV_SHOW_CODES = "0"                             # set "1" to show codes onscreen during testing
#
# Security note:
#   - Never put real passwords in the code file.
#   - Use Streamlit Secrets for credentials and env toggles.
# -----------------------------------------------------------------------------

from __future__ import annotations

# ---- Standard library imports
import os
import re
import ssl
import json
import time
import smtplib
import hashlib
import textwrap
import unicodedata
from email.message import EmailMessage
from email.utils import formataddr
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# ---- Third-party imports (from requirements.txt)
import streamlit as st
from fpdf import FPDF
from PIL import Image

# -----------------------------------------------------------------------------
# Global configuration and app-wide constants
# -----------------------------------------------------------------------------

# The six Life Minus Work themes used for scoring and display.
THEMES: List[str] = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]

# OpenAI model + token caps; these only matter when AI is enabled.
AI_MODEL: str = "gpt-5-mini"
AI_MAX_TOKENS_CAP: int = 8000
AI_MAX_TOKENS_FALLBACK: int = 7000

# Future Snapshot horizon: fixed internally to 4 weeks,
# but all user-facing language is "1 month ahead".
FUTURE_WEEKS_DEFAULT: int = 4

# Safe Mode:
# - When ON ("1"), we DO NOT call OpenAI—prevents hanging on Streamlit Cloud
#   if the key/network isn't configured. Turn OFF by setting "0" in Secrets.
SAFE_MODE: bool = os.getenv("LW_SAFE_MODE", st.secrets.get("LW_SAFE_MODE", "1")) == "1"

# Email (Gmail SMTP via App Password). Defaults point to your testing address.
# Override via Streamlit Secrets in production.
GMAIL_USER: str = st.secrets.get("GMAIL_USER", "whatisyourminus@gmail.com")
GMAIL_APP_PASSWORD: str = st.secrets.get("GMAIL_APP_PASSWORD", "")
SENDER_NAME: str = st.secrets.get("SENDER_NAME", "Life Minus Work")
REPLY_TO: str = st.secrets.get("REPLY_TO", GMAIL_USER)

# Developer helper: if True, verification codes are also shown on-screen
# (useful if SMTP isn't configured yet).
DEV_SHOW_CODES: bool = os.getenv("DEV_SHOW_CODES", st.secrets.get("DEV_SHOW_CODES", "0")) == "1"

# Attempt to import OpenAI SDK. We do not error if it's missing—Safe Mode can keep going.
OPENAI_OK: bool = False
try:
    from openai import OpenAI  # OpenAI Python SDK (>=1.0)
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

# -----------------------------------------------------------------------------
# Utilities: filesystem path, question loading, session keys
# -----------------------------------------------------------------------------

def here() -> Path:
    """Return the directory containing this app.py."""
    return Path(__file__).parent


def load_questions(filename: str = "questions.json") -> Tuple[List[dict], List[str]]:
    """
    Attempt to load the primary questionnaire from questions.json (next to app.py).
    If missing or unreadable, use a tiny fallback so the app ALWAYS renders.
    """
    path = here() / filename
    if not path.exists():
        # Fallback questions ensure the app can be demoed even if JSON isn't deployed yet.
        st.warning(f"{filename} not found at {path}. Using built-in fallback questions.")
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
            {
                "id": "q3",
                "text": "I regularly engage in activities that support my physical and mental health.",
                "choices": [
                    {"label": "Strongly disagree", "weights": {"Peace": 0}},
                    {"label": "Disagree", "weights": {"Peace": 1}},
                    {"label": "Neutral", "weights": {"Peace": 2}},
                    {"label": "Agree", "weights": {"Peace": 3}},
                    {"label": "Strongly agree", "weights": {"Peace": 4}},
                ],
            },
        ]
        return fallback, THEMES

    # Attempt to parse the file as JSON.
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Use the JSON's themes if present; otherwise default to our global THEMES.
    return data["questions"], data.get("themes", THEMES)


def q_version_hash(questions: List[dict]) -> str:
    """
    Create a short hash of the current questions (id + text only) so we can
    reset state if the questions change between deployments.
    """
    core = [{"id": q["id"], "text": q["text"]} for q in questions]
    s = json.dumps(core, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def ensure_state(questions: List[dict]) -> None:
    """
    Initialize Streamlit session_state keys and reconcile them when the
    questionnaire changes.
    """
    version = q_version_hash(questions)

    # Initialize if missing
    if "answers_by_qid" not in st.session_state:
        st.session_state["answers_by_qid"] = {}
    if "free_by_qid" not in st.session_state:
        st.session_state["free_by_qid"] = {}

    # Reset the per-question entries if the version hash changed.
    if st.session_state.get("q_version") != version:
        prev_answers = st.session_state.get("answers_by_qid", {})
        prev_free = st.session_state.get("free_by_qid", {})

        st.session_state["answers_by_qid"] = {q["id"]: prev_answers.get(q["id"]) for q in questions}
        st.session_state["free_by_qid"] = {q["id"]: prev_free.get(q["id"], "") for q in questions}
        st.session_state["q_version"] = version


def choice_key(qid: str) -> str:
    """Session key for a question's selected option."""
    return f"{qid}__choice"


def free_key(qid: str) -> str:
    """Session key for a question's free-text field."""
    return f"{qid}__free"

# -----------------------------------------------------------------------------
# FPDF 1.x safety: convert text to Latin-1 and provide small draw helpers
# -----------------------------------------------------------------------------

LATIN1_MAP: Dict[str, str] = {
    "—": "-", "–": "-", "―": "-",
    "“": '"', "”": '"', "„": '"',
    "’": "'", "‘": "'", "‚": "'",
    "•": "-", "·": "-", "∙": "-",
    "…": "...",
    "\u00a0": " ", "\u200b": ""
}

def to_latin1(text: str) -> str:
    """
    fpdf==1.7.2 wants Latin-1. We normalize and replace common Unicode punctuation.
    We also break very long tokens so PDF rendering won't blow up.
    """
    if not isinstance(text, str):
        text = str(text)

    # Normalization + simple map for common characters
    t = unicodedata.normalize("NFKD", text)
    for k, v in LATIN1_MAP.items():
        t = t.replace(k, v)

    # Encode down to latin-1 (or ascii as last resort)
    try:
        t = t.encode("latin-1", errors="ignore").decode("latin-1")
    except Exception:
        t = t.encode("ascii", errors="ignore").decode("ascii")

    # Avoid super-long unbreakable tokens (URLs, etc.)
    t = re.sub(r"(\S{80})\S+", r"\1", t)
    return t


def mc(pdf: "FPDF", text: str, h: float = 6.0) -> None:
    """Multi-cell with Latin-1 safety."""
    pdf.multi_cell(0, h, to_latin1(text))


def sc(pdf: "FPDF", w: float, h: float, text: str) -> None:
    """Single cell with Latin-1 safety."""
    pdf.cell(w, h, to_latin1(text))

# -----------------------------------------------------------------------------
# PDF helpers (titles, sections, bar chart, lists, checkboxes)
# -----------------------------------------------------------------------------

class PDF(FPDF):
    """Thin wrapper so we can extend later if needed."""
    pass


def setf(pdf: FPDF, style: str = "", size: int = 12) -> None:
    """Set Helvetica font, keeping this consistent across the report."""
    pdf.set_font("Helvetica", style, size)


def section_break(pdf: FPDF, title: str, desc: str = "") -> None:
    """Standard section header with optional descriptive line."""
    pdf.ln(3)
    setf(pdf, "B", 14)
    mc(pdf, title, h=7)
    if desc:
        setf(pdf, "", 11)
        mc(pdf, desc, h=6)
    pdf.ln(1)


def draw_scores_barchart(pdf: FPDF, scores: Dict[str, int]) -> None:
    """
    Lightweight horizontal bar chart drawn using fpdf rects.
    Only draws positive bars; a theme with zero or negative just shows the number.
    """
    setf(pdf, "B", 14)
    mc(pdf, "Your Theme Snapshot", h=7)

    setf(pdf, "", 12)
    positive_values = [v for v in scores.values() if v > 0]
    max_pos = max(positive_values) if positive_values else 1

    bar_w_max = 120  # pixels
    x_left = pdf.get_x() + 10
    y = pdf.get_y()

    for theme in THEMES:
        val = int(scores.get(theme, 0))

        pdf.set_xy(x_left, y)
        sc(pdf, 38, 6, theme)

        bar_x = x_left + 40
        bar_h = 4.5

        if val > 0:
            bar_w = (val / float(max_pos)) * bar_w_max
            pdf.set_fill_color(30, 144, 255)  # DodgerBlue-ish
            pdf.rect(bar_x, y + 1.3, bar_w, bar_h, "F")
            num_x = bar_x + bar_w + 2.0
        else:
            num_x = bar_x + 2.0

        pdf.set_xy(num_x, y)
        sc(pdf, 0, 6, str(val))

        y += 7

    pdf.set_y(y + 4)


def bullet_list(pdf: FPDF, items: List[str]) -> None:
    """Simple bullet list with consistent font/spacing."""
    setf(pdf, "", 11)
    for it in items or []:
        mc(pdf, f"- {it}")


def two_cols_lists(pdf: FPDF,
                   left_title: str, left_items: List[str],
                   right_title: str, right_items: List[str]) -> None:
    """Side-by-side lists with small headers."""
    setf(pdf, "B", 12)
    mc(pdf, left_title)
    setf(pdf, "", 11)
    bullet_list(pdf, left_items)

    pdf.ln(2)
    setf(pdf, "B", 12)
    mc(pdf, right_title)
    setf(pdf, "", 11)
    bullet_list(pdf, right_items)


def checkbox_line(pdf: FPDF, text: str, line_height: float = 8.0) -> None:
    """Draw a small checkbox plus a line of text."""
    x = pdf.get_x()
    y = pdf.get_y()
    box = 4.5

    pdf.rect(x, y + 2, box, box)
    pdf.set_xy(x + box + 3, y)
    mc(pdf, text, h=line_height)


def signature_week_block(pdf: FPDF, steps: List[str]) -> None:
    """Printable checklist page: 'Signature Week - At a glance' block."""
    section_break(
        pdf,
        "Signature Week - At a glance",
        "A simple plan you can print or screenshot. Check items off as you go."
    )
    setf(pdf, "", 12)
    for step in steps:
        checkbox_line(pdf, step)


def tiny_progress_block(pdf: FPDF, milestones: List[str]) -> None:
    """Tiny progress tracker (3 short milestones with checkboxes)."""
    section_break(pdf, "Tiny Progress Tracker", "Three tiny milestones you can celebrate this week.")
    setf(pdf, "", 12)
    for m in milestones:
        checkbox_line(pdf, m)

# -----------------------------------------------------------------------------
# Scoring functions
# -----------------------------------------------------------------------------

def compute_scores(questions: List[dict], answers_by_qid: Dict[str, str]) -> Dict[str, int]:
    """
    Aggregate per-theme scores from the chosen answers' weights.
    We skip questions that do not have an answer yet.
    """
    scores: Dict[str, int] = {t: 0 for t in THEMES}

    for q in questions:
        selected_label = answers_by_qid.get(q["id"])

        if not selected_label:
            # No answer chosen for this question yet.
            continue

        for choice in q["choices"]:
            if choice["label"] == selected_label:
                # Sum the choice's per-theme weights into the score dict.
                for theme_name, weight in choice.get("weights", {}).items():
                    scores[theme_name] = scores.get(theme_name, 0) + int(weight)
                break

    return scores


def top_n_themes(scores: Dict[str, int], n: int = 3) -> List[str]:
    """Return the top N theme names by score (descending)."""
    sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [t for t, _ in sorted_items[:n]]

# -----------------------------------------------------------------------------
# AI helpers
# -----------------------------------------------------------------------------

def ai_enabled() -> bool:
    """
    We only enable AI if:
      - Safe Mode is OFF, and
      - the OpenAI SDK is installed, and
      - an OPENAI_API_KEY is present (as env var or Streamlit Secret).
    """
    if SAFE_MODE:
        return False
    if not OPENAI_OK:
        return False

    key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))
    return bool(key)


def format_ai_prompt(first_name: str,
                     horizon_weeks: int,
                     scores: Dict[str, int],
                     free_responses: Dict[str, str],
                     top3: List[str]) -> str:
    """
    Build a single prompt that asks the model for all of the structured sections
    we need in JSON form. All wording uses "1 month later" for Future Snapshot.
    """
    score_lines = ", ".join(f"{k}:{v}" for k, v in scores.items())

    # Concatenate free-text answers as a simple source for synthesis.
    free_bits: List[str] = []
    for qid, txt in (free_responses or {}).items():
        if txt and txt.strip():
            free_bits.append(f"{qid}: {txt.strip()}")

    free_str = "\n".join(free_bits) if free_bits else "None provided."

    prompt = textwrap.dedent(f"""
        You are a master reflection coach. Using the theme scores, top themes, and the user's own words,
        produce ONE JSON object with EXACTLY these keys. Use ASCII only.

        # Header identity
        archetype: short string (<= 3 words)
        core_need: short phrase (<= 8 words)
        signature_metaphor: short phrase (<= 6 words)
        signature_sentence: single sentence (<= 16 words)

        # Narrative
        deep_insight: 400-600 words, second-person, practical, warm
        why_now: 120-180 words
        future_snapshot: 150-220 words (as if it is 1 month later)

        # Lists/briefs
        from_your_words: object with:
          summary: 60-110 words synthesizing their write-ins
          keepers: array of 2-3 short quotes or one-liners (<= 12 words)
        one_liners_to_keep: array of 3-5 short one-liners (<= 10 words)
        personal_pledge: one sentence in first person ("I will ...", <= 16 words)
        what_this_really_says: 180-260 words

        # Coaching blocks already in your app
        signature_strengths: array of 3-5 short phrases (<= 8 words)
        energy_map: object with energizers: array(3-6), drainers: array(3-6) (each <= 8 words)
        hidden_tensions: array of 2-4 short items (<= 12 words)
        watch_out: one gentle blind spot (<= 40 words)
        actions_7d: array of exactly 3 items (<= 12 words)
        impl_if_then: array of exactly 3 items "If X, then I will Y"
        plan_1_week: array of 5-7 steps (<= 12 words)
        balancing_opportunity: array of 1-2 one-liners for low themes (<= 14 words)
        keep_in_view: array of 2-4 reminders (<= 10 words)
        tiny_progress: array of exactly 3 milestones (<= 10 words)

        INPUT
        Name: {first_name or "friend"}
        Top themes: {", ".join(top3) if top3 else "-"}
        Theme scores: {score_lines}
        Their own words:
        {free_str}
    """).strip()

    return prompt


def parse_json_from_text(text: str) -> Optional[dict]:
    """
    Try to parse JSON out of a model response.
    We accept plain JSON objects or a fenced ```json ...``` block.
    """
    # Attempt plain JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt fenced block
    fenced = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            return None

    # Attempt first balanced-ish brace block
    blob = re.search(r"(\{.*\})", text, flags=re.S)
    if blob:
        try:
            return json.loads(blob.group(1))
        except Exception:
            return None

    return None


def run_ai(first_name: str,
           horizon_weeks: int,
           scores: Dict[str, int],
           free_responses: Dict[str, str],
           top3: List[str],
           cap_tokens: int) -> Tuple[Dict[str, Any], Dict[str, int], str]:
    """
    Call OpenAI Responses API (when enabled). If anything fails,
    we return a concise fallback JSON so the app continues gracefully.
    """
    if not ai_enabled():
        return ({}, {}, "AI disabled (Safe Mode ON or missing key/SDK)")

    client = OpenAI()
    prompt = format_ai_prompt(first_name, horizon_weeks, scores, free_responses, top3)

    def call(max_output_tokens: int):
        return client.responses.create(
            model=AI_MODEL,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )

    usage: Dict[str, int] = {}
    raw_text: str = ""
    data: Optional[dict] = None

    try:
        # First attempt with the higher cap
        resp = call(cap_tokens)
        raw_text = getattr(resp, "output_text", "") or ""
        if getattr(resp, "usage", None):
            u = resp.usage
            usage = {
                "input": getattr(u, "input_tokens", None) or getattr(u, "prompt_tokens", None) or 0,
                "output": getattr(u, "output_tokens", None) or getattr(u, "completion_tokens", None) or 0,
                "total": getattr(u, "total_tokens", None) or 0,
            }
        data = parse_json_from_text(raw_text)
        if not data:
            raise ValueError("AI did not return valid JSON")

    except Exception:
        # Second attempt: shorter cap
        try:
            resp = call(AI_MAX_TOKENS_FALLBACK)
            raw_text = getattr(resp, "output_text", "") or ""
            if getattr(resp, "usage", None):
                u = resp.usage
                usage = {
                    "input": getattr(u, "input_tokens", None) or getattr(u, "prompt_tokens", None) or 0,
                    "output": getattr(u, "output_tokens", None) or getattr(u, "completion_tokens", None) or 0,
                    "total": getattr(u, "total_tokens", None) or 0,
                }
            data = parse_json_from_text(raw_text)
        except Exception:
            data = None

    # Best-effort: write a short debug head to disk (useful on Streamlit Cloud).
    try:
        (here() / "_last_ai.json").write_text(raw_text[:12000], encoding="utf-8")
        (Path("/tmp") / "last_ai.json").write_text(raw_text[:12000], encoding="utf-8")
    except Exception:
        pass

    # Fallback content if AI response isn't parseable
    if not data:
        return ({
            "archetype": "Curious Connector",
            "core_need": "Growth with people",
            "signature_metaphor": "Compass in motion",
            "signature_sentence": "Small shared adventures are your fastest route to growth.",
            "deep_insight": "You are closer than you think. Focus on a few levers that energize you and remove one small drainer.",
            "why_now": "This season rewards steady experiments, kind boundaries, and inviting others into your progress.",
            "future_snapshot": "In 1 month you feel lighter and clearer. Small wins stacked into momentum.",
            "from_your_words": {"summary": "Your notes highlight craving motion with meaning.", "keepers": ["Try one new thing", "Invite a friend"]},
            "one_liners_to_keep": ["Small beats perfect", "Invite, don't wait", "Debrief 2 minutes"],
            "personal_pledge": "I will try one small new thing each week.",
            "what_this_really_says": "You thrive where novelty meets relationship. Design tiny anchors so experiences turn into growth.",
            "signature_strengths": ["Curiosity in action", "People-first focus", "Follow-through under constraints"],
            "energy_map": {"energizers": ["Tiny wins daily", "Learning in motion"], "drainers": ["Overcommitment", "Unclear next step"]},
            "hidden_tensions": ["High standards vs limited time"],
            "watch_out": "Beware scattering energy across too many half-starts.",
            "actions_7d": ["One 20m skill rep", "One connection invite", "One micro-adventure"],
            "impl_if_then": ["If distracted, then 10m timer", "If overwhelmed, then one step", "If stuck, then message ally"],
            "plan_1_week": ["Mon choose lever", "Tue 20m rep", "Wed invite friend", "Thu reset space", "Fri micro-adventure", "Sat reflect 10m", "Sun prep next"],
            "balancing_opportunity": ["Protect calm blocks", "Batch small chores"],
            "keep_in_view": ["Small > perfect", "Ask for help"],
            "tiny_progress": ["Finish one rep", "Send one invite", "Take one walk"],
        }, usage, raw_text[:800])

    return (data, usage, raw_text[:800])

# -----------------------------------------------------------------------------
# Email helpers (SMTP via Gmail App Password)
# -----------------------------------------------------------------------------

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def valid_email(addr: str) -> bool:
    """Quick-and-sane email format validation."""
    return bool(EMAIL_RE.match((addr or "").strip()))


def send_email(to_addr: str,
               subject: str,
               text_body: str,
               html_body: Optional[str] = None,
               attachments: Optional[List[Tuple[str, bytes, str]]] = None) -> None:
    """
    Send an email using Gmail's SMTP relay with TLS on port 587.
    Requires GMAIL_USER and GMAIL_APP_PASSWORD in Streamlit Secrets.
    """
    if not (GMAIL_USER and GMAIL_APP_PASSWORD):
        raise RuntimeError("Email not configured: set GMAIL_USER and GMAIL_APP_PASSWORD in Streamlit Secrets.")

    msg = EmailMessage()
    msg["From"] = formataddr((SENDER_NAME, GMAIL_USER))
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg["Reply-To"] = REPLY_TO

    # Plain text part is always present
    msg.set_content(text_body or "")

    # Optional HTML alternative
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    # Optional attachments
    for (filename, data, mime_type) in (attachments or []):
        maintype, subtype = mime_type.split("/", 1)
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)

    # TLS context + login + send
    context = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
        server.starttls(context=context)
        server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
        server.send_message(msg)


def generate_code() -> str:
    """Generate a 6-digit numeric code as a zero-padded string."""
    # Secure-ish: 3 random bytes => 0..16,777,215; mod 1,000,000 keeps it 6 digits.
    return f"{int.from_bytes(os.urandom(3), 'big') % 1_000_000:06d}"

# -----------------------------------------------------------------------------
# PDF builder (2 pages, consistent copy with "1 month ahead")
# -----------------------------------------------------------------------------

def make_pdf_bytes(first_name: str,
                   email: str,
                   scores: Dict[str, int],
                   top3: List[str],
                   ai: Dict[str, Any],
                   horizon_weeks: int,
                   logo_path: Optional[Path] = None) -> bytes:
    """
    Build the full Reflection Report PDF:
      - Page 1: Title, Top Themes + bars, narrative & structured sections
      - Page 2: Signature Week checklist + Tiny Progress tracker
    """
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    # --- Logo area and title -------------------------------------------------
    y_after_logo = 12
    if logo_path and Path(logo_path).exists():
        try:
            img = Image.open(logo_path).convert("RGBA")
            tmp = here() / "_logo_tmp.png"
            img.save(tmp, format="PNG")
            pdf.image(str(tmp), x=10, y=10, w=28)
            y_after_logo = 44
        except Exception:
            y_after_logo = 24

    pdf.set_y(y_after_logo)

    setf(pdf, "B", 18)
    mc(pdf, "Life Minus Work - Reflection Report", h=9)
    setf(pdf, "", 12)
    mc(pdf, f"Hi {first_name or 'there'},")

    # --- Top themes and bars -------------------------------------------------
    section_break(pdf, "Top Themes", "Where your energy is strongest right now.")
    mc(pdf, ", ".join(top3) if top3 else "-")
    draw_scores_barchart(pdf, scores)

    # --- From your words -----------------------------------------------------
    fyw = ai.get("from_your_words") or {}
    if fyw.get("summary"):
        section_break(pdf, "From your words", "We pulled a few cues from what you typed.")
        mc(pdf, fyw["summary"])

    # --- One-liners & Personal pledge ---------------------------------------
    if ai.get("one_liners_to_keep"):
        section_break(pdf, "One-liners to keep", "Tiny reminders that punch above their weight.")
        bullet_list(pdf, ai["one_liners_to_keep"])

    if ai.get("personal_pledge"):
        section_break(pdf, "Personal pledge", "Your simple promise to yourself.")
        mc(pdf, ai["personal_pledge"])

    # --- What this really says about you ------------------------------------
    if ai.get("what_this_really_says"):
        section_break(pdf, "What this really says about you", "A kind, honest read of your pattern.")
        mc(pdf, ai["what_this_really_says"])

    # --- Insights, Why Now, Future Snapshot ---------------------------------
    if ai.get("deep_insight"):
        section_break(pdf, "Insights", "A practical, encouraging synthesis of your answers.")
        mc(pdf, ai["deep_insight"])

    if ai.get("why_now"):
        section_break(pdf, "Why Now", "Why these themes may be active in this season.")
        mc(pdf, ai["why_now"])

    if ai.get("future_snapshot"):
        section_break(pdf, "Future Snapshot", "A short postcard from 1 month ahead.")
        mc(pdf, ai["future_snapshot"])

    # --- Coaching sections ---------------------------------------------------
    if ai.get("signature_strengths"):
        section_break(pdf, "Signature Strengths", "Traits to lean on when momentum matters.")
        bullet_list(pdf, ai["signature_strengths"])

    energy_map = ai.get("energy_map", {}) or {}
    if energy_map.get("energizers") or energy_map.get("drainers"):
        section_break(pdf, "Energy Map", "Name what fuels you, and what quietly drains you.")
        two_cols_lists(pdf, "Energizers", energy_map.get("energizers", []),
                       "Drainers", energy_map.get("drainers", []))

    if ai.get("hidden_tensions"):
        section_break(pdf, "Hidden Tensions", "Small frictions to watch with kindness.")
        bullet_list(pdf, ai["hidden_tensions"])

    if ai.get("watch_out"):
        section_break(pdf, "Watch-out (gentle blind spot)", "A nudge to keep you steady.")
        mc(pdf, ai["watch_out"])

    if ai.get("actions_7d"):
        section_break(pdf, "3 Next-step Actions (7 days)", "Tiny moves that compound quickly.")
        bullet_list(pdf, ai["actions_7d"])

    if ai.get("impl_if_then"):
        section_break(pdf, "Implementation Intentions (If-Then)", "Pre-decide responses to common bumps.")
        bullet_list(pdf, ai["impl_if_then"])

    if ai.get("plan_1_week"):
        section_break(pdf, "1-Week Gentle Plan", "A light structure you can actually follow.")
        bullet_list(pdf, ai["plan_1_week"])

    if ai.get("balancing_opportunity"):
        section_break(pdf, "Balancing Opportunity", "Low themes to tenderly rebalance.")
        bullet_list(pdf, ai["balancing_opportunity"])

    if ai.get("keep_in_view"):
        section_break(pdf, "Keep This In View", "Tiny reminders that protect progress.")
        bullet_list(pdf, ai["keep_in_view"])

    # --- Footer / next page note --------------------------------------------
    pdf.ln(4)
    setf(pdf, "B", 12)
    mc(pdf, "Next Page: Printable 'Signature Week - At a glance' + Tiny Progress Tracker")
    setf(pdf, "", 11)
    mc(pdf, "Tip: Put this on your fridge, desk, or phone notes.")

    # --- Page 2: checklists --------------------------------------------------
    pdf.add_page()

    default_plan = [
        "Day 1 (Mon): Review ideas 10m; pick a micro-adventure",
        "Day 2 (Tue): Invite one person with a clear, easy plan",
        "Day 3 (Wed): Prep one-line purpose and a simple backup",
        "Day 4 (Thu): Do the micro-adventure or 20m skill practice",
        "Day 5 (Fri): Send a short thank-you or highlight",
        "Day 6 (Sat): Reflect 5–10m; one lesson + one joy",
        "Day 7 (Sun): Rest; add two fresh ideas to the list",
    ]
    plan_steps = ai.get("plan_1_week") or default_plan
    signature_week_block(pdf, plan_steps)

    pdf.ln(2)
    default_tiny = [
        "Choose one small new activity + invite someone",
        "Capture one lesson + one gratitude",
        "Block a weekly 10-minute planning slot",
    ]
    tiny_progress_block(pdf, ai.get("tiny_progress") or default_tiny)

    # Footer lines
    pdf.ln(6)
    setf(pdf, "", 10)
    mc(pdf, f"Requested for: {email or '-'}")
    pdf.ln(6)
    setf(pdf, "", 9)
    mc(pdf, "Life Minus Work * This report is a starting point for reflection. Nothing here is medical or financial advice.")

    # Return bytes
    out = pdf.output(dest="S")
    if isinstance(out, str):
        out = out.encode("latin-1", errors="ignore")
    return out

# -----------------------------------------------------------------------------
# Streamlit UI (long, explicit version)
# -----------------------------------------------------------------------------

# Basic page setup
st.set_page_config(page_title="Life Minus Work — Questionnaire", page_icon="🧭", layout="centered")
st.title("Life Minus Work — Questionnaire")

# A visible boot marker. We'll remove this near the end of the project.
st.caption("✅ App booted. If you see this, imports & first render succeeded.")

# Load questions (or fallback) and set up session state.
questions, _themes = load_questions("questions.json")
ensure_state(questions)

# Introductory text
st.write(
    "Answer the questions, see your **Mini Report** instantly, then unlock your **complete Reflection Report** "
    "with a quick email verification. **Desktop:** Ctrl+Enter submits text areas. **Mobile:** tap outside to save."
)

# The "horizon" is now a fixed internal value. All user copy says "1 month ahead".
horizon_weeks: int = FUTURE_WEEKS_DEFAULT

# --- Questionnaire loop ------------------------------------------------------
for i, q in enumerate(questions, start=1):
    st.subheader(f"Q{i}. {q['text']}")

    # Build the list of radio labels, plus a write-in option.
    labels = [c["label"] for c in q["choices"]]
    WRITE_IN = "✍️ I'll write my own answer"
    labels_plus = labels + [WRITE_IN]

    # Restore previous selection if there was one.
    prev_choice = st.session_state["answers_by_qid"].get(q["id"])
    if prev_choice in labels_plus:
        default_index = labels_plus.index(prev_choice)
    else:
        default_index = 0

    # Radio select for this question
    selected = st.radio(
        "Pick one",
        labels_plus,
        index=default_index,
        key=choice_key(q["id"]),
        label_visibility="collapsed"
    )

    # Persist selection
    st.session_state["answers_by_qid"][q["id"]] = selected

    # If the user chose to write their own answer, show a text area and save it.
    if selected == WRITE_IN:
        ta_key = free_key(q["id"])
        default_text = st.session_state["free_by_qid"].get(q["id"], "")
        new_text = st.text_area(
            "Your words (a sentence or two)",
            value=default_text,
            key=ta_key,
            placeholder="Type here… (on mobile, tap outside to save)",
            height=90,
        )
        st.session_state["free_by_qid"][q["id"]] = new_text or ""
    else:
        # Remove any stale write-in if they switched back to a preset option.
        st.session_state["free_by_qid"].pop(q["id"], None)

# --- Future Snapshot explainer ----------------------------------------------
st.divider()
st.subheader("Future Snapshot")
st.write("Your full report includes a short **postcard from 1 month ahead** based on your answers and notes.")

# --- Mini Report trigger form -----------------------------------------------
with st.form("mini_form"):
    first_name_input = st.text_input(
        "Your first name (for the report greeting)",
        key="first_name_input",
        placeholder="First name"
    )
    submit_preview = st.form_submit_button("Show My Mini Report")

if submit_preview:
    st.session_state["preview_ready"] = True

# --- Mini Report (rich preview) ---------------------------------------------
if st.session_state.get("preview_ready"):
    # Compute scores and top-3 themes from current answers.
    scores = compute_scores(questions, st.session_state["answers_by_qid"])
    top3 = top_n_themes(scores, 3)

    # A bordered container previewing key pieces
    with st.container(border=True):
        st.subheader("Your Mini Report (Preview)")

        # Top themes in a quick line
        if top3:
            st.write(f"**Top themes:** {', '.join(top3)}")
        else:
            st.write("**Top themes:** -")

        # A small table of all theme scores (sorted)
        if scores:
            sorted_items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            st.table({
                "Theme": [k for k, _ in sorted_items],
                "Score": [v for _, v in sorted_items],
            })

        # Tailored tiny actions to make the preview feel personal
        tailored_actions: List[str] = []
        if "Connection" in top3:
            tailored_actions.append("Invite someone for a 20-minute walk this week.")
        if "Growth" in top3:
            tailored_actions.append("Schedule one 20-minute skill rep on your calendar.")
        if "Peace" in top3:
            tailored_actions.append("Block two 15-minute quiet blocks—phone away.")
        if "Identity" in top3:
            tailored_actions.append("Draft a 3-line purpose that feels true today.")
        if "Adventure" in top3:
            tailored_actions.append("Plan one micro-adventure within 30 minutes from home.")
        if "Contribution" in top3:
            tailored_actions.append("Offer a 30-minute help session to someone this week.")

        if tailored_actions:
            st.markdown("**Tiny actions to try this week:**")
            for act in tailored_actions[:3]:
                st.markdown(f"- {act}")

        # A clear promise of what the full report unlocks
        st.markdown("**What you’ll unlock with the full report:**")
        st.markdown(
            "- Your *postcard from 1 month ahead* (Future Snapshot)\n"
            "- Insights & “Why Now” (personalized narrative)\n"
            "- 3 next-step actions + If–Then plan\n"
            "- Energy Map (energizers & drainers)\n"
            "- Printable ‘Signature Week’ checklist page"
        )

    st.caption(
        "Unlock your complete Reflection Report to see your postcard from 1 month ahead, "
        "insights, plan & checklist."
    )

    # -------------------------------------------------------------------------
    # Email Gate: state initialization (we keep this outside of the button
    # callbacks so reruns rehydrate everything predictably).
    # -------------------------------------------------------------------------
    if "verify_state" not in st.session_state:
        # Possible values: "collect" (default) -> "sent" -> "verified"
        st.session_state.verify_state = "collect"

    if "pending_email" not in st.session_state:
        st.session_state.pending_email = ""

    if "pending_code" not in st.session_state:
        st.session_state.pending_code = ""

    if "code_issued_at" not in st.session_state:
        st.session_state.code_issued_at = 0.0

    if "last_send_ts" not in st.session_state:
        st.session_state.last_send_ts = 0.0

    # -------------------------------------------------------------------------
    # Email capture and code sending UI
    # -------------------------------------------------------------------------
    st.divider()
    st.subheader("Unlock your complete Reflection Report")
    st.write("We’ll email a 6-digit code to verify it’s really you. No spam—ever.")

    # Always show the email field (pre-populate if we already have a pending email).
    email_input = st.text_input(
        "Your email",
        value=st.session_state.pending_email or "",
        placeholder="you@example.com",
        key="gate_email"
    )

    col_send, col_caption = st.columns([1, 1])
    with col_send:
        send_code_clicked = st.button("Email me a 6-digit code")
    with col_caption:
        st.caption("You’ll enter it below to unlock your full report (PDF included).")

    if send_code_clicked:
        # Validate address first
        if not valid_email(email_input):
            st.error("Please enter a valid email address.")
        else:
            # Rate-limit the send button to prevent spamming
            now = time.time()
            if now - st.session_state.last_send_ts < 25:
                st.warning("Please wait a moment before requesting another code.")
            else:
                # Generate and store the new code
                code = generate_code()
                st.session_state.pending_email = email_input.strip()
                st.session_state.pending_code = code
                st.session_state.code_issued_at = now
                st.session_state.last_send_ts = now

                # Try sending the email; if email not configured, use Dev mode fallback.
                try:
                    plain = (
                        f"Your Life Minus Work verification code is: {code}\n"
                        f"This code expires in 10 minutes."
                    )
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
                    # Force a rerun so the code field appears immediately.
                    st.rerun()

                except Exception as e:
                    if DEV_SHOW_CODES:
                        st.warning(
                            f"(Dev Mode) Email not configured; using on-screen code: **{code}**"
                        )
                        st.session_state.verify_state = "sent"
                        st.rerun()
                    else:
                        st.error(f"Couldn’t send the code. {e}")

    # -------------------------------------------------------------------------
    # Code entry UI: ALWAYS visible once we have a pending email/code.
    # This solves the "I got the email but there's no field to enter it" issue.
    # -------------------------------------------------------------------------
    have_pending = bool(st.session_state.pending_email) and (
        bool(st.session_state.pending_code) or DEV_SHOW_CODES
    )

    if have_pending:
        st.info(f"Enter the 6-digit code sent to **{st.session_state.pending_email}**.")

        if DEV_SHOW_CODES:
            st.caption(f"(Dev) Code: **{st.session_state.pending_code}**")

        code_entered = st.text_input("Verification code", max_chars=6, key="verify_code_input")

        col_verify, col_resend = st.columns([1, 1])
        with col_verify:
            verify_clicked = st.button("Verify")
        with col_resend:
            resend_clicked = st.button("Resend code")

        # Handle verify
        if verify_clicked:
            # Expire codes after 10 minutes
            if time.time() - st.session_state.code_issued_at > 600:
                st.error("This code has expired. Please request a new one.")
            else:
                if (code_entered or "").strip() == st.session_state.pending_code:
                    st.success("Verified! Your full report is unlocked.")
                    st.session_state.verify_state = "verified"
                    st.rerun()
                else:
                    st.error("That code didn’t match. Please try again.")

        # Handle resend
        if resend_clicked:
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
                        text_body=(
                            f"Your code is: {st.session_state.pending_code}\n"
                            f"This code expires in 10 minutes."
                        ),
                        html_body=(
                            f"<p>Your code is:</p>"
                            f"<h2>{st.session_state.pending_code}</h2>"
                            f"<p>This code expires in 10 minutes.</p>"
                        )
                    )
                    st.success("We’ve sent a new code.")
                except Exception as e:
                    if DEV_SHOW_CODES:
                        st.warning(
                            f"(Dev Mode) Email not configured; new on-screen code: "
                            f"**{st.session_state.pending_code}**"
                        )
                    else:
                        st.error(f"Couldn’t resend the code. {e}")

    # -------------------------------------------------------------------------
    # Full report generation: only when verified.
    # We build the PDF and optionally email it to the verified address.
    # -------------------------------------------------------------------------
    if st.session_state.verify_state == "verified":
        st.success("Your email is verified.")

        # Recompute (in case they changed answers between preview and verify).
        scores = compute_scores(questions, st.session_state["answers_by_qid"])
        top3 = top_n_themes(scores, 3)
        free_responses = {
            qid: txt
            for qid, txt in (st.session_state.get("free_by_qid") or {}).items()
            if txt and txt.strip()
        }

        # Call AI (or use fallback) AFTER verification to save tokens.
        ai_sections, usage, raw_head = run_ai(
            first_name=st.session_state.get("first_name_input", first_name_input),
            horizon_weeks=horizon_weeks,
            scores=scores,
            free_responses=free_responses,
            top3=top3,
            cap_tokens=AI_MAX_TOKENS_CAP,
        )

        # Build the PDF bytes.
        logo = here() / "Life-Minus-Work-Logo.webp"
        pdf_bytes = make_pdf_bytes(
            first_name=st.session_state.get("first_name_input", first_name_input),
            email=st.session_state.pending_email,
            scores=scores,
            top3=top3,
            ai=ai_sections,
            horizon_weeks=horizon_weeks,
            logo_path=logo if logo.exists() else None,
        )

        # Download button
        st.subheader("Your Complete Reflection Report")
        st.write("Includes your postcard from **1 month ahead**, insights, plan & printable checklist.")
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="LifeMinusWork_Reflection_Report.pdf",
            mime="application/pdf",
        )

        # Optional: email the PDF to the verified address
        email_pdf_clicked = st.button("Email me the PDF")
        if email_pdf_clicked:
            try:
                send_email(
                    to_addr=st.session_state.pending_email,
                    subject="Your Life Minus Work Reflection Report",
                    text_body="Attached is your complete Reflection Report. Keep it handy!",
                    html_body="<p>Attached is your complete <b>Life Minus Work</b> Reflection Report. Keep it handy!</p>",
                    attachments=[("LifeMinusWork_Reflection_Report.pdf", pdf_bytes, "application/pdf")]
                )
                st.success("We’ve emailed your report.")
            except Exception as e:
                st.error(f"Could not email the PDF. {e}")

        # Debug expander for AI state (optional)
        with st.expander("AI status (debug)", expanded=False):
            st.write(f"AI enabled: {ai_enabled()}")
            st.write(f"Model: {AI_MODEL}")
            st.write(f"Max tokens: {AI_MAX_TOKENS_CAP} (fallback {AI_MAX_TOKENS_FALLBACK})")
            if usage:
                st.write(
                    f"Token usage — input: {usage.get('input', 0)}, "
                    f"output: {usage.get('output', 0)}, total: {usage.get('total', 0)}"
                )
            else:
                st.write("No usage returned (Safe Mode or fallback).")
            st.text("Raw head (first ~800 chars)")
            st.code(raw_head or "(empty)")
