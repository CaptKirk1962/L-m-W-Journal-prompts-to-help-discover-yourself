# app.py — Life Minus Work (Streamlit)
# Features:
# - Future Snapshot fixed to "1 month ahead" (no weeks slider)
# - Mini-report preview, gated full report via 6-digit email verification
# - Send emails from Gmail via SMTP (App Password)
# - Safe Mode for AI (bypass OpenAI calls unless LW_SAFE_MODE=0 and OPENAI_API_KEY present)
# - Robust PDF builder (2 pages, bars, coach sections), Latin-1 safe for fpdf 1.x
# - Hardened questions loader with fallback

from __future__ import annotations
import os, json, re, hashlib, unicodedata, textwrap, time, ssl, smtplib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from email.message import EmailMessage
from email.utils import formataddr

import streamlit as st
from fpdf import FPDF
from PIL import Image

# -----------------------------
# App Config & Constants
# -----------------------------

THEMES = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]

AI_MODEL = "gpt-5-mini"
AI_MAX_TOKENS_CAP = 8000
AI_MAX_TOKENS_FALLBACK = 7000

# Future Snapshot horizon: fixed internally; wording everywhere is "1 month ahead"
FUTURE_WEEKS_DEFAULT = 4

# Safe Mode (default ON) disables AI calls to avoid cloud spinner if networking is blocked
SAFE_MODE = os.getenv("LW_SAFE_MODE", st.secrets.get("LW_SAFE_MODE", "1")) == "1"

# Email / SMTP (Gmail App Password)
GMAIL_USER = st.secrets.get("GMAIL_USER", "whatisyourminus@gmail.com")
GMAIL_APP_PASSWORD = st.secrets.get("GMAIL_APP_PASSWORD", "")  # 16-char App Password (not your login password)
SENDER_NAME = st.secrets.get("SENDER_NAME", "Life Minus Work")
REPLY_TO = st.secrets.get("REPLY_TO", "whatisyourminus@gmail.com")

# Developer helper: set DEV_SHOW_CODES="1" in secrets to echo verification codes onscreen (for testing only)
DEV_SHOW_CODES = os.getenv("DEV_SHOW_CODES", st.secrets.get("DEV_SHOW_CODES", "0")) == "1"

# Try OpenAI SDK
OPENAI_OK = False
try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

# -----------------------------
# Utility / Paths
# -----------------------------

def here() -> Path:
    return Path(__file__).parent

def load_questions(filename="questions.json") -> Tuple[List[dict], List[str]]:
    """Load questions from JSON; if missing, use a tiny fallback so the app always renders."""
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
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"], data.get("themes", THEMES)

def q_version_hash(questions: List[dict]) -> str:
    core = [{"id": q["id"], "text": q["text"]} for q in questions]
    s = json.dumps(core, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

# -----------------------------
# Session State
# -----------------------------

def ensure_state(questions: List[dict]):
    ver = q_version_hash(questions)
    if "answers_by_qid" not in st.session_state:
        st.session_state["answers_by_qid"] = {}
    if "free_by_qid" not in st.session_state:
        st.session_state["free_by_qid"] = {}
    if st.session_state.get("q_version") != ver:
        old_a = st.session_state.get("answers_by_qid", {})
        old_f = st.session_state.get("free_by_qid", {})
        st.session_state["answers_by_qid"] = {q["id"]: old_a.get(q["id"]) for q in questions}
        st.session_state["free_by_qid"] = {q["id"]: old_f.get(q["id"], "") for q in questions}
        st.session_state["q_version"] = ver

def choice_key(qid: str) -> str:
    return f"{qid}__choice"

def free_key(qid: str) -> str:
    return f"{qid}__free"

# -----------------------------
# fpdf 1.x Latin-1 safety
# -----------------------------

LATIN1_MAP = {
    "—": "-", "–": "-", "―": "-",
    "“": '"', "”": '"', "„": '"',
    "’": "'", "‘": "'", "‚": "'",
    "•": "-", "·": "-", "∙": "-",
    "…": "...",
    "\u00a0": " ", "\u200b": ""
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
    t = re.sub(r"(\S{80})\S+", r"\1", t)  # avoid very long unbreakable tokens
    return t

def mc(pdf: "FPDF", text: str, h: float = 6):
    pdf.multi_cell(0, h, to_latin1(text))

def sc(pdf: "FPDF", w: float, h: float, text: str):
    pdf.cell(w, h, to_latin1(text))

# -----------------------------
# PDF Helpers
# -----------------------------

class PDF(FPDF):
    pass

def setf(pdf: FPDF, style: str = "", size: int = 12):
    pdf.set_font("Helvetica", style, size)

def section_break(pdf: FPDF, title: str, desc: str = ""):
    pdf.ln(3)
    setf(pdf, "B", 14)
    mc(pdf, title, h=7)
    if desc:
        setf(pdf, "", 11)
        mc(pdf, desc, h=6)
    pdf.ln(1)

def draw_scores_barchart(pdf: FPDF, scores: Dict[str, int]):
    setf(pdf, "B", 14)
    mc(pdf, "Your Theme Snapshot", h=7)
    setf(pdf, "", 12)
    positive = [v for v in scores.values() if v > 0]
    max_pos = max(positive) if positive else 1
    bar_w_max = 120
    x_left = pdf.get_x() + 10
    y = pdf.get_y()
    for theme in THEMES:
        val = int(scores.get(theme, 0))
        pdf.set_xy(x_left, y)
        sc(pdf, 38, 6, theme)
        bar_x = x_left + 40
        bar_h = 4.5
        if val > 0:
            bar_w = (val / max_pos) * bar_w_max
            pdf.set_fill_color(30, 144, 255)
            pdf.rect(bar_x, y + 1.3, bar_w, bar_h, "F")
            num_x = bar_x + bar_w + 2
        else:
            num_x = bar_x + 2
        pdf.set_xy(num_x, y)
        sc(pdf, 0, 6, str(val))
        y += 7
    pdf.set_y(y + 4)

def bullet_list(pdf: FPDF, items: List[str]):
    setf(pdf, "", 11)
    for it in items or []:
        mc(pdf, f"- {it}")

def two_cols_lists(pdf: FPDF, left_title: str, left_items: List[str],
                   right_title: str, right_items: List[str]):
    setf(pdf, "B", 12)
    mc(pdf, left_title)
    setf(pdf, "", 11)
    bullet_list(pdf, left_items)
    pdf.ln(2)
    setf(pdf, "B", 12)
    mc(pdf, right_title)
    setf(pdf, "", 11)
    bullet_list(pdf, right_items)

def checkbox_line(pdf: FPDF, text: str, line_height: float = 8.0):
    x = pdf.get_x()
    y = pdf.get_y()
    box = 4.5
    pdf.rect(x, y + 2, box, box)
    pdf.set_xy(x + box + 3, y)
    mc(pdf, text, h=line_height)

def signature_week_block(pdf: FPDF, steps: List[str]):
    section_break(
        pdf,
        "Signature Week - At a glance",
        "A simple plan you can print or screenshot. Check items off as you go."
    )
    setf(pdf, "", 12)
    for step in steps:
        checkbox_line(pdf, step)

def tiny_progress_block(pdf: FPDF, milestones: List[str]):
    section_break(pdf, "Tiny Progress Tracker", "Three tiny milestones you can celebrate this week.")
    setf(pdf, "", 12)
    for m in milestones:
        checkbox_line(pdf, m)

# -----------------------------
# Scoring
# -----------------------------

def compute_scores(questions: List[dict], answers_by_qid: Dict[str, str]) -> Dict[str, int]:
    scores = {t: 0 for t in THEMES}
    for q in questions:
        sel = answers_by_qid.get(q["id"])
        if not sel:
            continue
        for c in q["choices"]:
            if c["label"] == sel:
                for k, v in c.get("weights", {}).items():
                    scores[k] = scores.get(k, 0) + int(v)
                break
    return scores

def top_n_themes(scores: Dict[str, int], n: int = 3) -> List[str]:
    return [t for t, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]]

# -----------------------------
# AI
# -----------------------------

def ai_enabled() -> bool:
    return (not SAFE_MODE) and OPENAI_OK and bool(os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", "")))

def format_ai_prompt(first_name: str, horizon_weeks: int, scores: Dict[str, int],
                     free_responses: Dict[str, str], top3: List[str]) -> str:
    score_lines = ", ".join(f"{k}:{v}" for k, v in scores.items())
    fr_bits = []
    for qid, txt in free_responses.items():
        if txt and txt.strip():
            fr_bits.append(f"{qid}: {txt.strip()}")
    fr_str = "\n".join(fr_bits) if fr_bits else "None provided."

    return textwrap.dedent(f"""
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
    {fr_str}
    """).strip()

def parse_json_from_text(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    m = re.search(r"(\{.*\})", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None

def run_ai(first_name: str, horizon_weeks: int, scores: Dict[str, int],
           free_responses: Dict[str, str], top3: List[str],
           cap_tokens: int) -> Tuple[Dict[str, Any], Dict[str, int], str]:
    if not ai_enabled():
        return ({}, {}, "AI disabled (Safe Mode on or missing key)")

    client = OpenAI()
    prompt = format_ai_prompt(first_name, horizon_weeks, scores, free_responses, top3)

    def call(max_output_tokens: int):
        return client.responses.create(
            model=AI_MODEL,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )

    usage, raw_text, data = {}, "", None
    try:
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

    # Save head for debugging (best-effort)
    try:
        (here() / "_last_ai.json").write_text(raw_text[:12000], encoding="utf-8")
        (Path("/tmp") / "last_ai.json").write_text(raw_text[:12000], encoding="utf-8")
    except Exception:
        pass

    if not data:
        # concise fallback with month phrasing
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

# -----------------------------
# Email Helpers (SMTP via Gmail)
# -----------------------------

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def valid_email(e: str) -> bool:
    return bool(EMAIL_RE.match((e or "").strip()))

def send_email(to_addr: str, subject: str, text_body: str, html_body: str | None = None,
               attachments: list[tuple[str, bytes, str]] | None = None):
    """Send email using Gmail SMTP with App Password (TLS on 587)."""
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
    # 000000–999999, always 6 digits
    return f"{int.from_bytes(os.urandom(3), 'big') % 1_000_000:06d}"

# -----------------------------
# PDF Builder
# -----------------------------

def make_pdf_bytes(
    first_name: str,
    email: str,
    scores: Dict[str, int],
    top3: List[str],
    ai: Dict[str, Any],
    horizon_weeks: int,
    logo_path: Optional[Path] = None
) -> bytes:
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    # Logo → title
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

    section_break(pdf, "Top Themes", "Where your energy is strongest right now.")
    mc(pdf, ", ".join(top3) if top3 else "-")
    draw_scores_barchart(pdf, scores)

    fyw = ai.get("from_your_words") or {}
    if fyw.get("summary"):
        section_break(pdf, "From your words", "We pulled a few cues from what you typed.")
        mc(pdf, fyw["summary"])

    if ai.get("one_liners_to_keep"):
        section_break(pdf, "One-liners to keep", "Tiny reminders that punch above their weight.")
        bullet_list(pdf, ai["one_liners_to_keep"])

    if ai.get("personal_pledge"):
        section_break(pdf, "Personal pledge", "Your simple promise to yourself.")
        mc(pdf, ai["personal_pledge"])

    if ai.get("what_this_really_says"):
        section_break(pdf, "What this really says about you", "A kind, honest read of your pattern.")
        mc(pdf, ai["what_this_really_says"])

    if ai.get("deep_insight"):
        section_break(pdf, "Insights", "A practical, encouraging synthesis of your answers.")
        mc(pdf, ai["deep_insight"])

    if ai.get("why_now"):
        section_break(pdf, "Why Now", "Why these themes may be active in this season.")
        mc(pdf, ai["why_now"])

    if ai.get("future_snapshot"):
        section_break(pdf, "Future Snapshot", "A short postcard from 1 month ahead.")
        mc(pdf, ai["future_snapshot"])

    if ai.get("signature_strengths"):
        section_break(pdf, "Signature Strengths", "Traits to lean on when momentum matters.")
        bullet_list(pdf, ai["signature_strengths"])

    em = ai.get("energy_map", {}) or {}
    if em.get("energizers") or em.get("drainers"):
        section_break(pdf, "Energy Map", "Name what fuels you, and what quietly drains you.")
        two_cols_lists(pdf, "Energizers", em.get("energizers", []), "Drainers", em.get("drainers", []))

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

    pdf.ln(4)
    setf(pdf, "B", 12)
    mc(pdf, "Next Page: Printable 'Signature Week - At a glance' + Tiny Progress Tracker")
    setf(pdf, "", 11)
    mc(pdf, "Tip: Put this on your fridge, desk, or phone notes.")

    pdf.add_page()

    plan = ai.get("plan_1_week") or [
        "Day 1 (Mon): Review ideas 10m; pick a micro-adventure",
        "Day 2 (Tue): Invite one person with a clear, easy plan",
        "Day 3 (Wed): Prep one-line purpose and a simple backup",
        "Day 4 (Thu): Do the micro-adventure or 20m skill practice",
        "Day 5 (Fri): Send a short thank-you or highlight",
        "Day 6 (Sat): Reflect 5–10m; one lesson + one joy",
        "Day 7 (Sun): Rest; add two fresh ideas to the list",
    ]
    signature_week_block(pdf, plan)

    pdf.ln(2)
    tiny = ai.get("tiny_progress") or [
        "Choose one small new activity + invite someone",
        "Capture one lesson + one gratitude",
        "Block a weekly 10-minute planning slot",
    ]
    tiny_progress_block(pdf, tiny)

    pdf.ln(6)
    setf(pdf, "", 10)
    mc(pdf, f"Requested for: {email or '-'}")
    pdf.ln(6)
    setf(pdf, "", 9)
    mc(pdf, "Life Minus Work * This report is a starting point for reflection. Nothing here is medical or financial advice.")

    out = pdf.output(dest="S")
    if isinstance(out, str):
        out = out.encode("latin-1", errors="ignore")
    return out

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Life Minus Work — Questionnaire", page_icon="🧭", layout="centered")
st.title("Life Minus Work — Questionnaire")
st.caption("✅ App booted. If you see this, imports & first render succeeded.")

questions, _themes = load_questions("questions.json")
ensure_state(questions)

st.write(
    "Answer the questions, add your own reflections, and unlock a personalized PDF summary. "
    "**Desktop:** press Ctrl+Enter in text boxes to apply. **Mobile:** tap outside the box to save."
)

# Fixed Future Snapshot horizon (no slider)
horizon_weeks = FUTURE_WEEKS_DEFAULT

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
            placeholder="Type here… (on mobile, tap outside to save)",
            height=90,
        )
        st.session_state["free_by_qid"][q["id"]] = new_text or ""
    else:
        st.session_state["free_by_qid"].pop(q["id"], None)

st.divider()
st.subheader("Future Snapshot")
st.write("Your full report includes a short **postcard from 1 month ahead** based on your answers and notes.")

# Submit basic answers to show Mini Report preview
with st.form("mini_form"):
    first_name = st.text_input("Your first name (for the report greeting)", key="first_name_input", placeholder="First name")
    submit_preview = st.form_submit_button("Show My Mini Report")

if submit_preview:
    st.session_state["preview_ready"] = True

if st.session_state.get("preview_ready"):
    # Compute preview data
    scores = compute_scores(questions, st.session_state["answers_by_qid"])
    top3 = top_n_themes(scores, 3)
    with st.container(border=True):
        st.subheader("Your Mini Report (Preview)")
        st.write(f"**Top themes:** {', '.join(top3) if top3 else '-'}")
        # Add 1-2 neutral one-liners as free value
        st.markdown(
            "- Small > perfect — pick one tiny action this week.\n"
            "- Invite a friend — progress sticks better together."
        )
    st.caption("Unlock your complete Reflection Report to see your postcard from 1 month ahead, insights, plan & checklist.")

    # -------------------------
    # Email Gate State
    # -------------------------
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

    # Step A: Collect email & send code
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
                    except Exception as e:
                        if DEV_SHOW_CODES:
                            st.warning(f"(Dev Mode) Email not configured; using on-screen code: **{code}**")
                            st.session_state.verify_state = "sent"
                        else:
                            st.error(f"Couldn’t send the code. {e}")

    # Step B: Enter code & verify
    elif st.session_state.verify_state == "sent":
        st.info(f"Enter the 6-digit code we emailed to **{st.session_state.pending_email}**.")
        v = st.text_input("Verification code", max_chars=6)
        c1, c2 = st.columns([1, 1])
        with c1:
            verify_btn = st.button("Verify")
        with c2:
            resend = st.button("Resend code")
        if verify_btn:
            # expire after 10 minutes
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

    # Step C: Verified → build full report, download & email PDF
    elif st.session_state.verify_state == "verified":
        st.success("Your email is verified.")

        # Build full report (AI only runs now, and only if Safe Mode off)
        scores = compute_scores(questions, st.session_state["answers_by_qid"])
        top3 = top_n_themes(scores, 3)
        free_responses = {qid: txt for qid, txt in (st.session_state.get("free_by_qid") or {}).items() if txt and txt.strip()}

        ai_sections, usage, raw_head = run_ai(
            first_name=st.session_state.get("first_name_input", first_name),
            horizon_weeks=horizon_weeks,
            scores=scores,
            free_responses=free_responses,
            top3=top3,
            cap_tokens=AI_MAX_TOKENS_CAP,
        )

        logo = here() / "Life-Minus-Work-Logo.webp"
        pdf_bytes = make_pdf_bytes(
            first_name=st.session_state.get("first_name_input", first_name),
            email=st.session_state.pending_email,
            scores=scores,
            top3=top3,
            ai=ai_sections,
            horizon_weeks=horizon_weeks,
            logo_path=logo if logo.exists() else None,
        )

        st.subheader("Your Complete Reflection Report")
        st.write("Includes your postcard from **1 month ahead**, insights, plan & printable checklist.")
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="LifeMinusWork_Reflection_Report.pdf",
            mime="application/pdf",
        )

        # Optional: email the PDF to the verified address
        if st.button("Email me the PDF"):
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

        with st.expander("AI status (debug)", expanded=False):
            st.write(f"AI enabled: {ai_enabled()}")
            st.write(f"Model: {AI_MODEL}")
            st.write(f"Max tokens: {AI_MAX_TOKENS_CAP} (fallback {AI_MAX_TOKENS_FALLBACK})")
            if usage:
                st.write(f"Token usage — input: {usage.get('input', 0)}, output: {usage.get('output', 0)}, total: {usage.get('total', 0)}")
            else:
                st.write("No usage returned (Safe Mode or fallback).")
            st.text("Raw head (first 800 chars)")
            st.code(raw_head or "(empty)")
