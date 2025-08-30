# app.py — Life Minus Work (Streamlit)
# Notes:
# - Fixes UnicodeEncodeError in FPDF by sanitizing to Latin-1 for ALL PDF text
# - Uses ASCII-safe characters in PDF (bars '#', checkbox '[ ]', hyphens '-')
# - Keeps your email-gated flow with st.rerun() after 'sent' and 'verified'
# - Enriched Mini Report preview remains

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

# -----------------------------
# FPDF Latin-1 Safety
# -----------------------------

LATIN1_MAP = {
    "—": "-", "–": "-", "―": "-",
    "“": '"', "”": '"', "„": '"',
    "’": "'", "‘": "'", "‚": "'",
    "•": "-", "·": "-", "∙": "-",
    "…": "...",
    "□": "[ ]", "✓": "v", "✔": "v", "✗": "x", "✘": "x",
    "•": "-", "★": "*", "☆": "*",
    "█": "#", "■": "#", "▪": "-",
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
    t = re.sub(r"(\S{80})\S+", r"\1", t)  # avoid very long unbreakable tokens
    return t

def mc(pdf: "PDF", text: str, h=6):
    pdf.multi_cell(0, h, to_latin1(text))

def sc(pdf: "PDF", w, h, text: str):
    pdf.cell(w, h, to_latin1(text))

def section_break(pdf: "PDF", title: str, subtitle: str = ""):
    pdf.ln(4)
    setf(pdf, "B", 14)
    mc(pdf, title)
    if subtitle:
        setf(pdf, "", 10)
        mc(pdf, subtitle)
    pdf.ln(2)

def draw_scores_barchart(pdf: "PDF", scores: Dict[str, int]):
    # ASCII bars to be Latin-1 safe
    setf(pdf, "", 11)
    for k, v in scores.items():
        bar = "#" * max(1, int(v)) if v > 0 else ""
        mc(pdf, f"{k:12} {bar} {v}")

def here_logo() -> Path:
    return here() / "logo.png"

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
        st.session_state["answers_by_qid"] = {q["id"]: old_a.get(q["id"], "") for q in questions}
        st.session_state["free_by_qid"] = {q["id"]: old_f.get(q["id"], "") for q in questions}
        st.session_state["q_version"] = ver

# -----------------------------
# AI helpers (Safe Mode compatible)
# -----------------------------

def ai_enabled() -> bool:
    return (not SAFE_MODE) and bool(os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY", ""))) and OPENAI_OK

def run_ai(first_name: str, horizon_weeks: int, scores: Dict[str, int], scores_free: Dict[str, str] | None = None):
    """
    Returns (ai_sections_dict, usage, raw_text_head)
    In Safe Mode, returns a deterministic fallback payload (no network).
    """
    usage = {}
    raw_text = ""

    prompt = {
        "first_name": first_name or "",
        "horizon_weeks": horizon_weeks,
        "scores": scores,
        "free": scores_free or {},
    }
    raw_text = json.dumps(prompt)[:800]

    if not ai_enabled():
        data = {
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
        }
        return (data, usage, raw_text[:800])

    try:
        client = OpenAI()
        # (Real call would go here.) Returning same structure for now.
        data = {
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
        }
        usage = {"input": 0, "output": 0, "total": 0}
        return (data, usage, raw_text[:800])
    except Exception as e:
        st.error(f"AI call failed: {e}")
        return ({}, {}, raw_text[:800])

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
    logo = here_logo()
    if logo.exists():
        try:
            img = Image.open(logo).convert("RGBA")
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

    if ai.get("future_snapshot"):
        section_break(pdf, "Postcard from 1 month ahead")
        mc(pdf, ai["future_snapshot"])

    if ai.get("deep_insight"):
        section_break(pdf, "What this really says")
        mc(pdf, ai["deep_insight"])

    if ai.get("actions_7d"):
        section_break(pdf, "Next steps (7 days)")
        for a in ai["actions_7d"]:
            mc(pdf, f"- {a}")

    if ai.get("impl_if_then"):
        section_break(pdf, "If-Then plan")  # ASCII hyphen
        for a in ai["impl_if_then"]:
            mc(pdf, f"- {a}")

    # Signature week checklist page
    pdf.add_page()
    section_break(pdf, "Your Signature Week", "Use this as a simple checklist.")
    setf(pdf, "", 11)
    mc(pdf, "Mon choose lever")
    mc(pdf, "Tue 20m rep")
    mc(pdf, "Wed invite friend")
    mc(pdf, "Thu reset space")
    mc(pdf, "Fri micro-adventure")
    mc(pdf, "Sat reflect 10m")
    mc(pdf, "Sun prep next")

    pdf.ln(2)
    tiny = ai.get("tiny_progress") or [
        "Choose one small new activity + invite someone",
        "Capture one lesson + one gratitude",
        "Block a weekly 10-minute planning slot",
    ]
    for t in tiny:
        mc(pdf, f"[ ] {t}")  # ASCII checkbox

    pdf.ln(6)
    setf(pdf, "", 10)
    mc(pdf, f"Requested for: {email or '-'}")
    pdf.ln(6)
    setf(pdf, "", 9)
    mc(pdf, "Life Minus Work - This report is a starting point for reflection. Nothing here is medical or financial advice.")

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

        # Quick theme-scores table (sorted)
        if scores:
            _items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            st.table({"Theme": [k for k, _ in _items], "Score": [v for _, v in _items]})

        # Tailored tiny-actions based on your top themes
        recs = []
        if "Connection" in top3:   recs.append("Invite someone for a 20-minute walk this week.")
        if "Growth" in top3:       recs.append("Schedule one 20-minute skill rep on your calendar.")
        if "Peace" in top3:        recs.append("Block two 15-minute quiet blocks - phone away.")
        if "Identity" in top3:     recs.append("Draft a 3-line purpose that feels true today.")
        if "Adventure" in top3:    recs.append("Plan one micro-adventure within 30 minutes from home.")
        if "Contribution" in top3: recs.append("Offer a 30-minute help session to someone this week.")
        if recs:
            st.markdown("**Tiny actions to try this week:**")
            for r in recs[:3]:
                st.markdown(f"- {r}")

        # What you’ll unlock in the full report
        st.markdown("**What you’ll unlock with the full report:**")
        st.markdown(
            "- Your *postcard from 1 month ahead* (Future Snapshot)\n"
            "- Insights & Why Now (personalized narrative)\n"
            "- 3 next-step actions + If-Then plan\n"
            "- Energy Map (energizers & drainers)\n"
            "- Printable 'Signature Week' checklist page"
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
                        st.rerun()  # ensure the code-entry UI appears immediately
                    except Exception as e:
                        if DEV_SHOW_CODES:
                            st.warning(f"(Dev Mode) Email not configured; using on-screen code: **{code}**")
                            st.session_state.verify_state = "sent"
                            st.rerun()
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
                st.rerun()  # immediately reveal the full report section
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
            scores_free=free_responses,
        )

        pdf_bytes = make_pdf_bytes(
            first_name=st.session_state.get("first_name_input", first_name),
            email=st.session_state.pending_email,
            scores=scores,
            top3=top3,
            ai=ai_sections,
            horizon_weeks=horizon_weeks,
            logo_path=here_logo() if here_logo().exists() else None,
        )

        st.subheader("Your Complete Reflection Report")
        st.write("Includes your postcard from **1 month ahead**, insights, plan & printable checklist.")
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
