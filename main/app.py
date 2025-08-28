# app.py — fpdf 1.x safe, full Latin-1 sanitization, write-ins, clean PDF

from __future__ import annotations
import json
import hashlib
import unicodedata
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st
from fpdf import FPDF
from PIL import Image

THEMES = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]
FUTURE_WEEKS_MIN, FUTURE_WEEKS_MAX, FUTURE_WEEKS_DEFAULT = 2, 8, 4

# ---------------------------
# Files / Loading
# ---------------------------
def here() -> Path:
    return Path(__file__).parent

def load_questions(filename="questions.json") -> Tuple[List[dict], List[str]]:
    p = here() / filename
    if not p.exists():
        st.error(f"Could not find {filename} at {p}. Make sure it’s next to app.py.")
        try:
            st.caption("Directory listing for debugging:")
            for child in here().iterdir():
                st.write("-", child.name)
        except Exception:
            pass
        st.stop()
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"], data.get("themes", THEMES)

def q_version_hash(questions: List[dict]) -> str:
    # Version using id + text, so reorders don’t break, content changes do
    core = [{"id": q["id"], "text": q["text"]} for q in questions]
    s = json.dumps(core, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]

# ---------------------------
# State bootstrap
# ---------------------------
def ensure_state(questions: List[dict]):
    ver = q_version_hash(questions)
    if "answers_by_qid" not in st.session_state:
        st.session_state["answers_by_qid"] = {}
    if st.session_state.get("q_version") != ver:
        old = st.session_state.get("answers_by_qid", {})
        st.session_state["answers_by_qid"] = {q["id"]: old.get(q["id"]) for q in questions}
        st.session_state["q_version"] = ver

def choice_key(qid: str) -> str:
    return f"{qid}__choice"

def free_key(qid: str) -> str:
    return f"{qid}__free"

# ---------------------------
# fpdf 1.x Latin-1 safety
# ---------------------------
LATIN1_MAP = {
    "—": "-", "–": "-", "―": "-",
    "“": '"', "”": '"', "„": '"',
    "’": "'", "‘": "'", "‚": "'",
    "•": "-", "·": "-", "∙": "-",
    "…": "...",
    "\u00a0": " ",   # non-breaking space
    "\u200b": "",    # zero-width space
}

def to_latin1(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    t = unicodedata.normalize("NFKD", text)
    for k, v in LATIN1_MAP.items():
        t = t.replace(k, v)
    # strip anything non-latin1
    try:
        t = t.encode("latin-1", errors="ignore").decode("latin-1")
    except Exception:
        t = t.encode("ascii", errors="ignore").decode("ascii")
    # prevent ultra-long unbreakable tokens (fpdf 1 line-break quirk)
    t = re.sub(r"(\S{80})\S+", r"\1", t)
    return t

def mc(pdf: FPDF, text: str, h: float = 6):
    pdf.multi_cell(0, h, to_latin1(text))

def sc(pdf: FPDF, w: float, h: float, text: str):
    pdf.cell(w, h, to_latin1(text))

# ---------------------------
# PDF helpers (fpdf 1.x)
# ---------------------------
class PDF(FPDF):
    pass

def setf(pdf: FPDF, style: str = "", size: int = 12):
    # Helvetica (Core 14) is safe for Latin-1 in fpdf 1.x
    pdf.set_font("Helvetica", style, size)

def section_break(pdf: FPDF, title: str, description: str = ""):
    pdf.ln(3)
    setf(pdf, "B", 14); mc(pdf, title, h=7)
    if description:
        setf(pdf, "", 11); mc(pdf, description, h=6)
    pdf.ln(1)

def draw_scores_barchart(pdf: FPDF, scores: Dict[str, int]):
    setf(pdf, "B", 14); mc(pdf, "Your Theme Snapshot", h=7)
    setf(pdf, "", 12)

    positive_vals = [v for v in scores.values() if v > 0]
    max_pos = max(positive_vals) if positive_vals else 1

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

def make_pdf_bytes(
    first_name: str,
    email: str,
    scores: Dict[str, int],
    top3: List[str],
    sections: Dict[str, str],
    horizon_weeks: int,
    logo_path: Optional[Path] = None
) -> bytes:
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    # Optional logo (webp->png)
    if logo_path and Path(logo_path).exists():
        try:
            img = Image.open(logo_path).convert("RGBA")
            tmp = here() / "_logo_tmp.png"
            img.save(tmp, format="PNG")
            pdf.image(str(tmp), x=10, y=8, w=28)
        except Exception:
            pass
        pdf.ln(6)

    # Header
    pdf.set_y(14)
    setf(pdf, "B", 18); mc(pdf, "Life Minus Work — Reflection Report", h=9)
    setf(pdf, "", 12); mc(pdf, f"Hi {first_name or 'there'},")
    pdf.ln(2)

    # Top themes + bars
    section_break(pdf, "Top Themes", "Your strongest areas of focus right now.")
    mc(pdf, ", ".join(top3) if top3 else "—", h=6)
    draw_scores_barchart(pdf, scores)

    # Narrative sections (non-AI placeholders; swap with AI later if desired)
    if sections.get("deep_insight"):
        section_break(pdf, "Insights", "A synthesis of your answers into practical guidance.")
        mc(pdf, sections["deep_insight"])
    if sections.get("why_now"):
        section_break(pdf, "Why Now", "Why these themes may be active in this season.")
        mc(pdf, sections["why_now"])
    if sections.get("future_snapshot"):
        section_break(pdf, "Future Snapshot", f"A short ‘postcard from the future’ ~{horizon_weeks} weeks ahead.")
        mc(pdf, sections["future_snapshot"])

    # Pointer to next page
    pdf.ln(4)
    setf(pdf, "B", 12); mc(pdf, "Next Page: Printable ‘Signature Week’ Checklist")
    setf(pdf, "", 11); mc(pdf, "A one-page checklist you can use right away.")

    # Checklist page
    pdf.add_page()
    section_break(pdf, "Signature Week — At a glance", "Tick off 3 small wins this week.")
    for i in range(1, 4):
        sc(pdf, 8, 8, "[ ]"); sc(pdf, 0, 8, f"Milestone {i}"); pdf.ln(8)

    # Footer
    pdf.ln(4)
    if email:
        setf(pdf, "", 10); mc(pdf, f"Requested for: {email}")

    # fpdf 1.x returns str; convert to bytes
    out = pdf.output(dest="S")
    if isinstance(out, str):
        out = out.encode("latin-1", errors="ignore")
    return out

# ---------------------------
# Scoring
# ---------------------------
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

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Life Minus Work — Questionnaire", page_icon="🧭", layout="centered")
st.title("Life Minus Work — Questionnaire")

questions, _themes = load_questions("questions.json")
ensure_state(questions)

st.write(
    "Answer the prompts below. Choose one option per question. "
    "**Desktop:** Press Ctrl+Enter inside a text box to apply. "
    "**Mobile:** Tap outside the text box to save."
)

# Future Snapshot horizon (clear label)
horizon_weeks = st.slider(
    "How far ahead should we imagine your Future Snapshot? (in weeks)",
    min_value=FUTURE_WEEKS_MIN, max_value=FUTURE_WEEKS_MAX, value=FUTURE_WEEKS_DEFAULT,
    help="This sets the time horizon for the 'Future Snapshot' story in your report."
)

# Q&A Loop with optional write-in
for idx, q in enumerate(questions, start=1):
    st.subheader(f"Q{idx}. {q['text']}")
    labels = [c["label"] for c in q["choices"]]
    WRITE_IN = "✍️ I’ll write my own answer"
    labels_plus = labels + [WRITE_IN]

    prev = st.session_state["answers_by_qid"].get(q["id"])
    if prev not in labels_plus:
        prev_index = 0
    else:
        prev_index = labels_plus.index(prev)

    selected = st.radio(
        "Pick one",
        labels_plus,
        index=prev_index,
        key=choice_key(q["id"]),
        label_visibility="collapsed"
    )
    st.session_state["answers_by_qid"][q["id"]] = selected

    if selected == WRITE_IN:
        ta_key = free_key(q["id"])
        default_text = st.session_state.get(ta_key, "")
        new_text = st.text_area(
            "Your words (a sentence or two):",
            value=default_text,
            key=ta_key,
            placeholder="Type here… (on mobile, tap outside to save)",
            height=90
        )
        # Store your words for later narrative use (even without AI this is kept)
        st.session_state[ta_key] = new_text
    else:
        st.session_state.pop(free_key(q["id"]), None)

st.divider()

# Email & Generate
st.subheader("Email & Download")
with st.form("finish"):
    first_name = st.text_input("Your first name (for the report greeting)", key="first_name_input", placeholder="First name")
    email_val = st.text_input("Your email (to show on your PDF footer)", key="email_input", placeholder="you@example.com")
    consent_val = st.checkbox(
        "I agree to receive my results and occasional updates from Life Minus Work.",
        key="consent_input",
        value=st.session_state.get("consent_input", False),
    )
    submit_clicked = st.form_submit_button(
        "Generate My Personalized Report",
        help="This can take up to 1 minute"
    )

# Always-visible reminder under the form
st.caption("⏳ Generating can take up to 1 minute. Once it’s ready, tap **Download PDF** below.")

if submit_clicked:
    if not email_val or not consent_val:
        st.error("Please enter your email and give consent to continue.")
    else:
        st.session_state["email"] = email_val.strip()
        st.session_state["consent"] = True

        # Compute scores
        scores = compute_scores(questions, st.session_state["answers_by_qid"])
        top3 = top_n_themes(scores, 3)

        # Minimal narrative (non-AI) — safe placeholders (Latin-1 friendly)
        sections = {
            "deep_insight": (
                "Your current pattern highlights what energizes you most. "
                "Lean on your strongest theme to create small, repeatable wins this week."
            ),
            "why_now": (
                "This season likely calls for gentle focus and one lever at a time so you can "
                "build momentum without burning out."
            ),
            "future_snapshot": (
                f"In about {horizon_weeks} weeks, you’ve stacked small wins. "
                "Your days feel more yours—simpler, connected, and alive."
            ),
        }

        # Build PDF
        logo = here() / "Life-Minus-Work-Logo.webp"
        pdf_bytes = make_pdf_bytes(
            first_name=st.session_state.get("first_name_input", first_name),
            email=st.session_state.get("email", email_val),
            scores=scores,
            top3=top3,
            sections=sections,
            horizon_weeks=horizon_weeks,
            logo_path=logo if logo.exists() else None
        )

        st.success("Your PDF is ready.")
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="LifeMinusWork_Reflection_Report.pdf",
            mime="application/pdf",
        )
