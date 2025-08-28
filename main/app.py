# app.py
from __future__ import annotations
import json, os, hashlib, textwrap, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st
from fpdf import FPDF
from PIL import Image

# Optional OpenAI (Responses API)
OPENAI_OK = False
try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False

# =========================
# App Constants & Settings
# =========================
THEMES = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]

# AI knobs (safe, “deluxe” defaults)
AI_ENABLED_DEFAULT = True
AI_MODEL = "gpt-5-mini"     # keep as you requested
AI_PATH_NOTE = "chat+rf_mct"  # just shown in debug, not sent to API
AI_MAX_TOKENS_HIGH = 7000     # safe cap; you can raise to 8000 if you like
AI_MAX_TOKENS_FALLBACK = 6000 # one retry at slightly lower cap
AI_TARGETS = {
    "deep_insight": (400, 600),
    "why_now": (120, 180),
    "future_snapshot": (150, 220),
}
FUTURE_WEEKS_MIN, FUTURE_WEEKS_MAX, FUTURE_WEEKS_DEFAULT = 2, 8, 4

# =======================
# Utility / IO functions
# =======================
def here() -> Path:
    return Path(__file__).parent

def load_questions(filename="questions.json") -> Tuple[List[dict], List[str]]:
    p = here() / filename
    if not p.exists():
        st.error(f"Could not find {filename} at {p}. Make sure it’s next to app.py.")
        try:
            st.caption("Directory listing:")
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

# =======================
# State Bootstrap / Keys
# =======================
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

# =======================
# PDF Helpers (fpdf2)
# =======================
class PDF(FPDF):
    pass

def add_font_dejavu(pdf: PDF) -> bool:
    """Embed DejaVuSans.ttf if present for full Unicode. Otherwise use Helvetica."""
    ttf = here() / "DejaVuSans.ttf"
    try:
        if ttf.exists():
            pdf.add_font("DejaVu", "", str(ttf), uni=True)
            pdf.add_font("DejaVu", "B", str(ttf), uni=True)
            pdf.set_font("DejaVu", "", 12)
            return True
    except Exception:
        pass
    pdf.set_font("Helvetica", "", 12)  # core14 (ASCII safe)
    return False

def setf(pdf: PDF, style: str = "", size: int = 12):
    family = "DejaVu" if "DejaVu" in pdf.fonts else "Helvetica"
    pdf.set_font(family, style, size)

def section_break(pdf: PDF, title: str, description: str = ""):
    pdf.ln(3)
    setf(pdf, "B", 14)
    pdf.multi_cell(0, 7, title)
    if description:
        setf(pdf, "", 11)
        pdf.multi_cell(0, 6, description)
    pdf.ln(1)

def draw_scores_barchart(pdf: PDF, scores: Dict[str, int]):
    """Draw horizontal bars; never draw negative-width bars (show number only)."""
    setf(pdf, "B", 14); pdf.multi_cell(0, 7, "Your Theme Snapshot")
    setf(pdf, "", 12)
    positive_vals = [v for v in scores.values() if v > 0]
    max_pos = max(positive_vals) if positive_vals else 1

    bar_w_max = 120
    x_left = pdf.get_x() + 10
    y = pdf.get_y()

    for theme in THEMES:
        val = int(scores.get(theme, 0))
        pdf.set_xy(x_left, y)
        pdf.cell(38, 6, theme)

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
        pdf.cell(0, 6, str(val))
        y += 7

    pdf.set_y(y + 4)

def make_pdf_bytes(first_name: str, email: str, scores: Dict[str, int],
                   top3: List[str], sections: Dict[str, str],
                   logo_path: Optional[Path] = None) -> bytes:
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    unicode_ok = add_font_dejavu(pdf)

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
    setf(pdf, "B", 18)
    pdf.multi_cell(0, 9, "Life Minus Work — Reflection Report")
    setf(pdf, "", 12)
    greet = f"Hi {first_name}," if first_name else "Hi there,"
    pdf.multi_cell(0, 6, greet)
    pdf.ln(2)

    # Top themes + bars
    section_break(pdf, "Top Themes", "Your strongest areas of focus right now.")
    pdf.multi_cell(0, 6, ", ".join(top3) if top3 else "—")
    draw_scores_barchart(pdf, scores)

    # AI sections (if provided)
    if sections.get("deep_insight"):
        section_break(pdf, "Insights", "A synthesis of your answers into practical guidance.")
        pdf.multi_cell(0, 6, sections["deep_insight"])
    if sections.get("why_now"):
        section_break(pdf, "Why Now", "Why these themes may be active in this season.")
        pdf.multi_cell(0, 6, sections["why_now"])
    if sections.get("future_snapshot"):
        section_break(pdf, "Future Snapshot", "A short ‘postcard from the future’ to help you see it.")
        pdf.multi_cell(0, 6, sections["future_snapshot"])

    # Pointer to next page
    pdf.ln(4)
    setf(pdf, "B", 12)
    pdf.multi_cell(0, 6, "Next Page: Printable ‘Signature Week’ Checklist")
    setf(pdf, "", 11)
    pdf.multi_cell(0, 6, "A one-page checklist you can use right away.")

    # Checklist page
    pdf.add_page()
    section_break(pdf, "Signature Week — At a glance", "Tick off 3 small wins this week.")
    for i in range(1, 4):
        pdf.cell(8, 8, "☐"); pdf.cell(0, 8, f"Milestone {i}"); pdf.ln(8)

    # Footer
    pdf.ln(4)
    if email:
        setf(pdf, "", 10)
        pdf.multi_cell(0, 5, f"Requested for: {email}")

    # Output (latin-1 safe if no DejaVu)
    return pdf.output(dest="S").encode("latin-1", errors="ignore" if not unicode_ok else "strict")

# =======================
# Scoring / Utility
# =======================
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

# =======================
# OpenAI glue (optional)
# =======================
def ai_enabled() -> bool:
    return AI_ENABLED_DEFAULT and OPENAI_OK and bool(os.getenv("OPENAI_API_KEY"))

def format_ai_prompt(first_name: str, horizon_weeks: int, scores: Dict[str, int],
                     free_responses: Dict[str, str]) -> str:
    score_lines = ", ".join(f"{k}:{v}" for k, v in scores.items())
    fr_bits = []
    for qid, txt in free_responses.items():
        if txt and txt.strip():
            fr_bits.append(f"{qid}: {txt.strip()}")
    fr_str = "\n".join(fr_bits) if fr_bits else "None provided."

    di_lo, di_hi = AI_TARGETS["deep_insight"]
    wn_lo, wn_hi = AI_TARGETS["why_now"]
    fs_lo, fs_hi = AI_TARGETS["future_snapshot"]

    return textwrap.dedent(f"""
    You are an expert reflection coach. Using the user’s theme scores and a few of their own words,
    produce a JSON object with exactly these keys:
    - "deep_insight": {di_lo}-{di_hi} words, second-person, practical, warm.
    - "why_now": {wn_lo}-{wn_hi} words, brief context for this season; no medical claims.
    - "future_snapshot": {fs_lo}-{fs_hi} words, write as if it’s {horizon_weeks} weeks later and they nailed it.

    Constraints:
    - No bullet characters; plain paragraphs only.
    - Do not repeat their words verbatim; synthesize.
    - Use “you” voice.

    INPUT:
    Name: {first_name or "friend"}
    Theme scores: {score_lines}
    Their own words:
    {fr_str}
    """).strip()

def parse_json_from_text(text: str) -> Optional[dict]:
    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Try fenced block
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    # Try first { … } block
    m = re.search(r"(\{.*\})", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None

def run_ai(first_name: str, horizon_weeks: int, scores: Dict[str, int],
           free_responses: Dict[str, str], max_tokens: int) -> Tuple[Dict[str, str], Dict[str, int], str]:
    """
    Returns: (sections, usage_dict, raw_head)
    sections: {"deep_insight":..., "why_now":..., "future_snapshot":...}
    usage_dict: {"input":x,"output":y,"total":z} if available
    raw_head: first ~800 chars of raw text for debug
    """
    if not ai_enabled():
        return ({}, {}, "AI disabled or API key missing.")

    client = OpenAI()  # uses env var OPENAI_API_KEY
    prompt = format_ai_prompt(first_name, horizon_weeks, scores, free_responses)

    # first attempt
    usage = {}
    raw_text = ""
    try:
        resp = client.responses.create(
            model=AI_MODEL,
            input=prompt,
            max_output_tokens=max_tokens,
        )
        raw_text = getattr(resp, "output_text", "") or ""
        # Usage (defensive)
        if getattr(resp, "usage", None):
            u = resp.usage
            usage = {
                "input": getattr(u, "input_tokens", None) or getattr(u, "prompt_tokens", None) or 0,
                "output": getattr(u, "output_tokens", None) or getattr(u, "completion_tokens", None) or 0,
                "total": getattr(u, "total_tokens", None) or 0,
            }
        data = parse_json_from_text(raw_text)
        if not data:
            raise ValueError("AI did not return valid JSON.")
        sections = {
            "deep_insight": data.get("deep_insight", "").strip(),
            "why_now": data.get("why_now", "").strip(),
            "future_snapshot": data.get("future_snapshot", "").strip(),
        }
        return (sections, usage, raw_text[:800])
    except Exception as e:
        # fallback attempt at lower cap
        try:
            resp = client.responses.create(
                model=AI_MODEL,
                input=prompt,
                max_output_tokens=AI_MAX_TOKENS_FALLBACK,
            )
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
                raise ValueError("Fallback: AI did not return valid JSON.")
            sections = {
                "deep_insight": data.get("deep_insight", "").strip(),
                "why_now": data.get("why_now", "").strip(),
                "future_snapshot": data.get("future_snapshot", "").strip(),
            }
            return (sections, usage, raw_text[:800])
        except Exception as ee:
            # final concise fallback
            sections = {
                "deep_insight": "You’re closer than you think. Start small, repeat, and protect the energy sources that actually move you.",
                "why_now": "This season is asking for gentle focus and one lever at a time—so you can build momentum without burning out.",
                "future_snapshot": f"In about {horizon_weeks} weeks, you’ve strung together tiny wins. Your days feel more yours—simpler, connected, and alive."
            }
            return (sections, usage, f"AI failed: {e} | {ee}")

# =======================
# Streamlit UI
# =======================
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

# Q&A Loop with write-in option
free_responses: Dict[str, str] = {}
for idx, q in enumerate(questions, start=1):
    st.subheader(f"Q{idx}. {q['text']}")
    labels = [c["label"] for c in q["choices"]]

    # Add write-in sentinel
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
        # show text area immediately
        ta_key = free_key(q["id"])
        default_text = st.session_state.get(ta_key, "")
        new_text = st.text_area(
            "Your words (a sentence or two):",
            value=default_text,
            key=ta_key,
            placeholder="Type here… (on mobile, tap outside to save)",
            height=90
        )
        # store for AI
        free_responses[q["id"]] = new_text
    else:
        # ensure write-in cleared for this q if any
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

# Always-visible reminder (placed AFTER the form so it can't affect submit)
st.caption("📌 After generation, click **Download PDF** to receive your report.")

# Handle submit
if submit_clicked:
    if not email_val or not consent_val:
        st.error("Please enter your email and give consent to continue.")
    else:
        st.session_state["email"] = email_val.strip()
        st.session_state["consent"] = True

        # Compute scores
        scores = compute_scores(questions, st.session_state["answers_by_qid"])
        # NOTE: if you want to forbid negatives entirely, uncomment:
        # for t in THEMES: scores[t] = max(0, scores[t])

        top3 = top_n_themes(scores, 3)

        # AI (optional)
        sections, usage, raw_head = run_ai(
            first_name=st.session_state.get("first_name_input", first_name),
            horizon_weeks=horizon_weeks,
            scores=scores,
            free_responses={k: st.session_state.get(free_key(k), "") for k in st.session_state.get("answers_by_qid", {})},
            max_tokens=AI_MAX_TOKENS_HIGH
        )

        # Build PDF
        logo = here() / "Life-Minus-Work-Logo.webp"
        pdf_bytes = make_pdf_bytes(
            first_name=st.session_state.get("first_name_input", first_name),
            email=st.session_state.get("email", email_val),
            scores=scores,
            top3=top3,
            sections=sections,
            logo_path=logo if logo.exists() else None
        )

        st.success("Your PDF is ready.")
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="LifeMinusWork_Reflection_Report.pdf",
            mime="application/pdf",
        )

        # Token tracker / diagnostics
        with st.expander("AI status (debug)", expanded=False):
            st.write(f"AI enabled: {ai_enabled()}")
            st.write(f"Model: {AI_MODEL}")
            st.write(f"Max tokens (cap): {AI_MAX_TOKENS_HIGH} (fallback {AI_MAX_TOKENS_FALLBACK})")
            if usage:
                st.write(f"Token usage — input: {usage.get('input', 0)}, output: {usage.get('output', 0)}, total: {usage.get('total', 0)}")
            else:
                st.write("No usage returned by the API (some models/paths omit it).")
            st.text("Raw head (first 800 chars):")
            st.code(raw_head)

