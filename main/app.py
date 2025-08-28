# app.py — fpdf1 safe + AI re-enabled (Responses API), Latin-1 sanitization everywhere

from __future__ import annotations
import json
import os
import hashlib
import unicodedata
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st
from fpdf import FPDF
from PIL import Image

# ====== THEMES / SETTINGS ======
THEMES = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]
FUTURE_WEEKS_MIN, FUTURE_WEEKS_MAX, FUTURE_WEEKS_DEFAULT = 2, 8, 4

# AI knobs
AI_ENABLED_DEFAULT = True
AI_MODEL = "gpt-5-mini"         # keep your chosen model
AI_MAX_TOKENS_CAP = 7000        # safe cap for deluxe report
AI_MAX_TOKENS_FALLBACK = 6000   # one retry at a lower cap
AI_TARGETS = {                  # word targets to guide the model
    "deep_insight": (400, 600),
    "why_now": (120, 180),
    "future_snapshot": (150, 220),
}

# Try to import OpenAI client
OPENAI_OK = False
try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False


# ====== FILES / LOADING ======
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
    core = [{"id": q["id"], "text": q["text"]} for q in questions]
    s = json.dumps(core, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


# ====== STATE ======
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


# ====== FPDF 1.x LATIN-1 SAFETY ======
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
    try:
        t = t.encode("latin-1", errors="ignore").decode("latin-1")
    except Exception:
        t = t.encode("ascii", errors="ignore").decode("ascii")
    # prevent ultra-long unbreakable tokens (fpdf 1 line-break quirk)
    t = re.sub(r"(\S{80})\S+", r"\1", t)
    return t

def mc(pdf: "FPDF", text: str, h: float = 6):
    pdf.multi_cell(0, h, to_latin1(text))

def sc(pdf: "FPDF", w: float, h: float, text: str):
    pdf.cell(w, h, to_latin1(text))


# ====== PDF HELPERS (fpdf1) ======
class PDF(FPDF):
    pass

def setf(pdf: FPDF, style: str = "", size: int = 12):
    pdf.set_font("Helvetica", style, size)  # Core14; safe for Latin-1

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

    # AI narrative sections
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

    out = pdf.output(dest="S")
    if isinstance(out, str):
        out = out.encode("latin-1", errors="ignore")
    return out


# ====== SCORING ======
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


# ====== OPENAI GLUE ======
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
    # direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # fenced
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S | re.I)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    # first { ... }
    m = re.search(r"(\{.*\})", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None

def run_ai(first_name: str, horizon_weeks: int, scores: Dict[str, int],
           free_responses: Dict[str, str], cap_tokens: int) -> Tuple[Dict[str, str], Dict[str, int], str]:
    """
    Returns: (sections, usage, raw_head)
      sections -> dict with deep_insight/why_now/future_snapshot (strings)
      usage    -> dict with input/output/total if available
      raw_head -> first 800 chars of raw model text for debugging
    """
    if not ai_enabled():
        return ({}, {}, "AI disabled or missing OPENAI_API_KEY.")

    client = OpenAI()
    prompt = format_ai_prompt(first_name, horizon_weeks, scores, free_responses)

    def call(max_output_tokens: int):
        return client.responses.create(
            model=AI_MODEL,
            input=prompt,
            max_output_tokens=max_output_tokens,
        )

    usage = {}
    raw_text = ""
    # Attempt 1
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
            raise ValueError("AI did not return valid JSON.")
        sections = {
            "deep_insight": data.get("deep_insight", "").strip(),
            "why_now": data.get("why_now", "").strip(),
            "future_snapshot": data.get("future_snapshot", "").strip(),
        }
        return (sections, usage, raw_text[:800])
    except Exception as e1:
        # Attempt 2 (fallback)
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
            if not data:
                raise ValueError("Fallback: AI did not return valid JSON.")
            sections = {
                "deep_insight": data.get("deep_insight", "").strip(),
                "why_now": data.get("why_now", "").strip(),
                "future_snapshot": data.get("future_snapshot", "").strip(),
            }
            return (sections, usage, raw_text[:800])
        except Exception as e2:
            # Final fallback (concise, Latin-1 friendly)
            sections = {
                "deep_insight": "You’re closer than you think. Start small, repeat, and protect the energy sources that actually move you.",
                "why_now": "This season is asking for gentle focus and one lever at a time—so you can build momentum without burning out.",
                "future_snapshot": f"In about {horizon_weeks} weeks, you’ve strung together tiny wins. Your days feel more yours—simpler, connected, and alive.",
            }
            return (sections, usage, f"AI failed: {e1} | {e2}")  # raw_head contains errors


# ====== STREAMLIT UI ======
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

# Q&A loop with optional write-in
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

# Always-visible reminder
st.caption("⏳ Generating can take up to 1 minute. Once it’s ready, tap **Download PDF** below.")

if submit_clicked:
    if not email_val or not consent_val:
        st.error("Please enter your email and give consent to continue.")
    else:
        st.session_state["email"] = email_val.strip()
        st.session_state["consent"] = True

        # Scores
        scores = {t: 0 for t in THEMES}
        for q in questions:
            sel = st.session_state["answers_by_qid"].get(q["id"])
            if not sel:
                continue
            for c in q["choices"]:
                if c["label"] == sel:
                    for k, v in c.get("weights", {}).items():
                        scores[k] += int(v)
                    break

        top3 = [t for t, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]]

        # Gather free responses for AI context
        free_responses = {}
        for q in questions:
            fr = st.session_state.get(free_key(q["id"]), "")
            if fr and fr.strip():
                free_responses[q["id"]] = fr.strip()

        # AI call (robust)
        sections, usage, raw_head = run_ai(
            first_name=st.session_state.get("first_name_input", first_name),
            horizon_weeks=horizon_weeks,
            scores=scores,
            free_responses=free_responses,
            cap_tokens=AI_MAX_TOKENS_CAP,
        )

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

        # AI diagnostics
        with st.expander("AI status (debug)", expanded=False):
            st.write(f"AI enabled: {ai_enabled()}")
            st.write(f"Model: {AI_MODEL}")
            st.write(f"Caps: {AI_MAX_TOKENS_CAP} (fallback {AI_MAX_TOKENS_FALLBACK})")
            if usage:
                st.write(f"Token usage — input: {usage.get('input', 0)}, output: {usage.get('output', 0)}, total: {usage.get('total', 0)}")
            else:
                st.write("No usage returned by the API (some models/paths omit it).")
            st.text("Raw head (first 800 chars):")
            st.code(raw_head)
