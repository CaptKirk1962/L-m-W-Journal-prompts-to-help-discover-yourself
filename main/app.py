# app.py — Life Minus Work (rich report with Archetype, Core Need, Metaphor, etc.)
from __future__ import annotations
import os, json, re, hashlib, unicodedata, textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import streamlit as st
from fpdf import FPDF
from PIL import Image
import requests
from requests.exceptions import Timeout

# ---------- Config ----------
THEMES = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]

AI_MODEL = "gpt-4o-mini"  # Fallback to a known model due to gpt-5-mini issues
AI_MAX_TOKENS_CAP = 7000
AI_MAX_TOKENS_FALLBACK = 6000
AI_TIMEOUT_SECONDS = 30  # Timeout for API calls

FUTURE_WEEKS_MIN, FUTURE_WEEKS_MAX, FUTURE_WEEKS_DEFAULT = 2, 8, 4

# OpenAI SDK import flag
OPENAI_OK = False
try:
    from openai import OpenAI
    OPENAI_OK = True
except Exception:
    OPENAI_OK = False


# ---------- File & questions ----------
def here() -> Path:
    return Path(__file__).parent

def load_questions(filename="questions.json") -> Tuple[List[dict], List[str]]:
    p = here() / filename
    if not p.exists():
        st.error(f"Could not find {filename} at {p}. Make sure it is next to app.py.")
        try:
            st.caption("Directory listing (debug):")
            for c in here().iterdir():
                st.write("-", c.name)
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


# ---------- Session state scaffolding ----------
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


# ---------- fpdf 1.x Latin-1 safety ----------
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
    # avoid very long unbreakable tokens
    t = re.sub(r"(\S{80})\S+", r"\1", t)
    return t

def mc(pdf: "FPDF", text: str, h: float = 6):
    pdf.multi_cell(0, h, to_latin1(text))

def sc(pdf: "FPDF", w: float, h: float, text: str):
    pdf.cell(w, h, to_latin1(text))


# ---------- PDF helpers ----------
class PDF(FPDF):
    pass

def setf(pdf: FPDF, style: str = "", size: int = 12):
    pdf.set_font("Helvetica", style, size)

def section_break(pdf: FPDF, title: str, desc: str = ""):
    pdf.ln(3)
    setf(pdf, "B", 14); mc(pdf, title, h=7)
    if desc:
        setf(pdf, "", 11); mc(pdf, desc, h=6)
    pdf.ln(1)

def draw_scores_barchart(pdf: FPDF, scores: Dict[str, int]):
    setf(pdf, "B", 14); mc(pdf, "Your Theme Snapshot", h=7)
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
    setf(pdf, "B", 12); mc(pdf, left_title); setf(pdf, "", 11); bullet_list(pdf, left_items)
    pdf.ln(2)
    setf(pdf, "B", 12); mc(pdf, right_title); setf(pdf, "", 11); bullet_list(pdf, right_items)

def checkbox_line(pdf: FPDF, text: str, line_height: float = 10.0):
    # draw an empty square, then the text
    x = pdf.get_x()
    y = pdf.get_y()
    box = 4.5
    pdf.rect(x, y + 2, box, box)
    pdf.set_xy(x + box + 3, y)
    mc(pdf, text, h=line_height)

def signature_week_block(pdf: FPDF, steps: list[str]):
    section_break(pdf, "Signature Week – At a glance",
                  "A simple plan you can print or screenshot. Check items off as you go.")
    setf(pdf, "", 12)
    for step in steps:
        checkbox_line(pdf, step)

def tiny_progress_block(pdf: FPDF, milestones: list[str]):
    section_break(pdf, "Tiny Progress Tracker", "Three tiny milestones you can celebrate this week.")
    setf(pdf, "", 12)
    for m in milestones:
        checkbox_line(pdf, m)


# ---------- Scoring ----------
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


# ---------- AI ----------
def ai_enabled() -> bool:
    return OPENAI_OK and bool(os.getenv("OPENAI_API_KEY"))

def format_ai_prompt(first_name: str, horizon_weeks: int, scores: Dict[str, int],
                     free_responses: Dict[str, str], top3: List[str]) -> str:
    score_lines = ", ".join(f"{k}:{v}" for k, v in scores.items())
    fr_bits = []
    for qid, txt in free_responses.items():
        if txt and txt.strip():
            fr_bits.append(f"{qid}: {txt.strip()}")
    fr_str = "\n".join(fr_bits) if fr_bits else "None provided."

    # one JSON for everything; ASCII-only to be fpdf-safe
    return textwrap.dedent(f"""
    You are a master reflection coach. Using the theme scores, top themes, and the user's own words,
    produce ONE JSON object with EXACTLY these keys. Use ASCII only.

    # Header identity
    archetype: short string (<= 3 words)
    core_need: short phrase (<= 8 words)
    signature_metaphor: short phrase (<= 8 words)
    signature_sentence: short sentence (<= 12 words)

    # From your words
    from_your_words: dict with exactly these keys:
        summary: 1-2 sentence summary of key cues from user's words (<= 80 chars)
        bullets: list of 2-5 quoted or paraphrased bits from user's words (<= 10 words each)

    # Personal pledge
    personal_pledge: one sentence pledge (<= 15 words)

    # What this really says about you
    what_this_really_says: 4-6 paragraphs of encouraging, practical insight (each <= 100 chars)

    # Why this matters now
    why_now: short paragraph explaining why now is a good time (<= 80 chars)

    # Future Snapshot
    future_snapshot: short narrative postcard from {horizon_weeks} weeks ahead (<= 120 chars)

    # Signature strengths
    signature_strengths: list of 2-5 strengths (<= 5 words each)

    # Energy Map
    energy_map: dict with exactly these keys:
        energizers: list of 2-5 energizers (<= 5 words each)
        drainers: list of 2-5 drainers (<= 5 words each)

    # Hidden Tensions
    hidden_tensions: list of 2-5 tensions (<= 8 words each)

    # Watch-out (gentle blind spot)
    watch_out: short paragraph (<= 80 chars)

    # 3 Next-step Actions (7 days)
    actions_7d: list of 3 actions (<= 10 words each)

    # Implementation Intentions (If-Then)
    impl_if_then: list of 2-5 if-then statements (<= 15 words each)

    # 1-Week Gentle Plan
    plan_1_week: list of 4 steps (<= 15 words each)

    # Balancing Opportunity
    balancing_opportunity: list of 2-3 low-theme boosters (<= 10 words each)

    # Tiny Progress Tracker
    tiny_progress: list of 3 milestones (<= 10 words each)

    User data:
    First name: {first_name or "Friend"}
    Scores: {score_lines}
    Top themes: {", ".join(top3)}
    User's words: {fr_str}
    """)

def run_ai(first_name: str, horizon_weeks: int, scores: Dict[str, int],
           free_responses: Dict[str, str], top3: List[str],
           cap_tokens: int) -> Tuple[dict, Optional[dict], str]:
    if not ai_enabled():
        return {}, None, ""

    client = OpenAI()
    system = "Return ONLY the JSON object. Use ASCII only; no fancy punctuation."
    user = format_ai_prompt(first_name, horizon_weeks, scores, free_responses, top3)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    model = AI_MODEL
    try:
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=cap_tokens,
            response_format={"type": "json_object"},
            timeout=AI_TIMEOUT_SECONDS,
        )
        raw = r.choices[0].message.content if r.choices else ""
        usage = getattr(r, "usage", None)
        usage_dict = None
        if usage is not None:
            usage_dict = {
                "input": getattr(usage, "prompt_tokens", None),
                "output": getattr(usage, "completion_tokens", None),
                "total": getattr(usage, "total_tokens", None),
            }
        try:
            return json.loads(raw), usage_dict, raw[:800]
        except json.JSONDecodeError:
            st.warning("AI returned invalid JSON. Falling back.")
    except (Timeout, Exception) as e:
        st.warning(f"AI call failed: {str(e)}. Trying fallback model.")

    # fallback mode
    model = FALLBACK_MODEL
    try:
        r = client.chat.completions.create(
            model=model,
            messages=messages,
            max_completion_tokens=AI_MAX_TOKENS_FALLBACK,
            response_format={"type": "json_object"},
            timeout=AI_TIMEOUT_SECONDS,
        )
        raw = r.choices[0].message.content if r.choices else ""
        usage = getattr(r, "usage", None)
        usage_dict = None
        if usage is not None:
            usage_dict = {
                "input": getattr(usage, "prompt_tokens", None),
                "output": getattr(usage, "completion_tokens", None),
                "total": getattr(usage, "total_tokens", None),
            }
        try:
            return json.loads(raw), usage_dict, raw[:800]
        except json.JSONDecodeError:
            st.warning("Fallback AI returned invalid JSON.")
    except (Timeout, Exception) as e:
        st.error(f"Fallback AI failed: {str(e)}. Using default template.")
    return {}, None, ""


# ---------- PDF generator ----------
def make_pdf_bytes(
    first_name: str,
    email: str,
    scores: Dict[str, int],
    top3: List[str],
    ai: dict,
    horizon_weeks: int,
    logo_path: Optional[Path],
) -> bytes:
    try:
        pdf = PDF(orientation="P", unit="mm", format="A4")
        pdf.set_margins(10, 10, 10)
        pdf.add_page()
        setf(pdf, "", 12)

        # Logo if available
        if logo_path and logo_path.exists():
            try:
                with Image.open(logo_path) as img:
                    img = img.convert("RGB")
                    logo_temp = Path("/tmp/lmw_logo.png")
                    img.save(logo_temp, "PNG")
                    pdf.image(str(logo_temp), x=10, y=8, w=30)
            except Exception as e:
                st.warning(f"Failed to load logo: {str(e)}")

        # Title and greeting
        setf(pdf, "B", 20); mc(pdf, "Your Reflection Report", h=8)
        setf(pdf, "", 12); mc(pdf, f"Hi {first_name or ''},", h=6)
        ts = datetime.datetime.now().strftime("%d %b %Y")
        mc(pdf, f"Date: {ts}", h=6)
        mc(pdf, f"Email: {email or ''}", h=6)
        pdf.ln(2)

        # Archetype, Core Need, Signature Metaphor
        mc(pdf, "Archetype")
        setf(pdf, "", 12); mc(pdf, ai.get("archetype", "Your Archetype"))
        mc(pdf, "Core Need")
        mc(pdf, ai.get("core_need", "Core Need..."))
        pdf.ln(1)
        mc(pdf, "Signature Metaphor")
        mc(pdf, ai.get("signature_metaphor", "Signature Metaphor..."))
        pdf.ln(2)
        mc(pdf, "Signature Sentence")
        mc(pdf, ai.get("signature_sentence", "Signature Sentence..."))
        pdf.ln(3)

        draw_scores_barchart(pdf, scores)

        # ===== From your words =====
        fyw = ai.get("from_your_words") or {}
        if fyw.get("summary"):
            section_break(pdf, "From your words", "We pulled a few cues from what you typed.")
            mc(pdf, fyw["summary"])
        if fyw.get("bullets"):
            bullet_list(pdf, fyw["bullets"])

        # ===== Personal pledge =====
        if ai.get("personal_pledge"):
            section_break(pdf, "Personal pledge", "Your simple promise to yourself.")
            mc(pdf, ai["personal_pledge"])

        # ===== What this really says about you =====
        if ai.get("what_this_really_says"):
            section_break(pdf, "What this really says about you", "A kind, honest read of your pattern.")
            mc(pdf, ai["what_this_really_says"])

        # ===== Why this matters now =====
        if ai.get("why_now"):
            section_break(pdf, "Why this matters now", "Why these themes may be active in this season.")
            mc(pdf, ai["why_now"])

        # ===== Future Snapshot =====
        if ai.get("future_snapshot"):
            section_break(pdf, f"Future Snapshot - {horizon_weeks} weeks", "A short postcard from ahead.")
            mc(pdf, ai["future_snapshot"])

        # ===== Signature strengths =====
        if ai.get("signature_strengths"):
            section_break(pdf, "Signature strengths", "Traits to lean on when momentum matters.")
            bullet_list(pdf, ai["signature_strengths"])

        # ===== Energy Map =====
        em = ai.get("energy_map", {}) or {}
        if em.get("energizers") or em.get("drainers"):
            section_break(pdf, "Energy Map", "Name what fuels you, and what quietly drains you.")
            two_cols_lists(pdf, "Energizers", em.get("energizers", []), "Drainers", em.get("drainers", []))

        # ===== Hidden Tensions =====
        if ai.get("hidden_tensions"):
            section_break(pdf, "Hidden Tensions", "Small frictions to watch with kindness.")
            bullet_list(pdf, ai["hidden_tensions"])

        # ===== Watch-out =====
        if ai.get("watch_out"):
            section_break(pdf, "Watch-out (gentle blind spot)", "A nudge to keep you steady.")
            mc(pdf, ai["watch_out"])

        # ===== 3 Next-step Actions (7 days) =====
        if ai.get("actions_7d"):
            section_break(pdf, "3 Next-step Actions (7 days)", "Tiny moves that compound quickly.")
            bullet_list(pdf, ai["actions_7d"])

        # ===== Implementation Intentions (If-Then) =====
        if ai.get("impl_if_then"):
            section_break(pdf, "Implementation Intentions (If-Then)", "Pre-decide responses to common bumps.")
            bullet_list(pdf, ai["impl_if_then"])

        # ===== 1-Week Gentle Plan =====
        if ai.get("plan_1_week"):
            section_break(pdf, "1-Week Gentle Plan", "A light structure you can actually follow.")
            bullet_list(pdf, ai["plan_1_week"])

        # ===== Balancing Opportunity =====
        if ai.get("balancing_opportunity"):
            section_break(pdf, "Balancing Opportunity", "Low themes to tenderly rebalance.")
            bullet_list(pdf, ai["balancing_opportunity"])

        pdf.ln(4)
        setf(pdf, "B", 12); mc(pdf, "Next Page: Printable 'Signature Week – At a glance' + Tiny Progress Tracker")
        setf(pdf, "", 11); mc(pdf, "Tip: Put this on your fridge, desk, or phone notes.")

        # ===== Page 2: Signature Week & Tiny Progress =====
        pdf.add_page()
        signature_week_block(pdf, ai.get("plan_1_week") or [
            "Day 1: Review ideas 10 min, pick adventure",
            "Day 2: Invite one person with clear plan",
            "Day 3: Prep a purpose and backup",
            "Day 4: Do adventure or skill practice",
            "Day 5: Send a thank-you or highlight",
            "Day 6: Reflect 5-10 min, note joy",
            "Day 7: Rest, add two new ideas",
        ])
        pdf.ln(10)  # Add roomy spacing
        tiny_progress_block(pdf, ai.get("tiny_progress") or [
            "Choose a new activity and invite",
            "Note one lesson and gratitude",
            "Plan a 10-min weekly slot",
        ])
        pdf.ln(10)  # Add roomy spacing

        pdf.ln(6)
        setf(pdf, "", 10); mc(pdf, f"Requested for: {email or '-'}")
        pdf.ln(6)
        setf(pdf, "", 9)
        mc(pdf, "Life Minus Work * This report is a starting point for reflection. Nothing here is medical or financial advice.")

        out = pdf.output(dest="S")
        if isinstance(out, str):
            out = out.encode("latin-1", errors="ignore")
        return out
    except Exception as e:
        st.error(f"PDF generation failed: {str(e)}. Check inputs or AI output.")
        pdf = PDF(orientation="P", unit="mm", format="A4")
        pdf.add_page()
        pdf.set_margins(10, 10, 10)
        setf(pdf, "", 12)
        mc(pdf, "Error generating report. Try shorter inputs.")
        out = pdf.output(dest="S")
        return out if isinstance(out, bytes) else out.encode("latin-1", errors="ignore")


# ---------- App UI ----------
st.set_page_config(page_title="Life Minus Work — Questionnaire", page_icon="🧭", layout="centered")
st.title("Life Minus Work — Questionnaire")

questions, _themes = load_questions("questions.json")
ensure_state(questions)

st.write(
    "Answer 15 questions, add your own reflections, and instantly download a personalized PDF summary. "
    "**Desktop:** press Ctrl+Enter in text boxes to apply. **Mobile:** tap outside the box to save."
)

horizon_weeks = st.slider(
    "Future Snapshot horizon (weeks)",
    min_value=FUTURE_WEEKS_MIN, max_value=FUTURE_WEEKS_MAX, value=FUTURE_WEEKS_DEFAULT,
    help="How far ahead should the 'Future Snapshot' imagine? This only affects the story tone."
)

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
        # keep a copy in our dict (never assign to the widget key)
        st.session_state["free_by_qid"][q["id"]] = new_text
    else:
        st.session_state["free_by_qid"].pop(q["id"], None)

st.divider()

st.subheader("Email & Download")
with st.form("finish"):
    first_name = st.text_input("Your first name (for the report greeting)", key="first_name_input", placeholder="First name")
    email_val = st.text_input("Your email (printed in PDF footer)", key="email_input", placeholder="you@example.com")
    consent_val = st.checkbox(
        "I agree to receive my results and occasional updates from Life Minus Work.",
        key="consent_input",
        value=st.session_state.get("consent_input", False),
    )
    submit = st.form_submit_button("Generate My Personalized Report", help="This can take up to 1 minute")

st.caption("⏳ It can take up to 1 minute to generate. When ready, click **Download PDF** below.")

if submit:
    if not email_val or not consent_val:
        st.error("Please enter your email and give consent to continue.")
    else:
        st.session_state["email"] = email_val.strip()
        st.session_state["consent"] = True

        scores = compute_scores(questions, st.session_state["answers_by_qid"])
        top3 = top_n_themes(scores, 3)
        free_responses = {qid: txt for qid, txt in st.session_state["free_by_qid"].items() if txt and txt.strip()}

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
            email=st.session_state.get("email", email_val),
            scores=scores,
            top3=top3,
            ai=ai_sections,
            horizon_weeks=horizon_weeks,
            logo_path=logo if logo.exists() else None,
        )

        st.success("Your PDF is ready.")
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name="LifeMinusWork_Reflection_Report.pdf",
            mime="application/pdf",
        )

        with st.expander("AI status (debug)", expanded=False):
            st.write(f"AI enabled: {ai_enabled()}")
            st.write(f"Model: {AI_MODEL}")
            st.write(f"Max tokens: {AI_MAX_TOKENS_CAP} (fallback {AI_MAX_TOKENS_FALLBACK})")
            if usage:
                st.write(f"Token usage — input: {usage.get('input', 0)}, output: {usage.get('output', 0)}, total: {usage.get('total', 0)}")
            else:
                st.write("No usage returned by the API (some models/paths omit it).")
            st.text("Raw head (first 800 chars)")
            st.code(raw_head)
