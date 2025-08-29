# app.py
# Life Minus Work — Questionnaire -> AI report -> PDF (FPDF v1)
# Streamlit Cloud friendly; OpenAI optional with safe fallbacks.

import os, json, datetime, csv, textwrap
from pathlib import Path

import streamlit as st
from fpdf import FPDF
from PIL import Image

# ---------- Config ----------
APP_TITLE = "Life Minus Work — Reflection Quiz"
LOGO_FILE = "Life-Minus-Work-Logo.webp"  # in /main next to app.py
QUESTIONS_FILE = "questions.json"        # in /main next to app.py

# OpenAI knobs (Responses API—robust to parameter differences)
OPENAI_MODEL = "gpt-5-mini"              # your chosen model
MAX_OUTPUT_TOKENS_HIGH = 7000            # generous but safe
MAX_OUTPUT_TOKENS_FALLBACK = 6000        # retry cap

# ---------- Small utilities ----------
def base_dir() -> Path:
    return Path(__file__).parent

def latin1(s: str) -> str:
    """Force text into latin-1 range so FPDF v1 won't crash."""
    if s is None: return ""
    if not isinstance(s, str): s = str(s)
    return s.encode("latin-1", errors="ignore").decode("latin-1", errors="ignore")

def wrap(s: str, width=95) -> str:
    lines = []
    for para in (s or "").splitlines():
        if not para.strip():
            lines.append("")
        else:
            lines.extend(textwrap.wrap(para, width=width, break_long_words=True, break_on_hyphens=True))
    return "\n".join(lines)

def load_questions(filename=QUESTIONS_FILE):
    p = base_dir() / filename
    if not p.exists():
        st.error(f"Could not find {filename} at {p}. Make sure it sits next to app.py.")
        try:
            st.caption("Directory listing next to app.py:")
            for q in base_dir().iterdir():
                st.write("-", q.name)
        except Exception:
            pass
        st.stop()
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"], data.get("themes", [])

# stable keys no matter the order
def choice_key(qid): return f"{qid}__choice"
def free_key(qid):   return f"{qid}__free"

# ---------- Scoring ----------
THEMES = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]

def compute_scores(answers, questions):
    scores = {t: 0 for t in THEMES}
    for q in questions:
        sel = answers.get(q["id"], {})
        # radio choice scoring
        idx = sel.get("choice_index")
        if isinstance(idx, int) and 0 <= idx < len(q["choices"]):
            weights = q["choices"][idx].get("weights", {})
            for k, v in weights.items():
                if k in scores: scores[k] += int(v)
        # free text adds Identity +1 (very gentle bias toward reflection)
        if sel.get("free"):
            scores["Identity"] += 1
    return scores

def rank_themes(scores):
    return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

# ---------- AI ----------
def make_ai_payload(first_name, email, scores, top3, free_texts, horizon_weeks):
    score_lines = ", ".join([f"{k}:{v}" for k, v in scores.items()])
    free_blob = "\n".join([f"- {t}" for t in free_texts if t.strip()])
    prompt = f"""
You are a coaching-style writing assistant. Create a single JSON object that fills ALL of these sections for a printable PDF.
Write in a friendly, grounded, non-therapeutic tone. Avoid medical/financial advice.

Input:
- Name: {first_name or ''}
- Email: {email or ''}
- Theme scores: {score_lines}
- Top themes: {', '.join(top3[:3]) if top3 else '-'}
- User free responses (bullets):
{free_blob or '(none)'}
- Future snapshot horizon (weeks): {horizon_weeks}

Return valid JSON with THIS schema (keep it compact, no markdown, no extra keys):
{{
  "archetype": "string (2-3 words)",
  "core_need": "string (<=12 words)",
  "signature_metaphor": "string (<=6 words)",
  "signature_sentence": "string (<=16 words)",
  "top_themes": ["string","string","string"],
  "from_your_words": {{
    "summary": "80-140 words distilling the user free text into a helpful insight."
  }},
  "why_now": "120-180 words explaining the leverage of acting now.",
  "future_snapshot": "150-220 words written as if {horizon_weeks} weeks later, vivid but realistic.",
  "signature_strengths": ["4 short bullets"],
  "energy_map": {{
    "energizers": ["3-4 bullets, +start"],
    "drainers": ["3-4 bullets, -start"]
  }},
  "hidden_tensions": ["1-3 bullets"],
  "watch_out": "one gentle blind spot (1-2 sentences)",
  "next_actions_7d": ["3 concrete actions (short lines)"],
  "implementation_if_then": ["3 lines 'If X then Y'"],
  "plan_1_week": ["7 lines Day 1..Day 7 ..."],
  "balancing_opportunity": ["call out 1-2 lowest themes w/ one-liners"],
  "keep_in_view": "one compact affirmation or reminder",
  "quotes": ["2-3 very short quotes (<=12 words)"],
  "tiny_progress": ["3 tiny milestones"],
  "token_hint": "optional string"
}}
    """.strip()
    return prompt

def call_openai(prompt, max_out):
    # Returns (sections_dict, usage_dict or None, raw_text_for_debug)
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return None, None, "NO_KEY"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Using Responses API; do NOT pass 'temperature' (some small models ignore/err).
        # Some models expect 'max_output_tokens' (not 'max_tokens').
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            max_output_tokens=max_out,
            # Not requesting strict JSON mode to maximize reliability; we will soft-parse below.
        )
        # Try to harvest usage if present
        usage = None
        try:
            usage = {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        except Exception:
            pass

        # Compose text output
        out = ""
        try:
            out = resp.output_text
        except Exception:
            # Fallback traverse
            try:
                out = "".join([c.text for c in resp.output if getattr(c, "type", "") == "output_text"])
            except Exception:
                out = ""

        # Try JSON parse
        try:
            parsed = json.loads(out)
            if isinstance(parsed, dict):
                return parsed, usage, out
        except Exception:
            pass

        # Soft-extract: find a JSON-looking region
        start = out.find("{")
        end = out.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(out[start:end+1])
                if isinstance(parsed, dict):
                    return parsed, usage, out
            except Exception:
                pass

        return None, usage, out  # will trigger template fallback
    except Exception as e:
        return None, None, f"ERR:{e}"

def sections_fallback(first_name, scores, top3, horizon_weeks):
    low = sorted(scores.items(), key=lambda kv: kv[1])[:2]
    low_lines = [f"{k}: try one tiny action this week." for k, _ in low]
    return {
        "archetype": "Curious Pathfinder",
        "core_need": "Small, repeatable progress that feels true",
        "signature_metaphor": "Compass with steady trail",
        "signature_sentence": "Tiny, true steps create real momentum.",
        "top_themes": top3[:3],
        "from_your_words": {"summary": f"{first_name or 'You'} showed interest in reflection and forward motion. Use gentle experiments to learn quickly and keep energy."},
        "why_now": "A short 4-week window is enough to test a new rhythm without pressure. Acting now produces quick feedback and keeps motivation alive.",
        "future_snapshot": f"In {horizon_weeks} weeks, you’ve piloted one small habit and invited a person into it. You feel clearer, lighter, and more consistent.",
        "signature_strengths": ["Curiosity that becomes action", "Learns by doing", "Cares about meaning", "Keeps going after small wins"],
        "energy_map": {
            "energizers": ["+ Short, contained experiments", "+ A partner to share it", "+ Clear finish lines", "+ Space to reflect 5 min"],
            "drainers": ["- Open-ended tasks", "- Messy expectations", "- Too many goals at once", "- Low sleep/overload"]
        },
        "hidden_tensions": ["Variety vs follow-through", "Spontaneity vs clarity for others"],
        "watch_out": "Excitement can outrun structure. Add one small container (timebox or checklist) so good intentions finish.",
        "next_actions_7d": ["Pick one mini-project", "Invite one person", "Capture one lesson + one gratitude"],
        "implementation_if_then": [
            "If I feel scattered, then I choose the smallest next action.",
            "If someone hesitates, then I simplify and offer an easy opt-out.",
            "If I miss a day, then I reschedule within three days."
        ],
        "plan_1_week": [
            "Day 1: Choose the week’s mini-project.",
            "Day 2: Invite someone; set simple logistics.",
            "Day 3: Prepare one-line purpose & backup.",
            "Day 4: Do it (2–3 hours).",
            "Day 5: Send a quick thank-you or highlight.",
            "Day 6: Reflect 5 minutes: one lesson & joy.",
            "Day 7: Rest; add two ideas to the list.",
        ],
        "balancing_opportunity": low_lines,
        "keep_in_view": "I move with purpose, lightly.",
        "quotes": ["“Small steps, big direction.”", "“Clarity loves action.”"],
        "tiny_progress": ["Choose + invite", "Do + note one lesson", "Schedule next week’s slot"]
    }

# ---------- PDF helpers (FPDF v1, latin-1 safe) ----------
class PDF(FPDF):
    def header(self):
        pass  # full header handled in body

def setf(pdf, style="", size=12):
    pdf.set_font("Helvetica", style, size)

def h1(pdf, txt):
    setf(pdf, "B", 24); pdf.cell(0, 12, latin1(txt), ln=1)

def h2(pdf, txt, sub=""):
    setf(pdf, "B", 16); pdf.cell(0, 8, latin1(txt), ln=1)
    if sub:
        setf(pdf, "", 11); pdf.multi_cell(0, 6, latin1(wrap(sub)))

def mc(pdf, txt, h=6):
    setf(pdf, "", 12); pdf.multi_cell(0, h, latin1(wrap(txt)))

def bullet_list(pdf, items):
    setf(pdf, "", 12)
    for it in items or []:
        pdf.cell(5, 6, u"\u2022".encode("latin-1", "ignore").decode("latin-1"), ln=0)  # simple bullet
        pdf.multi_cell(0, 6, latin1(wrap(str(it))))

def section_break(pdf, title, sub=""):
    pdf.ln(4)
    h2(pdf, title, sub)
    pdf.ln(1)

def add_logo_and_title(pdf, first_name, email):
    # Logo on top; then spacing; then title — no overlap
    logo_path = base_dir() / LOGO_FILE
    if logo_path.exists():
        try:
            # Convert to PNG in-memory (webp safe) then place
            im = Image.open(str(logo_path)).convert("RGBA")
            tmp = base_dir() / "._logo_tmp.png"
            im.save(tmp, "PNG")
            pdf.image(str(tmp), x=12, y=10, w=28)
            os.remove(tmp)
        except Exception:
            pass
    pdf.set_y(12 + 30)  # push below logo
    h1(pdf, "Life Minus Work - Reflection Report")
    setf(pdf, "", 12)
    pdf.cell(0, 6, latin1(f"Hi {first_name or 'there'},"), ln=1)
    pdf.cell(0, 6, latin1(f"Date: {datetime.date.today().isoformat()}"), ln=1)
    if email:
        pdf.cell(0, 6, latin1(f"Email: {email}"), ln=1)
    pdf.ln(2)

def theme_bars(pdf, scores):
    # simple horizontal bars
    section_break(pdf, "Your Theme Snapshot",
                  "A quick visual of where your energy clusters today.")
    maxv = max(1, max(scores.values()))
    for k in THEMES:
        v = scores.get(k, 0)
        setf(pdf, "", 14); pdf.cell(35, 8, latin1(k))
        bw = 110 * (v / maxv) if maxv > 0 else 0
        y = pdf.get_y()
        x = pdf.get_x()
        pdf.set_fill_color(40, 130, 255)
        pdf.rect(x, y + 2, bw, 6, "F")
        pdf.set_xy(x + 115, y)
        setf(pdf, "B", 14); pdf.cell(0, 8, latin1(str(v)), ln=1)
    pdf.ln(2)

def checkbox_line(pdf, text, line_height=8.0):
    x = pdf.get_x(); y = pdf.get_y()
    box = 4.5
    pdf.rect(x, y + 2, box, box)
    pdf.set_xy(x + box + 3, y)
    mc(pdf, text, h=line_height)

def signature_week_block(pdf, steps):
    section_break(pdf, "Signature Week - At a glance",
                  "A simple plan you can print or screenshot. Check items off as you go.")
    for s in steps or []:
        checkbox_line(pdf, s)

def tiny_progress_block(pdf, milestones):
    section_break(pdf, "Tiny Progress Tracker", "Three tiny milestones you can celebrate this week.")
    for m in milestones or []:
        checkbox_line(pdf, m)

def make_pdf_bytes(first_name, email, scores, top3, ai, horizon_weeks):
    pdf = PDF(orientation="P", unit="mm", format="Letter")
    pdf.set_margins(14, 12, 14)
    pdf.add_page()

    # Cover / greeting
    add_logo_and_title(pdf, first_name, email)

    # Identity block
    section_break(pdf, "Archetype", "Quick shorthand to remember your current pattern.")
    mc(pdf, ai.get("archetype", "Curious Pathfinder"))
    section_break(pdf, "Core Need", "What you’re really trying to feel more of right now.")
    mc(pdf, ai.get("core_need", "Small, repeatable progress that feels true."))
    section_break(pdf, "Signature Metaphor", "A tiny image to anchor the vibe.")
    mc(pdf, ai.get("signature_metaphor", "Compass with steady trail"))
    section_break(pdf, "Signature Sentence", "Your north star in one line.")
    mc(pdf, ai.get("signature_sentence", "Tiny, true steps create real momentum."))

    section_break(pdf, "Top Themes", "Your leading energies this round.")
    mc(pdf, ", ".join(top3[:3] or []))

    theme_bars(pdf, scores)

    # From your words (NO Keepers)
    fyw = ai.get("from_your_words") or {}
    if fyw.get("summary"):
        section_break(pdf, "From your words", "We pulled a few cues from what you typed.")
        mc(pdf, fyw["summary"])

    # Why now + Future snapshot
    section_break(pdf, "Why this matters now", "A little leverage goes a long way this month.")
    mc(pdf, ai.get("why_now", "Acting now gives quick feedback and keeps motivation alive."))
    section_break(pdf, f"Future Snapshot – {horizon_weeks} weeks",
                  "Imagine looking back after a short, focused experiment.")
    mc(pdf, ai.get("future_snapshot", "You’ll feel clearer and more consistent."))

    # Strengths / Energy Map
    section_break(pdf, "Signature strengths", "Levers you can lean on immediately.")
    bullet_list(pdf, ai.get("signature_strengths", []))
    section_break(pdf, "Energy map", "What tends to give/steal your energy.")
    mc(pdf, "Energizers")
    bullet_list(pdf, ai.get("energy_map", {}).get("energizers", []))
    mc(pdf, "\nDrainers")
    bullet_list(pdf, ai.get("energy_map", {}).get("drainers", []))

    # Tensions / Watch-out
    section_break(pdf, "Hidden tensions", "Natural trade-offs to hold lightly.")
    bullet_list(pdf, ai.get("hidden_tensions", []))
    section_break(pdf, "Watch-out (gentle blind spot)", "A tiny constraint that keeps results clean.")
    mc(pdf, ai.get("watch_out", ""))

    # Actions / If-Then / Plan
    section_break(pdf, "3 next-step actions (7 days)", "Keep these short and doable.")
    bullet_list(pdf, ai.get("next_actions_7d", []))
    section_break(pdf, "Implementation intentions (If-Then)", "Pre-decide what happens when life happens.")
    bullet_list(pdf, ai.get("implementation_if_then", []))
    section_break(pdf, "1-week gentle plan", "A light structure you can follow as-is.")
    bullet_list(pdf, ai.get("plan_1_week", []))

    # Balancing + Keep in view
    section_break(pdf, "Balancing Opportunity", "A nudge for the 1–2 lowest themes.")
    bullet_list(pdf, ai.get("balancing_opportunity", []))
    section_break(pdf, "Keep this in view", "A tiny reminder to anchor your week.")
    mc(pdf, ai.get("keep_in_view", ""))

    # Quotes (optional)
    if ai.get("quotes"):
        pdf.ln(2)
        setf(pdf, "I", 12)
        for q in ai["quotes"]:
            mc(pdf, f"“{q.strip('\"')}”")

    pdf.ln(4)
    setf(pdf, "B", 11)
    mc(pdf, "Next Page: Printable ‘Signature Week - At a glance’ + Tiny Progress Tracker")

    # Printable page
    pdf.add_page()
    signature_week_block(pdf, ai.get("plan_1_week", []))
    pdf.ln(2)
    tiny_progress_block(pdf, ai.get("tiny_progress", []))

    pdf.ln(6)
    setf(pdf, "", 10); mc(pdf, f"Requested for: {email or '-'}")
    pdf.ln(4)
    setf(pdf, "", 9)
    mc(pdf, "Life Minus Work * This report is a starting point for reflection. Nothing here is medical or financial advice.")

    # bytes
    return pdf.output(dest="S").encode("latin-1")

# ---------- Streamlit UI ----------
st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="centered")
st.title("Answer 15 questions, add your own reflections, and instantly download a personalized PDF summary.")

# Diagnostics (toggle)
with st.expander("AI status (debug)", expanded=False):
    st.write("AI enabled:", bool(os.getenv("OPENAI_API_KEY", "")))
    st.write("Model:", OPENAI_MODEL)
    st.write("Max tokens:", MAX_OUTPUT_TOKENS_HIGH, "(fallback", MAX_OUTPUT_TOKENS_FALLBACK, ")")
    if st.button("Test OpenAI now"):
        p = "Reply with exactly: OK"
        sec, usage, raw = call_openai(p, 64)
        st.info(f"OK — via Responses API. Output: {('OK' if (raw or '').strip()=='OK' else raw[:300])}")
        if usage: st.caption(f"usage: in={usage.get('input_tokens')} out={usage.get('output_tokens')} total={usage.get('total_tokens')}")

# Load questions
questions, themes = load_questions(QUESTIONS_FILE)

# Session scaffolding
if "answers_by_qid" not in st.session_state:
    st.session_state["answers_by_qid"] = {}
answers_by_qid = st.session_state["answers_by_qid"]

# Intro
st.subheader("Step 1: About you")
first_name = st.text_input("First name (for a personal greeting)", value=st.session_state.get("first_name",""))
st.session_state["first_name"] = first_name

horizon_weeks = st.slider("Future Snapshot Horizon (weeks)",
                          min_value=2, max_value=12, value=4, step=1,
                          help="We’ll write a short ‘as if it’s done’ snapshot this far into the future.")

st.write("---")
st.subheader("Step 2: 15 Questions")

for i, q in enumerate(questions, start=1):
    st.markdown(f"**Q{i}. {q['text']}**")
    qid = q["id"]
    # ensure container state
    if qid not in answers_by_qid:
        answers_by_qid[qid] = {"choice_index": None, "free": ""}

    # radio
    labels = [c["label"] for c in q["choices"]] + ["✍️ I’ll write my own answer"]
    default_idx = answers_by_qid[qid]["choice_index"]
    radio_value = st.radio(
        key=choice_key(qid),
        label="",
        options=list(range(len(labels))),
        format_func=lambda idx: labels[idx],
        index= default_idx if default_idx is not None else 0,
        horizontal=False,
    )
    answers_by_qid[qid]["choice_index"] = radio_value if radio_value < len(q["choices"]) else None

    # free text toggle
    want_free = (radio_value == len(labels)-1)
    if want_free:
        # text_area with stable key
        new_free = st.text_area(
            "Your words (optional, helps tailor the report)",
            key=free_key(qid),
            value=answers_by_qid[qid].get("free",""),
            placeholder="Type here… (on mobile, tap outside to save)",
            height=90,
        )
        answers_by_qid[qid]["free"] = new_free
    else:
        # Clear any stale free text
        answers_by_qid[qid]["free"] = ""

st.write("---")
st.subheader("Email & Download")
email = st.text_input("Your email (for your download link)", value=st.session_state.get("email",""))
st.session_state["email"] = email

st.caption("Heads-up: generating your PDF can take **up to ~1 minute** (AI + rendering).")

# Compute now so we can show top themes before button
scores = compute_scores(answers_by_qid, questions)
top3 = rank_themes(scores)

# Gather free text for AI
free_texts = [answers_by_qid[q["id"]]["free"] for q in questions if answers_by_qid[q["id"]].get("free")]

if st.button("Generate My Personalized Report", type="primary"):
    # 1) AI sections
    prompt = make_ai_payload(first_name, email, scores, top3, free_texts, horizon_weeks)
    sections, usage, raw = call_openai(prompt, MAX_OUTPUT_TOKENS_HIGH)
    if sections is None:
        # One retry with lower cap
        sections, usage, raw = call_openai(prompt, MAX_OUTPUT_TOKENS_FALLBACK)
    if sections is None:
        sections = sections_fallback(first_name, scores, top3, horizon_weeks)
        st.warning("AI could not generate JSON this run — using a concise template instead.")

    # 2) PDF bytes
    try:
        pdf_bytes = make_pdf_bytes(first_name, email, scores, top3, sections, horizon_weeks)
    except Exception as e:
        st.error(f"PDF error: {e}")
        st.stop()

    # 3) Offer download; also write temp CSV (Cloud-safe)
    st.download_button(
        "Download PDF now",
        data=pdf_bytes,
        file_name="LifeMinusWork_Reflection_Report.pdf",
        mime="application/pdf",
    )

    # Token usage display
    if usage:
        st.caption(
            f"Token usage (one run) — model: {OPENAI_MODEL} | "
            f"in: {usage.get('input_tokens')} out: {usage.get('output_tokens')} total: {usage.get('total_tokens')}"
        )

    # Save a light CSV row (ephemeral on Streamlit Cloud)
    try:
        row = {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "first_name": first_name,
            "email": email,
            "scores": json.dumps(scores),
            "top3": json.dumps(top3[:3]),
        }
        csv_path = "/tmp/responses.csv"
        exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(list(row.keys()))
            w.writerow(list(row.values()))
        st.caption("Saved to /tmp/responses.csv (ephemeral).")
    except Exception as e:
        st.caption(f"Could not save responses (demo only). {e}")
