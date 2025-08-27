import os, sys, re, json, hashlib, unicodedata, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import streamlit as st
from PIL import Image
from fpdf import FPDF

# ========== OpenAI ==========
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP_TITLE    = "Life Minus Work — Reflection Quiz (15 questions)"
REPORT_TITLE = "Your Reflection Report"
THEMES       = ["Identity", "Growth", "Connection", "Peace", "Adventure", "Contribution"]

st.set_page_config(page_title=APP_TITLE, page_icon="✨", layout="centered")

def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:  # type: ignore[attr-defined]
            return str(st.secrets[name])  # type: ignore[attr-defined]
    except Exception:
        pass
    return os.getenv(name, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

USE_AI = bool(OPENAI_API_KEY and OpenAI)
HIGH_MODEL = get_secret("OPENAI_HIGH_MODEL", "gpt-5-mini")
MAX_TOK_HIGH = int(get_secret("MAX_OUTPUT_TOKENS_HIGH", "8000"))
FALLBACK_CAP = int(get_secret("MAX_OUTPUT_TOKENS_FALLBACK", "7000"))

# --------- Diagnostics (temporary) ----------
with st.expander("🔧 Diagnostics (temporary)", expanded=False):
    st.write("Python:", sys.version.split()[0])
    here = Path(__file__).parent
    st.write("__file__:", __file__)
    st.write("cwd:", os.getcwd())
    try:
        st.write("Files near app.py:", [p.name for p in here.iterdir()])
    except Exception as e:
        st.write("Dir list failed:", e)
    masked = (OPENAI_API_KEY[:4] + "…" + OPENAI_API_KEY[-4:]) if OPENAI_API_KEY else "None"
    st.write("OPENAI_API_KEY present:", bool(OPENAI_API_KEY), "| key:", masked if OPENAI_API_KEY else "—")
    st.write("Model:", HIGH_MODEL, "| MAX_TOK_HIGH:", MAX_TOK_HIGH, "| FALLBACK_CAP:", FALLBACK_CAP)

# --------- Helpers: text safety ----------
def _ascii_only(s: str) -> str:
    return (s.replace("’", "'").replace("‘", "'")
             .replace("“", '"').replace("”", '"')
             .replace("–", "-").replace("—", "-")
             .replace("…", "...").replace("•", "*"))

def clean_text(s: str, max_len: int = 1000, ascii_fallback: bool = False) -> str:
    if not s:
        return ""
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)  # strip control chars
    tokens = []
    for t in s.split():
        tokens.append(t if len(t) <= max_len else t[:max_len] + "…")
    s = " ".join(tokens)
    if ascii_fallback:
        s = _ascii_only(s)
    return s

def to_bytes(x: Any) -> bytes:
    if x is None: return b""
    if isinstance(x, (bytes, bytearray)): return bytes(x)
    if isinstance(x, str): return x.encode("latin-1", errors="replace")
    if hasattr(x, "read"):
        try:
            data = x.read()
            return data if isinstance(data, bytes) else bytes(str(data), "utf-8", "ignore")
        except Exception:
            return b""
    try:
        return bytes(x)
    except Exception:
        return bytes(str(x), "utf-8", "ignore")

# --------- PDF MultiCell wrapper ----------
def mc(pdf: "FPDF", text: str, h: float = 6, unicode_ok: bool = False):
    try:
        w = float(pdf.w) - float(pdf.l_margin) - float(pdf.r_margin)
    except Exception:
        w = 180.0
    if w <= 0: w = 180.0
    s = clean_text((text or "").replace("\r\n", "\n").replace("\r", "\n"),
                   ascii_fallback=not unicode_ok)
    try:
        pdf.multi_cell(w, h, s); return
    except Exception:
        pass
    try:
        s2 = clean_text(s, ascii_fallback=True)
        pdf.multi_cell(w, h, s2); return
    except Exception:
        pass
    try:
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(w, h, "[...content truncated...]")
    except Exception:
        return

def setf(pdf: FPDF, style: str = "", size: int = 12):
    pdf.set_font("Helvetica", style or "", size)

# --------- Logo ----------
def get_logo_png_path() -> Optional[str]:
    here = Path(__file__).parent
    candidates = [
        here / "logo.png",
        here / "Life-Minus-Work-Logo.png",
        here / "Life-Minus-Work-Logo.webp",
        here / "assets" / "logo.png",
        here / "assets" / "Life-Minus-Work-Logo.png",
        here / "assets" / "Life-Minus-Work-Logo.webp",
    ]
    for p in candidates:
        if p.exists():
            if p.suffix.lower() == ".png":
                return str(p)
            if p.suffix.lower() == ".webp":
                try:
                    img = Image.open(p).convert("RGB")
                    out = Path("/tmp/logo.png")
                    img.save(out, format="PNG")
                    return str(out)
                except Exception:
                    return None
    return None

# --------- Questions loading w/ stable version ----------
@st.cache_data(show_spinner=False)
def load_questions_cached(filename: str = "questions.json"):
    base_dir = Path(__file__).parent
    path = base_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"{filename} not found at {path}")
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    # Stable hash of questions content to detect meaningful changes
    version = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return data["questions"], data.get("themes", []), version

def load_questions(filename="questions.json"):
    try:
        return load_questions_cached(filename)
    except FileNotFoundError as e:
        st.error(f"Could not find {filename}. Make sure it's next to app.py.")
        here = Path(__file__).parent
        try:
            st.caption("Directory listing:")
            for p in here.iterdir():
                st.write("-", p.name)
        except Exception:
            pass
        st.stop()

# --------- Scoring ----------
def compute_scores(answers_by_qid: dict, questions: list) -> Dict[str, int]:
    scores = {t: 0 for t in THEMES}
    for q in questions:
        qid = q["id"]
        choice_idx = answers_by_qid.get(qid, {}).get("choice_idx")
        if choice_idx is None:
            continue
        try:
            choice = q["choices"][choice_idx]
        except (IndexError, KeyError, TypeError):
            continue
        for theme, val in choice.get("weights", {}).items():
            scores[theme] = scores.get(theme, 0) + val
    return scores

def top_themes(scores: Dict[str, int], k: int = 3) -> List[str]:
    return [name for name, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]

def balancing_suggestion(theme: str) -> str:
    suggestions = {
        "Identity": "Choose one tiny ritual that reflects who you are becoming.",
        "Growth": "Pick one skill and practice 15 minutes today.",
        "Connection": "Send a 3-line check-in to someone who matters.",
        "Peace": "Name a 10-minute wind-down you will repeat daily.",
        "Adventure": "Plan a 30–60 minute micro-adventure within 7 days.",
        "Contribution": "Offer one concrete act of help this week.",
    }
    return suggestions.get(theme, "Take one small, visible step this week.")

# --------- OpenAI (strict JSON) ----------
def _call_openai_json(model: str, system: str, user: str, cap: int):
    if not (USE_AI and OpenAI):
        raise RuntimeError("OpenAI not configured")
    client = OpenAI()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        max_completion_tokens=cap,
        response_format={"type": "json_object"},
    )
    content = r.choices[0].message.content if r.choices else ""
    usage = getattr(r, "usage", None)
    usage_dict = None
    if usage is not None:
        usage_dict = {
            "input": getattr(usage, "prompt_tokens", None),
            "output": getattr(usage, "completion_tokens", None),
            "total": getattr(usage, "total_tokens", None),
        }
    return content, usage_dict, "chat+rf_mct"

def ai_sections_and_weights(scores, top3, free_responses, first_name, horizon_weeks=4) -> Optional[dict]:
    if not USE_AI:
        return None
    st.session_state["ai_debug"] = {}
    try:
        packed, allowed_ids = [], []
        for i, fr in enumerate(free_responses or []):
            if not isinstance(fr, dict):
                continue
            ans = str(fr.get("answer", "")).strip()
            if not ans:
                continue
            qid = str(fr.get("id") or fr.get("qid") or f"free_{i+1}")
            qtxt = str(fr.get("question", "")).strip()[:160]
            allowed_ids.append(qid)
            packed.append({"id": qid, "q": qtxt, "a": ans[:280]})

        score_lines = ", ".join([f"{k}: {v}" for k, v in scores.items()])

        prompt = f"""
You are a warm, practical life coach. Return STRICT JSON ONLY matching this schema:

{{
  "archetype": string,
  "core_need": string,
  "deep_insight": string,              // 400–600 words
  "why_now": string,                   // 120–180 words
  "strengths": [string, ...],          // 4–6
  "energizers": [string, ...],         // 4
  "drainers": [string, ...],           // 4
  "tensions": [string, ...],           // 2–3
  "blindspot": string,
  "actions": [string, string, string], // EXACTLY 3
  "if_then": [string, string, string], // EXACTLY 3
  "weekly_plan": [string, string, string, string, string, string, string], // 7
  "affirmation": string,               // <= 15 words
  "quote": string,                     // <= 20 words
  "signature_metaphor": string,        // <= 12 words
  "signature_sentence": string,        // <= 20 words
  "top_theme_boosters": [string, ...], // <= 4
  "pitfalls": [string, ...],           // <= 4
  "future_snapshot": string,           // 150–220 words, second-person, present tense, as if {horizon_weeks} weeks later
  "from_words": {{
    "themes": [string, string, string],              // 3
    "quotes": [string, string, string],              // 2–3, <= 12 words each
    "insight": string,                               // 80–120 words
    "ritual": string,
    "relationship_moment": string,
    "stress_reset": string
  }},
  "micro_pledge": string,              // first-person, <= 28 words
  "weights": {{
    "<question_id>": {{"Identity": int, "Growth": int, "Connection": int, "Peace": int, "Adventure": int, "Contribution": int}}
  }}
}}

Return ONLY JSON. NO markdown, no preface.

User first name: {first_name or 'Friend'}.
Theme scores: {score_lines}.
Top 3 themes: {", ".join(top3)}.
Horizon weeks: {horizon_weeks}.
Free-text answers (array of objects with id, q, a): {json.dumps(packed, ensure_ascii=False)}
IMPORTANT: Only use these IDs as keys inside "weights": {json.dumps(allowed_ids, ensure_ascii=False)}
Tone: empathetic, encouraging, plain language. No medical claims.
"""
        system = "Reply with helpful coaching guidance as STRICT JSON only."
        tries = [MAX_TOK_HIGH, FALLBACK_CAP, 6000, 4000, 2500, 1200]
        last_err = None
        for cap in tries:
            try:
                raw, usage, path = _call_openai_json(HIGH_MODEL, system, prompt, cap)
                raw = (raw or "").strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```[a-zA-Z]*\n", "", raw)
                    raw = re.sub(r"\n```$", "", raw)
                data = None
                try:
                    data = json.loads(raw)
                except Exception:
                    if "{" in raw and "}" in raw:
                        raw2 = raw[raw.find("{"): raw.rfind("}") + 1]
                        data = json.loads(raw2)
                if not isinstance(data, dict):
                    raise ValueError("No JSON object found in completion.")

                st.session_state["ai_debug"] = {
                    "path": path, "cap_used": cap, "raw_head": raw[:800], "raw_len": len(raw),
                }
                if usage:
                    st.session_state["token_usage"] = {
                        "model": HIGH_MODEL, "path": path, "cap_used": cap,
                        "input": usage.get("input"), "output": usage.get("output"),
                        "total": usage.get("total"),
                        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
                    }
                try:
                    Path("/tmp/last_ai.json").write_text(raw, encoding="utf-8")
                except Exception:
                    pass

                sg = lambda k: str(data.get(k, "") or "")
                out = {
                    "archetype": sg("archetype"),
                    "core_need": sg("core_need"),
                    "deep_insight": sg("deep_insight"),
                    "why_now": sg("why_now"),
                    "strengths": [str(x) for x in (data.get("strengths") or [])][:6],
                    "energizers": [str(x) for x in (data.get("energizers") or [])][:4],
                    "drainers": [str(x) for x in (data.get("drainers") or [])][:4],
                    "tensions": [str(x) for x in (data.get("tensions") or [])][:3],
                    "blindspot": sg("blindspot"),
                    "actions": [str(x) for x in (data.get("actions") or [])][:3],
                    "if_then": [str(x) for x in (data.get("if_then") or [])][:3],
                    "weekly_plan": [str(x) for x in (data.get("weekly_plan") or [])][:7],
                    "affirmation": sg("affirmation"),
                    "quote": sg("quote"),
                    "signature_metaphor": sg("signature_metaphor"),
                    "signature_sentence": sg("signature_sentence"),
                    "top_theme_boosters": [str(x) for x in (data.get("top_theme_boosters") or [])][:4],
                    "pitfalls": [str(x) for x in (data.get("pitfalls") or [])][:4],
                    "future_snapshot": sg("future_snapshot"),
                    "from_words": {},
                    "micro_pledge": sg("micro_pledge"),
                    "weights": {},
                }

                fw = data.get("from_words") or {}
                if isinstance(fw, dict) and len(fw) == 1:
                    only_key = list(fw.keys())[0]
                    if ("themes(" in only_key) or ("quotes(" in only_key) or ("relationship_moment" in only_key):
                        fw = {}
                elif not isinstance(fw, dict):
                    try:
                        maybe = json.loads(str(fw))
                        fw = maybe if isinstance(maybe, dict) else {}
                    except Exception:
                        fw = {}
                out["from_words"] = {
                    "themes": [str(x) for x in (fw.get("themes") or [])][:3],
                    "quotes": [str(x) for x in (fw.get("quotes") or [])][:3],
                    "insight": str(fw.get("insight", "")),
                    "ritual": str(fw.get("ritual", "")),
                    "relationship_moment": str(fw.get("relationship_moment", "")),
                    "stress_reset": str(fw.get("stress_reset", "")),
                }

                weights = data.get("weights") or {}
                if isinstance(weights, dict):
                    clean_w = {}
                    for qid, w in weights.items():
                        if not isinstance(w, dict):
                            continue
                        m = {}
                        for theme, val in w.items():
                            if theme in THEMES:
                                try:
                                    iv = int(val)
                                    iv = max(-2, min(2, iv))
                                    m[theme] = iv
                                except Exception:
                                    pass
                        if m:
                            clean_w[str(qid)] = m
                    out["weights"] = clean_w
                return out
            except Exception as e:
                last_err = e
                continue
        st.session_state["ai_debug"] = {"error": f"{type(last_err).__name__}: {last_err}"}
        return None
    except Exception as e:
        st.session_state["ai_debug"] = {"fatal": f"{type(e).__name__}: {e}"}
        return None

# --------- PDF helpers ----------
def draw_scores_barchart(pdf: FPDF, scores: Dict[str, int]):
    """Render a simple horizontal bar chart.
    - Positive scores get a bar.
    - Zero/negative scores show no bar, but we still print the value (e.g., -4).
    """
    setf(pdf, "B", 14); mc(pdf, "Your Theme Snapshot")
    setf(pdf, "", 12)

    # Scale only by the largest *positive* value so negatives don’t invert bars.
    positive_vals = [v for v in scores.values() if v > 0]
    max_pos = max(positive_vals) if positive_vals else 1

    bar_w_max = 120
    x_left = pdf.get_x() + 10
    y = pdf.get_y()

    for theme in THEMES:
        val = int(scores.get(theme, 0))

        # Label on the left
        pdf.set_xy(x_left, y)
        pdf.cell(35, 6, _ascii_only(theme))

        # Bar area baseline
        bar_x = x_left + 38
        bar_h = 4.5

        # Only draw a bar for positive values
        if val > 0:
            bar_w = (val / max_pos) * bar_w_max
            pdf.set_fill_color(30, 144, 255)
            pdf.rect(bar_x, y + 1.3, bar_w, bar_h, "F")
            num_x = bar_x + bar_w + 2
        else:
            # No bar for zero/negative; place the number just after the baseline start
            num_x = bar_x + 2

        # Print the numeric value (can be negative)
        pdf.set_xy(num_x, y)
        pdf.cell(0, 6, _ascii_only(str(val)))

        y += 7

    pdf.set_y(y + 4)

def paragraph(pdf: FPDF, title: str, body: str):
    setf(pdf, "B", 14); mc(pdf, title)
    setf(pdf, "", 12)
    for line in str(body).split("\n"):
        mc(pdf, line)
    pdf.ln(2)

def checkbox_line(pdf: FPDF, text: str):
    x = pdf.get_x(); y = pdf.get_y()
    pdf.rect(x, y + 1.5, 4, 4); pdf.set_x(x + 6); mc(pdf, text)

def label_value(pdf: FPDF, label: str, value: str):
    setf(pdf, "B", 12); mc(pdf, label)
    setf(pdf, "", 12);  mc(pdf, value)

def future_callout(pdf: FPDF, weeks: int, text: str):
    pdf.set_text_color(30, 60, 120)
    setf(pdf, "B", 14); mc(pdf, f"Future Snapshot — {weeks} weeks")
    pdf.set_text_color(0, 0, 0)
    setf(pdf, "I", 12); mc(pdf, text); pdf.ln(2)
    setf(pdf, "", 12)

def left_bar_callout(pdf: FPDF, title: str, body: str, bullets=None):
    if bullets is None:
        bullets = []
    x = pdf.get_x(); y = pdf.get_y()
    pdf.set_fill_color(30, 144, 255)
    pdf.rect(x, y, 2, 6, "F")
    pdf.set_x(x + 4)
    setf(pdf, "B", 13); mc(pdf, title)
    pdf.set_x(x + 4)
    setf(pdf, "", 12); mc(pdf, body)
    for b in bullets:
        pdf.set_x(x + 4); pdf.cell(4, 6, "*"); mc(pdf, b)
    pdf.ln(1)

def make_pdf_bytes(first_name: str, email: str, scores: Dict[str,int], top3: List[str],
                   sections: dict, free_responses: List[dict], logo_path: Optional[str]) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    setf(pdf, "B", 18)

    if logo_path:
        try:
            pdf.image(logo_path, w=40); pdf.ln(2)
        except Exception:
            pass

    mc(pdf, REPORT_TITLE)
    setf(pdf, "", 12)
    today = datetime.date.today().strftime("%d %b %Y")
    greet = f"Hi {first_name}," if first_name else "Hello,"
    mc(pdf, greet); mc(pdf, f"Date: {today}")
    if email: mc(pdf, f"Email: {email}")
    pdf.ln(1)

    if sections.get("archetype") or sections.get("core_need"):
        label_value(pdf, "Archetype", sections.get("archetype","") or "—")
        label_value(pdf, "Core Need", sections.get("core_need","") or "—")
        if sections.get("signature_metaphor"):
            label_value(pdf, "Signature Metaphor", sections.get("signature_metaphor",""))
        if sections.get("signature_sentence"):
            label_value(pdf, "Signature Sentence", sections.get("signature_sentence",""))
        pdf.ln(1)

    setf(pdf, "B", 14); mc(pdf, "Top Themes")
    setf(pdf, "", 12);  mc(pdf, ", ".join(top3)); pdf.ln(1)

    draw_scores_barchart(pdf, scores)

    fw = sections.get("from_words") or {}
    if isinstance(fw, dict) and (fw.get("insight") or fw.get("themes") or fw.get("quotes")):
        quotes = [f'"{_ascii_only(q)}"' for q in fw.get("quotes", []) if q]
        left_bar_callout(pdf, "From your words", _ascii_only(fw.get("insight","")), bullets=quotes)
        keep = [("Daily ritual", fw.get("ritual","")),
                ("Connection moment", fw.get("relationship_moment","")),
                ("Stress reset", fw.get("stress_reset",""))]
        if any(v for _, v in keep):
            setf(pdf, "B", 12); mc(pdf, "One-liners to keep")
            setf(pdf, "", 12)
            for lbl, val in keep:
                if val: mc(pdf, f"{lbl}: {_ascii_only(val)}")
            pdf.ln(1)
    if sections.get("micro_pledge"):
        label_value(pdf, "Personal pledge", _ascii_only(sections["micro_pledge"])); pdf.ln(1)

    if sections.get("deep_insight"):
        paragraph(pdf, "What this really says about you", _ascii_only(sections["deep_insight"]))
    if sections.get("why_now"):
        label_value(pdf, "Why this matters now", _ascii_only(sections["why_now"])); pdf.ln(1)

    if sections.get("future_snapshot"):
        future_callout(pdf, sections.get("horizon_weeks", 4), _ascii_only(sections["future_snapshot"]))

    if sections.get("strengths"):
        setf(pdf, "B", 14); mc(pdf, "Signature strengths")
        setf(pdf, "", 12)
        for s in sections["strengths"]:
            pdf.cell(4, 6, "*"); mc(pdf, _ascii_only(s))
        pdf.ln(1)

    if sections.get("energizers") or sections.get("drainers"):
        setf(pdf, "B", 14); mc(pdf, "Energy map")
        setf(pdf, "B", 12); mc(pdf, "Energizers")
        setf(pdf, "", 12)
        for e in sections.get("energizers", []):
            pdf.cell(4, 6, "+"); mc(pdf, _ascii_only(e))
        pdf.ln(1)
        setf(pdf, "B", 12); mc(pdf, "Drainers")
        setf(pdf, "", 12)
        for d in sections.get("drainers", []):
            pdf.cell(4, 6, "-"); mc(pdf, _ascii_only(d))
        pdf.ln(1)

    if sections.get("tensions"):
        setf(pdf, "B", 14); mc(pdf, "Hidden tensions")
        setf(pdf, "", 12)
        for t in sections["tensions"]:
            pdf.cell(4, 6, "*"); mc(pdf, _ascii_only(t))
        pdf.ln(1)
    if sections.get("blindspot"):
        label_value(pdf, "Watch-out (gentle blind spot)", _ascii_only(sections["blindspot"])); pdf.ln(1)

    if sections.get("actions"):
        setf(pdf, "B", 14); mc(pdf, "3 next-step actions (7 days)")
        setf(pdf, "", 12)
        for a in sections["actions"]:
            checkbox_line(pdf, _ascii_only(a))
        pdf.ln(1)

    if sections.get("if_then"):
        setf(pdf, "B", 14); mc(pdf, "Implementation intentions (If–Then)")
        setf(pdf, "", 12)
        for it in sections.get("if_then", []):
            pdf.cell(4, 6, "*"); mc(pdf, _ascii_only(it))
        pdf.ln(1)

    if sections.get("weekly_plan"):
        setf(pdf, "B", 14); mc(pdf, "1-week gentle plan")
        setf(pdf, "", 12)
        for i, item in enumerate(sections["weekly_plan"][:7]):
            mc(pdf, f"Day {i+1}: {_ascii_only(item)}")
        pdf.ln(1)

    lows = [name for name, _ in sorted(scores.items(), key=lambda x: x[1])[:2]]
    if lows:
        setf(pdf, "B", 14); mc(pdf, "Balancing Opportunity")
        setf(pdf, "", 12)
        for theme in lows:
            tip = balancing_suggestion(theme)
            mc(pdf, f"{theme}: {tip}")
        pdf.ln(1)

    if sections.get("affirmation") or sections.get("quote"):
        setf(pdf, "B", 12); mc(pdf, "Keep this in view")
        setf(pdf, "I", 11)
        if sections.get("affirmation"):
            mc(pdf, f"Affirmation: {_ascii_only(sections['affirmation'])}")
        if sections.get("quote"):
            mc(pdf, f"\"{_ascii_only(sections['quote'])}\"")
        pdf.ln(2); setf(pdf, "", 12)

    if free_responses:
        setf(pdf, "B", 14); mc(pdf, "Your words we heard")
        setf(pdf, "", 12)
        for fr in free_responses:
            if not fr.get("answer"): continue
            mc(pdf, f"* {fr.get('question','')}")
            mc(pdf, f"  {_ascii_only(fr.get('answer',''))}")
            pdf.ln(1)

    pdf.ln(3)
    setf(pdf, "B", 12)
    mc(pdf, "On the next page: a printable 'Signature Week — At a glance' checklist you can use right away.")

    pdf.add_page()
    setf(pdf, "B", 16); mc(pdf, "Signature Week — At a glance")
    setf(pdf, "", 12)
    mc(pdf, "A simple plan you can print or screenshot. Check items off as you go.")
    pdf.ln(2)

    week_items = sections.get("weekly_plan") or []
    if not week_items:
        week_items = [f"Do one small action for {t}" for t in top3] + ["Reflect and set next step"]
    for i, item in enumerate(week_items[:7]):
        x = pdf.get_x(); y = pdf.get_y()
        pdf.rect(x, y + 1.5, 4, 4); pdf.set_x(x + 6)
        mc(pdf, f"Day {i+1}: {_ascii_only(item)}")

    pdf.ln(2)
    setf(pdf, "B", 14); mc(pdf, "Tiny Progress Tracker")
    setf(pdf, "", 12)
    milestones = sections.get("actions") or [
        "Choose one tiny step and schedule it.",
        "Tell a friend your plan for gentle accountability.",
        "Spend 20 minutes on your step and celebrate completion."
    ]
    for m in milestones[:3]:
        x = pdf.get_x(); y = pdf.get_y()
        pdf.rect(x, y + 1.5, 4, 4); pdf.set_x(x + 6)
        mc(pdf, _ascii_only(m))

    pdf.ln(2); setf(pdf, "I", 10); pdf.ln(2)
    mc(pdf, "Life Minus Work • This report is a starting point for reflection. Nothing here is medical or financial advice.")
    setf(pdf, "", 12)

    raw = pdf.output(dest="S")
    return raw.encode("latin-1", errors="replace") if isinstance(raw, str) else to_bytes(raw)

# --------- UI ----------
st.title(APP_TITLE)
st.write("Answer 15 questions, add your own reflections, and instantly download a personalized PDF summary.")

# First name
st.session_state.setdefault("first_name", "")
first_name = st.text_input("First name", st.session_state["first_name"])
if first_name:
    st.session_state["first_name"] = first_name.strip()

# Load questions
questions, _, q_version = load_questions()

# Stable answers store (by qid) + migrate if version changed
st.session_state.setdefault("questions_version", q_version)
st.session_state.setdefault("answers_by_qid", {})  # {qid: {"choice_idx": int|None, "free_text": str}}

if st.session_state["questions_version"] != q_version:
    # migrate by qid; keep any matches
    old = st.session_state["answers_by_qid"]
    new_map = {}
    existing_ids = {q["id"] for q in questions}
    for qid, val in old.items():
        if qid in existing_ids:
            new_map[qid] = {"choice_idx": val.get("choice_idx"), "free_text": val.get("free_text","")}
    st.session_state["answers_by_qid"] = new_map
    st.session_state["questions_version"] = q_version

# Personalization
with st.expander("Personalization options"):
    horizon_weeks = st.slider("Future snapshot horizon (weeks)", 2, 8, 4)

# === Questionnaire (no forms; write-in appears instantly) ===
st.subheader("Questions")

def _save_choice(qid, base_options, placeholder, write_in_label):
    choice_label = st.session_state.get(f"{qid}__choice")
    if choice_label == placeholder:
        prev = st.session_state["answers_by_qid"].get(qid, {})
        st.session_state["answers_by_qid"][qid] = {"choice_idx": None, "free_text": prev.get("free_text","")}
    elif choice_label == write_in_label:
        prev = st.session_state["answers_by_qid"].get(qid, {})
        st.session_state["answers_by_qid"][qid] = {"choice_idx": None, "free_text": prev.get("free_text","")}
    else:
        try:
            idx = base_options.index(choice_label)
        except ValueError:
            idx = None
        st.session_state["answers_by_qid"][qid] = {"choice_idx": idx, "free_text": ""}

def _save_free(qid):
    val = (st.session_state.get(f"{qid}__free") or "").strip()
    prev = st.session_state["answers_by_qid"].get(qid, {})
    if st.session_state.get(f"{qid}__choice") == "✍️ I'll write my own answer":
        st.session_state["answers_by_qid"][qid] = {"choice_idx": None, "free_text": val}
    else:
        st.session_state["answers_by_qid"][qid] = {"choice_idx": prev.get("choice_idx"), "free_text": ""}

for q in questions:
    qid = q["id"]
    st.markdown(f"### {q['text']}")

    base_options = [c["label"] for c in q["choices"]]
    write_in_label = "✍️ I'll write my own answer"
    placeholder = "— Select —"
    options = [placeholder] + base_options + [write_in_label]

    saved = st.session_state["answers_by_qid"].get(qid, {})
    saved_idx = saved.get("choice_idx", None)
    saved_free = saved.get("free_text", "")

    if isinstance(saved_idx, int) and 0 <= saved_idx < len(base_options):
        current_label = base_options[saved_idx]
    elif saved_free:
        current_label = write_in_label
    else:
        current_label = placeholder

    st.session_state.setdefault(f"{qid}__choice", current_label)

    st.radio(
        "Choose one:",
        options,
        key=f"{qid}__choice",
        on_change=_save_choice,
        args=(qid, base_options, placeholder, write_in_label),
        horizontal=False,
    )

    if st.session_state.get(f"{qid}__choice") == write_in_label:
        st.text_area(
            "Your answer",
            value=saved_free,
            key=f"{qid}__free",
            height=80,
            placeholder="Type your own response…",
            on_change=_save_free,
            args=(qid,),
        )

    st.divider()

# Optional “save” (inputs are already persisted in session_state)
if st.button("Save my answers"):
    st.success("Saved! Scroll down to generate your PDF when ready.")

# Email + consent (form only here is fine)
st.subheader("Email & Download")
with st.form("finish_form"):
    email_val = st.text_input("Your email (for your download link)", key="email_input", placeholder="you@example.com")
    consent_val = st.checkbox(
        "I agree to receive my results and occasional updates from Life Minus Work.",
        key="consent_input",
        value=st.session_state.get("consent_input", False),
    )
    submit_clicked = st.form_submit_button(
    "Generate My Personalized Report",
    help="This can take up to 1 minute"
)
    if submit_clicked:
        if not email_val or not consent_val:
            st.error("Please enter your email and give consent to continue.")
        else:
            st.session_state["email"] = email_val.strip()
            st.session_state["consent"] = True
            st.session_state["request_report"] = True
            st.toast("Generating your report…", icon="⏳")

# Generate on flag
if st.session_state.get("request_report"):
    st.session_state["request_report"] = False

    answers = st.session_state.get("answers_by_qid", {})
    scores = compute_scores(answers, questions)
    top3 = top_themes(scores, 3)

    free_responses = []
    for q in questions:
        qid = q["id"]
        a = answers.get(qid, {})
        if a and a.get("free_text"):
            free_responses.append({"id": qid, "question": q["text"], "answer": a["free_text"]})

    sections = {"weekly_plan": [], "actions": [], "from_words": {}, "weights": {}}
    if USE_AI:
        maybe = ai_sections_and_weights(
            scores, top3, free_responses, st.session_state.get("first_name", ""), horizon_weeks=4
        )
        dbg = st.session_state.get("ai_debug") or {}
        tok = st.session_state.get("token_usage") or {}

        with st.expander("AI generation details (debug)", expanded=False):
            if dbg:
                for k, v in dbg.items():
                    if k == "raw_head" and isinstance(v, str):
                        st.text_area("raw_head (first 800 chars)", v, height=200)
                    else:
                        st.write(f"{k}: {v}")
            p = Path("/tmp/last_ai.json")
            if p.exists():
                st.download_button(
                    "Download last_ai.json",
                    data=p.read_bytes(),
                    file_name="last_ai.json",
                    mime="application/json",
                )

        with st.expander("Token usage (one run)", expanded=True):
            if tok:
                st.write(
                    f"Model: {tok.get('model','?')} | path: {tok.get('path','?')} | cap_used: {tok.get('cap_used','?')}"
                )
                st.write(
                    f"Input tokens: {tok.get('input','?')} | Output tokens: {tok.get('output','?')} | Total: {tok.get('total','?')}"
                )
                st.caption(f"Timestamp: {tok.get('ts','?')}")
            else:
                st.write("No usage returned by the API (some paths/models omit it).")

        if maybe:
            sections.update(maybe)
            weights = sections.get("weights") or {}
            for qid, wmap in weights.items():
                for theme, delta in wmap.items():
                    scores[theme] = scores.get(theme, 0) + int(delta)
            sections["horizon_weeks"] = 4
        else:
            st.warning("AI could not generate JSON this run — using a concise template instead.")

    # minimal fallback if AI missing
    if not sections.get("deep_insight"):
        top1 = top3[0] if top3 else "what energizes you"
        sections.update({
            "deep_insight": f"Thank you for completing the Reflection Quiz, {st.session_state.get('first_name','Friend')}.",
            "actions": [
                "Choose one tiny step you can take this week.",
                "Tell a friend your plan—gentle accountability.",
                "Schedule 20 minutes for reflection or journaling.",
            ],
            "weekly_plan": [
                "Name your intention.",
                "15–20 minutes of learning or practice.",
                "Reach out to someone who energizes you.",
                "Take a calm walk or mindful pause.",
                "Do one small adventurous thing.",
                "Offer help or encouragement to someone.",
                "Review your week and set the next tiny step.",
            ],
            "future_snapshot": (
                f"It is 4 weeks later. You have stayed close to {top1}. A few tiny actions, repeated, build confidence. "
                "You pause, adjust, and keep going."
            ),
            "horizon_weeks": 4,
            "archetype": "",
            "core_need": "",
            "affirmation": "",
            "quote": "",
            "signature_metaphor": "",
            "signature_sentence": "",
            "top_theme_boosters": [],
            "pitfalls": [],
            "tensions": [],
            "blindspot": "",
            "from_words": {},
            "micro_pledge": "",
        })

    logo_path = get_logo_png_path()
    pdf_bytes = make_pdf_bytes(
        st.session_state.get("first_name", ""),
        st.session_state.get("email", ""),
        scores, top3, sections, free_responses, logo_path
    )
    st.success("Your personalized report is ready!")
    st.download_button(
        "📥 Download Your PDF Report",
        data=to_bytes(pdf_bytes),
        file_name="LifeMinusWork_Reflection_Report.pdf",
        mime="application/pdf",
    )

    # Optional CSV (ephemeral)
    try:
        import csv
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        csv_path = "/tmp/responses.csv"
        exists = Path(csv_path).exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not exists:
                w.writerow(["timestamp", "first_name", "email", "scores", "top3"])
            w.writerow([
                ts,
                st.session_state.get("first_name", ""),
                st.session_state.get("email", ""),
                json.dumps(scores),
                json.dumps(top3),
            ])
        st.caption("Saved to /tmp/responses.csv (Cloud-safe, ephemeral).")
    except Exception as e:
        st.caption(f"Could not save responses (demo only). {e}")

# Quick AI test
with st.expander("AI status (debug)", expanded=False):
    st.write("AI enabled:", USE_AI)
    st.write("Model:", HIGH_MODEL)
    st.write("Max tokens:", MAX_TOK_HIGH, "(fallback", FALLBACK_CAP, ")")
    if USE_AI and st.button("Test OpenAI now"):
        try:
            raw, usage, path = _call_openai_json(
                HIGH_MODEL, "Return strict JSON only.", 'Return {"ok": true} as JSON.', cap=128
            )
            msg = f"OK — via {path}. Output: {raw}"
            if usage:
                msg += f" | usage: in={usage.get('input')} out={usage.get('output')} total={usage.get('total')}"
            st.success(msg)
        except Exception as e:
            st.error(f"OpenAI error: {e}")
