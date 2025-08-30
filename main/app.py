# app.py
# Life Minus Work (L-W) — Online Questionnaire (Future Snapshot fixed to ~4 weeks)
# Run:  python app.py
# Then open: http://127.0.0.1:5000

from flask import Flask, request, redirect, url_for, render_template_string, session, send_file, abort
from datetime import datetime
import io
import os

# Optional PDF generation (requires: pip install reportlab)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get("LW_SECRET_KEY", "dev-secret-change-me")

# -----------------------------
# Config / Constants
# -----------------------------
APP_BRAND = "Life Minus Work"
APP_TAGLINE = "Reflection Report"
FUTURE_HORIZON_WEEKS = 4  # <-- Fixed horizon: ~4 weeks (1 month). No slider in UI.

# Simple Likert questions (1–5). Adjust/expand as needed.
QUESTIONS = [
    {
        "id": "purpose",
        "label": "Purpose & Identity: I have a clear sense of purpose beyond work.",
        "category": "Purpose & Identity",
    },
    {
        "id": "social",
        "label": "Social Health: I feel connected to a supportive community.",
        "category": "Social Health & Community",
    },
    {
        "id": "vitality",
        "label": "Health & Vitality: I regularly engage in activities that support my physical and mental health.",
        "category": "Health & Vitality",
    },
    {
        "id": "learning",
        "label": "Learning & Growth: I am actively learning new skills or exploring new interests.",
        "category": "Learning & Growth",
    },
    {
        "id": "adventure",
        "label": "Adventure & Exploration: I plan or enjoy small adventures or new experiences.",
        "category": "Adventure & Exploration",
    },
    {
        "id": "giving",
        "label": "Giving Back: I contribute time, skills, or resources to help others.",
        "category": "Giving Back",
    },
]

LIKERT_CHOICES = ["1 - Strongly Disagree", "2 - Disagree", "3 - Neutral", "4 - Agree", "5 - Strongly Agree"]

# -----------------------------
# HTML Templates (inline for single-file app)
# -----------------------------

BASE_CSS = """
<style>
  :root {
    --brand:#0e7490; /* teal-700 */
    --brand-2:#14b8a6; /* teal-500 */
    --ink:#0f172a; /* slate-900 */
    --muted:#475569; /* slate-600 */
    --bg:#f8fafc; /* slate-50 */
    --card:#ffffff;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,"Noto Sans",sans-serif;line-height:1.4}
  header{padding:28px 20px;background:linear-gradient(135deg,var(--brand),var(--brand-2));color:white}
  .wrap{max-width:920px;margin:0 auto}
  h1,h2,h3{margin:0 0 10px}
  .card{background:var(--card);border-radius:16px;box-shadow:0 6px 24px rgba(2,6,23,.06);padding:24px;margin:20px 0}
  label{display:block;margin:.5rem 0 .25rem;font-weight:600}
  .muted{color:var(--muted);font-size:.95rem}
  .grid{display:grid;gap:16px}
  .g2{grid-template-columns:repeat(2,minmax(0,1fr))}
  select,textarea,input[type="text"],input[type="email"]{
    width:100%;border:1px solid #e2e8f0;border-radius:12px;padding:12px;background:white
  }
  textarea{min-height:120px;resize:vertical}
  .btn{
    display:inline-block;background:var(--brand);color:white;border:none;border-radius:999px;
    padding:12px 18px;font-weight:700;cursor:pointer
  }
  .btn.secondary{background:#0f172a}
  .kpis{display:flex;gap:14px;flex-wrap:wrap}
  .kpi{flex:1 1 140px;background:#ecfeff;border:1px solid #cffafe;border-radius:12px;padding:14px}
  .kpi .big{font-size:1.6rem;font-weight:800}
  .footer-note{font-size:.9rem;color:var(--muted);margin-top:4px}
  .pill{display:inline-block;padding:6px 10px;border-radius:999px;background:#e2e8f0;color:#0f172a;font-weight:700;font-size:.8rem;vertical-align:middle}
  .hr{height:1px;background:#e2e8f0;margin:18px 0}
  .brandfoot{display:flex;justify-content:space-between;align-items:center;font-size:.9rem;color:var(--muted)}
  .sr{position:absolute;left:-9999px}
</style>
"""

TEMPLATE_INDEX = f"""
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{APP_BRAND} — Questionnaire</title>
{BASE_CSS}
<body>
  <header>
    <div class="wrap">
      <h1>{APP_BRAND} — Online Questionnaire</h1>
      <p class="muted">A quick check-in across six areas, plus a short postcard from your future self.</p>
    </div>
  </header>

  <main class="wrap">
    <form class="card" method="post" action="{{{{ url_for('submit') }}}}">
      <h2>Readiness Check</h2>
      <p class="muted">Rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree).</p>

      <div class="grid">
        {% for q in questions %}
          <div>
            <label for="{{ q.id }}">{{ q.label }}</label>
            <select required id="{{ q.id }}" name="{{ q.id }}">
              <option value="">Select…</option>
              {% for choice in likert %}
                {% set val = loop.index %}  <!-- 1..5 -->
                <option value="{{ val }}">{{ choice }}</option>
              {% endfor %}
            </select>
          </div>
        {% endfor %}
      </div>

      <div class="hr"></div>

      <h2>Future Snapshot <span class="pill">~4 weeks</span></h2>
      <p class="muted">
        Imagine it’s about <strong>one month from now</strong>. Write yourself a short postcard from the future about how your
        Life Minus Work journey is unfolding. What are you proud of? What small wins have happened?
      </p>
      <label for="future_postcard">Your postcard (a few sentences is perfect)</label>
      <textarea id="future_postcard" name="future_postcard" placeholder="Dear me, it's been about a month and…"></textarea>

      <div class="hr"></div>

      <h3>Optional</h3>
      <div class="grid g2">
        <div>
          <label for="name">Your name (optional)</label>
          <input type="text" id="name" name="name" placeholder="Sam">
        </div>
        <div>
          <label for="email">Email (optional, for sending the report)</label>
          <input type="email" id="email" name="email" placeholder="you@example.com">
          <p class="footer-note">PDF download works without email. Email delivery can be added later.</p>
        </div>
      </div>

      <!-- Hidden fixed horizon (no slider shown) -->
      <input type="hidden" name="future_horizon_weeks" value="{FUTURE_HORIZON_WEEKS}">

      <div style="margin-top:18px">
        <button class="btn" type="submit">Generate Reflection Report</button>
      </div>
    </form>

    <div class="brandfoot">
      <div>© {{ year }} {APP_BRAND}</div>
      <div class="muted">{APP_TAGLINE}</div>
    </div>
  </main>
</body>
</html>
"""

TEMPLATE_REPORT = f"""
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{APP_BRAND} — {APP_TAGLINE}</title>
{BASE_CSS}
<body>
  <header>
    <div class="wrap">
      <h1>{APP_BRAND}</h1>
      <p class="muted">{APP_TAGLINE}</p>
    </div>
  </header>

  <main class="wrap">
    <div class="card">
      <h2>Overview</h2>
      <div class="kpis">
        <div class="kpi">
          <div class="big">{{ score }}/30</div>
          <div class="muted">Readiness Score</div>
        </div>
        <div class="kpi">
          <div class="big">{{ average }}/5</div>
          <div class="muted">Average per item</div>
        </div>
        <div class="kpi">
          <div class="big">~{{ horizon }} weeks</div>
          <div class="muted">Future Snapshot Horizon</div>
        </div>
        <div class="kpi">
          <div class="big">{{ date }}</div>
          <div class="muted">Report Date</div>
        </div>
      </div>

      {% if name %}
        <p class="muted" style="margin-top:8px">Prepared for <strong>{{ name }}</strong></p>
      {% endif %}
    </div>

    <div class="card">
      <h2>Category Breakdown</h2>
      <div class="grid g2">
        {% for row in breakdown %}
          <div>
            <div style="font-weight:700">{{ row.category }}</div>
            <div class="muted">{{ row.label }}</div>
            <div style="margin-top:6px"><span class="pill">{{ row.value }}/5</span></div>
          </div>
        {% endfor %}
      </div>
    </div>

    <div class="card">
      <h2>Future Snapshot <span class="pill">Postcard from ~{{ horizon }} weeks ahead</span></h2>
      {% if postcard %}
        <blockquote style="border-left:4px solid #e2e8f0;margin:0;padding:8px 14px;background:#f8fafc;border-radius:12px">
          {{ postcard | e }}
        </blockquote>
      {% else %}
        <p class="muted">No postcard text provided.</p>
      {% endif %}
      <p class="footer-note" style="margin-top:10px">
        Prompt used: “Imagine yourself about one month from now. Write a short postcard from your future self.”
      </p>
    </div>

    <div class="card">
      <h2>Next Tiny Step</h2>
      <p class="muted">Pick one action you can take within the next 7 days that nudges your average up by just 0.5 in any category.</p>
      <ul>
        <li>Purpose & Identity — write a 3-sentence purpose statement draft.</li>
        <li>Social Health — schedule a coffee or a walk with a friend.</li>
        <li>Health & Vitality — 20-minute walk × 3 this week.</li>
        <li>Learning & Growth — queue a 30-minute tutorial or class.</li>
        <li>Adventure & Exploration — plan one micro-adventure (new café, park, or route).</li>
        <li>Giving Back — offer help or mentorship for 30 minutes.</li>
      </ul>
    </div>

    <div class="card" style="display:flex;gap:12px;align-items:center;justify-content:space-between;flex-wrap:wrap">
      <div>
        <div class="muted">Save or share your report</div>
      </div>
      <div>
        <a class="btn" href="{{ url_for('download_pdf') }}">Download PDF</a>
        <a class="btn secondary" href="{{ url_for('index') }}">Start Over</a>
      </div>
    </div>

    <div class="brandfoot">
      <div>© {{ year }} {APP_BRAND}</div>
      <div class="muted">{APP_TAGLINE}</div>
    </div>
  </main>
</body>
</html>
"""

TEMPLATE_NO_PDF = f"""
<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{APP_BRAND} — PDF Unavailable</title>
{BASE_CSS}
<body>
  <header>
    <div class="wrap">
      <h1>{APP_BRAND}</h1>
      <p class="muted">{APP_TAGLINE}</p>
    </div>
  </header>

  <main class="wrap">
    <div class="card">
      <h2>PDF Generation Unavailable</h2>
      <p class="muted">This environment doesn’t have the <code>reportlab</code> package installed.</p>
      <p>You can still save your report as a PDF using your browser’s “Print → Save as PDF”.</p>
      <div style="margin-top:12px">
        <a class="btn" href="{{ url_for('report') }}">Back to Report</a>
      </div>
    </div>
  </main>
</body>
</html>
"""

# -----------------------------
# Helpers
# -----------------------------

def compute_score(form_data):
    """Compute totals and per-category breakdown from submitted form data."""
    breakdown = []
    total = 0
    for q in QUESTIONS:
        raw = form_data.get(q["id"], "").strip()
        try:
            val = int(raw)
        except Exception:
            val = 0
        total += val
        breakdown.append({
            "id": q["id"],
            "label": q["label"],
            "category": q["category"],
            "value": val
        })
    average = round(total / len(QUESTIONS), 2)
    return total, average, breakdown

# -----------------------------
# Routes
# -----------------------------

@app.route("/", methods=["GET"])
def index():
    session.pop("report", None)
    return render_template_string(
        TEMPLATE_INDEX,
        questions=QUESTIONS,
        likert=LIKERT_CHOICES,
        year=datetime.now().year
    )

@app.route("/submit", methods=["POST"])
def submit():
    total, average, breakdown = compute_score(request.form)
    name = request.form.get("name", "").strip()
    email = request.form.get("email", "").strip()
    postcard = request.form.get("future_postcard", "").strip()
    # Horizon comes from fixed hidden field; we trust our own constant but keep the form value for completeness.
    try:
        horizon_weeks = int(request.form.get("future_horizon_weeks", FUTURE_HORIZON_WEEKS))
    except Exception:
        horizon_weeks = FUTURE_HORIZON_WEEKS

    session["report"] = {
        "score": total,
        "average": average,
        "breakdown": breakdown,
        "name": name,
        "email": email,
        "postcard": postcard,
        "horizon": FUTURE_HORIZON_WEEKS,  # enforce fixed value
        "date": datetime.now().strftime("%Y-%m-%d"),
    }
    return redirect(url_for("report"))

@app.route("/report", methods=["GET"])
def report():
    data = session.get("report")
    if not data:
        return redirect(url_for("index"))
    return render_template_string(
        TEMPLATE_REPORT,
        score=data["score"],
        average=data["average"],
        breakdown=data["breakdown"],
        name=data["name"],
        postcard=data["postcard"],
        horizon=data["horizon"],
        date=data["date"],
        year=datetime.now().year
    )

@app.route("/download.pdf", methods=["GET"])
def download_pdf():
    data = session.get("report")
    if not data:
        return redirect(url_for("index"))

    if not REPORTLAB_AVAILABLE:
        # Graceful fallback if reportlab isn't installed
        return render_template_string(TEMPLATE_NO_PDF)

    # Build a very simple branded PDF
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    margin = 2 * cm
    x = margin
    y = height - margin

    def draw_line():
        c.setStrokeColorRGB(0.08, 0.46, 0.56)  # teal-ish
        c.setLineWidth(2)
        c.line(margin, y, width - margin, y)

    # Header
    c.setFont("Helvetica-Bold", 20)
    c.setFillColorRGB(0.06, 0.45, 0.56)
    c.drawString(x, y, f"{APP_BRAND} — {APP_TAGLINE}")
    y -= 14
    draw_line()
    y -= 20

    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0, 0, 0)
    c.drawString(x, y, f"Report Date: {data['date']}")
    if data["name"]:
        c.drawRightString(width - margin, y, f"Prepared for: {data['name']}")
    y -= 22

    # Overview
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "Overview")
    y -= 16
    c.setFont("Helvetica", 11)
    c.drawString(x, y, f"Readiness Score: {data['score']}/30")
    y -= 14
    c.drawString(x, y, f"Average per item: {data['average']}/5")
    y -= 14
    c.drawString(x, y, f"Future Snapshot Horizon: ~{data['horizon']} weeks")
    y -= 22

    # Breakdown
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, "Category Breakdown")
    y -= 16
    c.setFont("Helvetica", 11)
    for row in data["breakdown"]:
        line = f"{row['category']}: {row['value']}/5 — {row['label']}"
        # Wrap manually if needed
        y = draw_wrapped(c, line, x, y, width - margin, leading=14)
        y -= 4
        if y < margin + 120:
            c.showPage()
            y = height - margin
            c.setFont("Helvetica", 11)

    # Future Snapshot
    if y < margin + 160:
        c.showPage()
        y = height - margin

    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, f"Future Snapshot — Postcard from ~{data['horizon']} weeks ahead")
    y -= 16
    c.setFont("Helvetica-Oblique", 11)
    postcard = data.get("postcard", "").strip() or "No postcard provided."
    y = draw_wrapped(c, f"“{postcard}”", x, y, width - margin, leading=14)

    # Footer
    c.setFont("Helvetica", 9)
    c.drawRightString(width - margin, margin - 4, f"© {datetime.now().year} {APP_BRAND}")

    c.showPage()
    c.save()
    buffer.seek(0)
    filename = f"LW_Reflection_Report_{data['date']}.pdf"
    return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name=filename)


def draw_wrapped(c, text, x, y, max_x, leading=14):
    """Very simple word-wrap for reportlab canvas.drawString."""
    max_width = max_x - x
    words = text.split()
    line = ""
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, "Helvetica", 11) <= max_width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= leading
            line = w
            if y < 2.5 * cm:  # new page if near bottom
                c.showPage()
                y = A4[1] - 2 * cm
                c.setFont("Helvetica", 11)
    if line:
        c.drawString(x, y, line)
        y -= leading
    return y


if __name__ == "__main__":
    # Tip: set LW_SECRET_KEY in your env for production sessions
    app.run(debug=True)
