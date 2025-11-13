import os
import time
import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ───────────────────────────────────────
# Page configuration
# ───────────────────────────────────────
st.set_page_config(
    page_title="Churn AI — Customer Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
)

# ───────────────────────────────────────
# Global CSS
# ───────────────────────────────────────
st.markdown(
    """
<style>
/* pull content up a bit */
header { visibility: hidden; }

html, body {
  height: 100%;
  overflow: hidden;              /* hide scrollbar */
}

::-webkit-scrollbar {
  display: none;
}

.stApp {
  height: 100vh;
  background:
    radial-gradient(1200px 600px at -10% -20%, rgba(56, 189, 248, .26) 0%, transparent 60%),
    radial-gradient(900px 500px  at 120% 120%, rgba(56, 189, 248, .20) 0%, transparent 65%),
    radial-gradient(700px 350px  at 20% 100%,  rgba(34, 197, 94, .20) 0%, transparent 60%),
    linear-gradient(180deg, #041320 0%, #041826 35%, #020617 100%);
  background-attachment: fixed;
}

/* subtle animated overlay */
.stApp::before {
  content:"";
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml;utf8,\
  <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 320'>\
    <path fill='%2315B8FF' fill-opacity='0.22' d='M0,160L60,181.3C120,203,240,245,360,256C480,267,600,245,720,224C840,203,960,181,1080,181.3C1200,181,1320,203,1380,213.3L1440,224L1440,0L1380,0C1320,0,1200,0,1080,0C960,0,840,0,720,0C600,0,480,0,360,0C240,0,120,0,60,0L0,0Z'/>\
  </svg>");
  background-size: cover;
  opacity: .9;
  animation: waveFloat 26s ease-in-out infinite alternate;
  pointer-events: none;
}
@keyframes waveFloat {
  from { transform: translateY(-6px); }
  to   { transform: translateY(8px); }
}

/* main container – almost pinned to top */
.main .block-container {
  max-width: 1080px;
  padding-top: 0.2rem;
  padding-bottom: 1.2rem;
}

/* hero banner with logo */
.hero-container {
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 16px 24px;
    border-radius: 22px;
    background: linear-gradient(90deg, #0ea5e9, #6366f1, #9333ea);
    box-shadow: 0 10px 40px rgba(0,0,0,0.25);
    margin-top: 0.05rem;
    margin-bottom: 0.15rem;     /* no big gap */
}
.hero-title {
    font-size: 34px;
    font-weight: 900;
    margin: 0;
    color: white;
}
.hero-sub {
    font-size: 13px;
    color: #e2e8f0;
    margin-top: 4px;
}
.hero-logo {
    height: 50px;
    width: 50px;
    border-radius: 12px;
    object-fit: contain;
    background: rgba(15,23,42,0.45);
}

/* fonts & inputs */
html, body, [class*="css"] {
  font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
label, .stTextInput label, .stSelectbox label {
  color: #E5F0FF !important;
  font-weight: 600;
}
input, select, textarea {
  background: rgba(15, 23, 42, 0.9) !important;
  color: #F9FAFF !important;
  border-radius: 10px !important;
  border: 1px solid rgba(148, 163, 184, 0.7) !important;
}
input::placeholder {
  color: #9CA3AF !important;
}

/* section title directly under hero */
.section-title {
  color: #E5F0FF;
  font-weight: 800;
  font-size: 18px;
  margin-top: 0.3rem;
  margin-bottom: 0.5rem;
}

/* buttons */
div.stButton > button:first-child {
  background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
  color: #fff;
  border: 0;
  padding: 0.7rem 1.3rem;
  border-radius: 12px;
  font-weight: 800;
}
div.stButton > button.k-secondary {
  background: transparent;
  border-radius: 10px;
  border: 1px solid #6b7280;
}

/* result wrapper + central glass tile */
.result-wrapper {
  margin-top: 1.6rem;
  display: flex;
  justify-content: center;
  align-items: flex-start;
}

.result-panel {
  width: 100%;
  max-width: 520px;
  margin: 0 auto;
  padding: 24px 30px 22px 30px;
  border-radius: 30px;
  backdrop-filter: blur(18px);
  border: 1px solid rgba(148, 163, 184, 0.9);
  box-shadow:
    0 0 0 1px rgba(15,23,42,0.7),
    0 32px 90px rgba(0,0,0,0.85);
  background:
    radial-gradient(circle at top left, rgba(74,222,128,0.35), transparent 60%),
    radial-gradient(circle at top right, rgba(248,113,113,0.35), transparent 60%),
    rgba(15,23,42,0.96);
  color: #E5F0FF;
  text-align: center;
}
.result-panel.low {
  box-shadow:
    0 0 0 1px rgba(74,222,128,0.9),
    0 32px 90px rgba(22,163,74,0.95);
}
.result-panel.high {
  box-shadow:
    0 0 0 1px rgba(248,113,113,0.95),
    0 32px 100px rgba(239,68,68,0.98);
}

.result-header {
  font-weight: 900;
  font-size: 20px;
  margin-bottom: 8px;
}

/* circular gauge */
.gauge-shell {
  display: flex;
  justify-content: center;
  margin-bottom: 10px;
}
.gauge {
  width: 200px;
  height: 200px;
  border-radius: 999px;
  padding: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.gauge-inner {
  width: 100%;
  height: 100%;
  border-radius: 999px;
  background: radial-gradient(circle at 30% 10%, rgba(255,255,255,0.18), rgba(15,23,42,0.98));
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
}
.gauge-value {
  font-size: 32px;
  font-weight: 900;
}
.gauge-label {
  font-size: 13px;
  opacity: .95;
}

/* copy inside panel */
.result-title {
  font-weight: 800;
  font-size: 17px;
  margin-top: 4px;
  margin-bottom: 3px;
}
.result-sub {
  font-size: 13px;
  opacity: .95;
}
.result-sub-2 {
  font-size: 12px;
  opacity: .80;
  margin-top: 2px;
}

/* metrics row */
.metrics-row {
  margin-top: 16px;
  display: flex;
  gap: 24px;
  justify-content: center;
}
.metrics-block {
  text-align: center;
}
.metric-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #cbd5f5;
}
.metric-value {
  font-size: 17px;
  font-weight: 800;
}

/* buttons under panel */
.button-row {
  margin-top: 0.9rem;
  display: flex;
  justify-content: center;
  gap: 0.8rem;
}
.button-row > div {
  flex: 0 0 auto;
}

/* hide footer */
footer { visibility: hidden; height: 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────────────────
# Load model
# ───────────────────────────────────────
MODEL_PATH = Path("models/churn_xgb.joblib")
if not MODEL_PATH.exists():
    st.error("Model not found. Run: `python src/churn/train.py`", icon="⚠️")
    st.stop()

bundle = joblib.load(MODEL_PATH)
pipe = bundle["pipeline"]
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", "0.50"))

# ───────────────────────────────────────
# Session state
# ───────────────────────────────────────
if "mode" not in st.session_state:
    st.session_state.mode = "form"
if "threshold" not in st.session_state:
    st.session_state.threshold = DEFAULT_THRESHOLD
if "result" not in st.session_state:
    st.session_state.result = None

# ───────────────────────────────────────
# Helpers
# ───────────────────────────────────────
def to_num(txt):
    s = str(txt).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_logo_b64() -> str | None:
    logo_path = Path("app/assets/logo.png")
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


# ───────────────────────────────────────
# HERO
# ───────────────────────────────────────
logo_b64 = load_logo_b64()
if logo_b64:
    logo_tag = f"<img src='data:image/png;base64,{logo_b64}' class='hero-logo' />"
else:
    logo_tag = "<div class='hero-logo'></div>"

st.markdown(
    f"""
<div class='hero-container'>
   {logo_tag}
   <div>
      <div class='hero-title'>Churn AI</div>
      <div class='hero-sub'>Customer Risk Intelligence — one-screen intake with a focused results dashboard.</div>
   </div>
</div>
""",
    unsafe_allow_html=True,
)

# ───────────────────────────────────────
# FORM VIEW
# ───────────────────────────────────────
if st.session_state.mode == "form":
    st.markdown('<div class="section-title">Enter customer details</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age = st.text_input("Age", placeholder="e.g., 32")
        tenure = st.text_input("Tenure (months)", placeholder="e.g., 18")
        usage = st.text_input("Usage (score)", placeholder="e.g., 45")
        support = st.text_input("Support Calls", placeholder="e.g., 2")
        payment = st.text_input("Payment Delay (days)", placeholder="e.g., 0")
    with c2:
        gender = st.selectbox("Gender", ["— Select —", "Male", "Female", "Other"], index=0)
        subscription = st.selectbox("Subscription", ["— Select —", "Basic", "Standard", "Premium"], index=0)
        contract = st.selectbox("Contract", ["— Select —", "Monthly", "Quarterly", "Annual"], index=0)
        spend = st.text_input("Total Spend", placeholder="e.g., 900")
        last = st.text_input("Last Interaction (days)", placeholder="e.g., 7")

    tcol, bcol = st.columns([3, 1])
    with tcol:
        st.session_state.threshold = st.slider(
            "Decision Threshold",
            0.05,
            0.95,
            float(st.session_state.threshold),
            0.01,
        )
    with bcol:
        st.write("")
        predict = st.button("Predict", type="primary")

    if predict:
        row = {
            "Age": to_num(age),
            "Gender": gender,
            "Tenure": to_num(tenure),
            "Usage": to_num(usage),
            "Support": to_num(support),
            "PaymentDelay": to_num(payment),
            "Subscription": subscription,
            "Contract": contract,
            "TotalSpend": to_num(spend),
            "LastInteraction": to_num(last),
        }
        missing = [k for k, v in row.items() if v in (None, "", "— Select —")]
        if missing:
            st.warning("Please fill all fields: " + ", ".join(missing), icon="⚠️")
        else:
            df = pd.DataFrame([row])
            prob = float(pipe.predict_proba(df)[:, 1][0])
            thr = float(st.session_state.threshold)
            is_risk = prob >= thr
            st.session_state.result = {"prob": prob, "thr": thr, "is_risk": is_risk}
            st.session_state.mode = "result"
            st.rerun()

# ───────────────────────────────────────
# RESULT VIEW – clear centered tile
# ───────────────────────────────────────
else:
    res = st.session_state.result or {
        "prob": 0.0,
        "thr": float(st.session_state.threshold),
        "is_risk": False,
    }
    prob = res["prob"]
    thr = res["thr"]
    is_risk = res["is_risk"]

    # emoji burst in centre (disappears)
    emoji_ph = st.empty()
    if is_risk:
        emoji_ph.markdown("<h1 style='text-align:center;'>🚨😟🔥</h1>", unsafe_allow_html=True)
    else:
        emoji_ph.markdown("<h1 style='text-align:center;'>🎉✅🕺</h1>", unsafe_allow_html=True)
    time.sleep(1.1)
    emoji_ph.empty()

    card_class = "high" if is_risk else "low"
    primary = "#ef4444" if is_risk else "#22c55e"
    angle = max(10, min(180, int(prob * 1.8)))
    gauge_style = (
        f"background: conic-gradient({primary} {angle}deg, rgba(15,23,42,0.85) {angle}deg);"
    )

    if is_risk:
        title = "At Risk! High Churn Probability 🔥"
        sub = "Immediate retention efforts recommended."
        desc = "Contact this customer soon and consider targeted incentives to reduce the chance of churn."
    else:
        title = "Fantastic! Low Churn Probability 🎉"
        sub = "This customer is highly likely to stay."
        desc = "Great time to offer loyalty perks or upsell while satisfaction is high."

    panel_html = f"""
    <div class="result-wrapper">
      <div class="result-panel {card_class}">
        <div class="result-header">Prediction Result</div>
        <div class="gauge-shell">
          <div class="gauge" style="{gauge_style}">
            <div class="gauge-inner">
              <div class="gauge-value">{prob*100:.1f}%</div>
              <div class="gauge-label">Churn Risk</div>
            </div>
          </div>
        </div>
        <div class="result-title">{title}</div>
        <div class="result-sub">{sub}</div>
        <div class="result-sub-2">{desc}</div>
        <div class="metrics-row">
          <div class="metrics-block">
            <div class="metric-label">Decision Threshold</div>
            <div class="metric-value">{thr:.2f}</div>
          </div>
          <div class="metrics-block">
            <div class="metric-label">Risk Class</div>
            <div class="metric-value">{'High' if is_risk else 'Low'}</div>
          </div>
          <div class="metrics-block">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{(prob if is_risk else 1-prob):.1%}</div>
          </div>
        </div>
      </div>
    </div>
    """

    st.markdown(panel_html, unsafe_allow_html=True)

    # buttons under tile
    st.markdown('<div class="button-row">', unsafe_allow_html=True)
    col_again, col_close = st.columns([1, 1])
    with col_again:
        if st.button("Predict again", key="predict_again_btn"):
            st.session_state.mode = "form"
            st.rerun()
    with col_close:
        st.button("Close", key="close_btn")
    st.markdown("</div>", unsafe_allow_html=True)
