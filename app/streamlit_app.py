from __future__ import annotations

import io
import os
import textwrap
from datetime import datetime
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.backends.backend_pdf import PdfPages

st.set_page_config(page_title="Churn Risk Intelligence", layout="wide")

# Visual direction: medium-tone pastel palette (readable, not too light/dark)
PALETTE = {
    "teal": "#59B8B4",
    "sky": "#63A6E8",
    "purple": "#8D79D9",
    "coral": "#E88B87",
    "mint": "#7BCFA3",
    "amber": "#DFA75A",
    "slate": "#5B6D7A",
    "bg_soft": "#F6F8FC",
    "text": "#1F2937",
}
PASTELTE = "#8ECBC4"
PASTELPURP = "#A896DD"
CHART_FIGSIZE = (6.2, 4.2)
SMALL_FIGSIZE = (4.4, 2.7)
FEATURE_FIGSIZE = (4.8, 2.8)
FEATURE_HEAT_FIGSIZE = (5.0, 3.2)
TITLE_SIZE = 13
LABEL_SIZE = 11


def style_axis(
    ax,
    title: str,
    xlabel: str | None = None,
    ylabel: str | None = None,
    pastel_bg: bool = False,
) -> None:
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE, fontweight="bold")
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE, fontweight="bold")
    if pastel_bg:
        ax.set_facecolor("#DDEBFF")
        ax.figure.patch.set_facecolor("#D4E4FB")
        for spine in ax.spines.values():
            spine.set_color("#A8BFE6")
            spine.set_linewidth(0.9)
    ax.grid(axis="y", alpha=0.18)
    ax.tick_params(labelsize=10)


def annotate_bars(ax, percent: bool = False, currency: bool = False) -> None:
    for p in ax.patches:
        h = p.get_height()
        if np.isnan(h):
            continue
        if percent:
            txt = f"{h:.1%}"
        elif currency:
            txt = f"EUR {h:,.0f}"
        else:
            txt = f"{h:.2f}"
        ax.annotate(
            txt,
            (p.get_x() + p.get_width() / 2, h),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            xytext=(0, 4),
            textcoords="offset points",
        )


@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifact_dir = os.path.join(base_dir, "artifacts")
    model = joblib.load(os.path.join(artifact_dir, "churn_model.pkl"))
    scaler = joblib.load(os.path.join(artifact_dir, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(artifact_dir, "feature_names.pkl"))
    return model, scaler, feature_names


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
          :root {{
            --primary-color: #2563EB;
          }}
          .block-container {{
            padding-top: 1.0rem;
            max-width: 1280px;
            background: linear-gradient(180deg, #DDE8FA 0%, #D2E0F8 100%);
            border-radius: 18px;
          }}
          [data-testid="stSidebar"] > div:first-child {{
            background: linear-gradient(180deg, #DCE6FA 0%, #CFDBF5 100%);
          }}
          html, body, [class*="css"]  {{
            font-family: "Segoe UI", "Avenir Next", "Helvetica Neue", Arial, sans-serif;
            color: {PALETTE['text']};
          }}
          h1, h2, h3, h4 {{ letter-spacing: -0.02em; font-weight: 800 !important; color: #133265; }}
          h3 {{
            font-size: 1.18rem !important;
            margin-top: 0.35rem !important;
            margin-bottom: 0.30rem !important;
          }}
          p, .stCaption, [data-testid="stCaptionContainer"] {{
            font-size: 0.96rem !important;
            font-weight: 600 !important;
            color: #34507A !important;
          }}
          [data-testid="stMetricValue"] {{
            font-weight: 800 !important;
            color: #0f2e66;
          }}
          [data-testid="stMetricLabel"] {{
            font-weight: 800 !important;
            font-size: 1rem !important;
            color: #17386f !important;
          }}

          .hero {{
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(31, 41, 55, 0.10);
            border-radius: 22px;
            padding: 16px 18px 14px 18px;
            background:
              radial-gradient(900px 320px at 0% 0%, rgba(99,166,232,0.25), transparent 62%),
              radial-gradient(800px 300px at 100% 0%, rgba(141,121,217,0.22), transparent 62%),
              linear-gradient(180deg, #FFFFFF 0%, #F5F8FF 100%);
          }}
          .hero-title {{
            margin: 0;
            font-size: 40px;
            line-height: 1.05;
            color: #1E3A8A; /* fallback so heading always appears */
            background: linear-gradient(90deg, #0F172A 0%, #2563EB 45%, #7C3AED 100%);
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 1px 0 rgba(255, 255, 255, 0.3);
          }}
          .hero-sub {{
            margin-top: 6px;
            opacity: 0.86;
            font-size: 14px;
          }}
          .chip {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 999px;
            border: 1px solid rgba(31,41,55,0.14);
            background: rgba(255,255,255,0.8);
            font-size: 12px;
            margin-right: 7px;
          }}
          .card {{
            border: 1px solid rgba(31,41,55,0.11);
            border-radius: 16px;
            padding: 12px 14px 10px 14px;
            background: linear-gradient(180deg, #EAF2FF, #E1ECFF);
            box-shadow: 0 10px 25px rgba(15,23,42,0.05);
          }}

          /* Animated euro notes next to title */
          @keyframes noteFall {{
            0% {{ transform: translateY(-90px) translateX(0px) rotate(-8deg); opacity: 0; }}
            12% {{ opacity: 0.60; }}
            60% {{ opacity: 0.55; }}
            100% {{ transform: translateY(270px) translateX(18px) rotate(8deg); opacity: 0; }}
          }}
          .note {{
            position: absolute;
            width: 78px;
            height: 46px;
            border-radius: 10px;
            border: 1px solid rgba(31,41,55,0.15);
            background: linear-gradient(135deg, rgba(123,207,163,0.65), rgba(99,166,232,0.55));
            box-shadow: 0 14px 24px rgba(15,23,42,0.10);
          }}
          .note:after {{
            content: "EUR";
            position: absolute;
            inset: 10px;
            border: 1px dashed rgba(31,41,55,0.20);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
            color: #1f2937;
            font-weight: 700;
            letter-spacing: 0.08em;
          }}
          .n1 {{ right: 20px; top: -20px; animation: noteFall 7.1s ease-in-out infinite; }}
          .n2 {{ right: 98px; top: -40px; animation: noteFall 8.0s ease-in-out infinite; animation-delay: .9s; background: linear-gradient(135deg, rgba(141,121,217,0.58), rgba(89,184,180,0.50)); }}
          .n3 {{ right: 58px; top: -30px; animation: noteFall 7.6s ease-in-out infinite; animation-delay: 1.6s; background: linear-gradient(135deg, rgba(232,139,135,0.58), rgba(223,167,90,0.50)); }}

          /* Blue control accent to match hero theme */
          input[type="checkbox"], input[type="radio"], input[type="range"] {{ accent-color: #2563EB; }}
          [data-testid="stSlider"] {{
            --primary-color: #2563EB !important;
          }}
          [data-baseweb="slider"] [role="slider"] {{
            border-color: #2563EB !important;
            background-color: #2563EB !important;
            box-shadow: 0 0 0 4px rgba(37,99,235,0.15) !important;
          }}
          [data-baseweb="slider"] [role="progressbar"] {{
            background: #2563EB !important;
          }}
          [data-baseweb="slider"] > div > div {{
            background-color: rgba(37,99,235,0.25) !important;
          }}
          [data-testid="stSidebar"] .stSlider [role="slider"] {{
            background-color: #2563EB !important;
            border-color: #2563EB !important;
          }}
          [data-baseweb="select"] [data-baseweb="tag"] {{
            background-color: #DBEAFE !important;
            border: 1px solid #93C5FD !important;
            color: #1D4ED8 !important;
            font-weight: 700 !important;
          }}
          [data-baseweb="select"] input,
          [data-baseweb="select"] div {{
            color: #1D4ED8 !important;
          }}
          [role="listbox"] [role="option"][aria-selected="true"] {{
            background-color: #DBEAFE !important;
            color: #1D4ED8 !important;
            font-weight: 700 !important;
          }}

          /* hyperlink-like tab highlight */
          [data-testid="stTabs"] button[role="tab"] {{
            border-radius: 10px !important;
            border: 1px solid rgba(37,99,235,0.18) !important;
            transition: all 0.22s ease !important;
            font-weight: 700 !important;
          }}
          [data-testid="stTabs"] button[role="tab"]:hover {{
            background: #EAF2FF !important;
            border-color: #93C5FD !important;
            box-shadow: 0 6px 14px rgba(37,99,235,0.14) !important;
            transform: translateY(-1px);
          }}
          [data-testid="stTabs"] button[aria-selected="true"] {{
            background: #DBEAFE !important;
            color: #1D4ED8 !important;
            border-color: #60A5FA !important;
          }}
          [data-testid="stButton"] > button {{
            background: linear-gradient(90deg, #CDEFCF, #BFE6C1, #D9F5DB) !important;
            color: #111827 !important;
            border: 1px solid #8FCF95 !important;
            font-weight: 800 !important;
            box-shadow: 0 0 0 rgba(143,207,149,0.55);
            animation: pulseGlow 1.9s ease-in-out infinite;
            transition: transform .18s ease;
          }}
          [data-testid="stButton"] > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(95,160,108,0.35);
          }}
          @keyframes pulseGlow {{
            0% {{ box-shadow: 0 0 0 0 rgba(143,207,149,0.55); }}
            70% {{ box-shadow: 0 0 0 10px rgba(143,207,149,0.0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(143,207,149,0.0); }}
          }}
          .sidebar-note {{
            background: linear-gradient(180deg, #E5EEFF 0%, #DAE7FF 100%);
            border: 1px solid #9EB8E8;
            border-radius: 12px;
            padding: 10px 12px;
            margin-bottom: 8px;
          }}
          .sidebar-note h4 {{
            margin: 0 0 4px 0;
            font-size: 1rem;
            color: #17386f;
          }}
          .sidebar-note p {{
            margin: 0;
            font-size: 0.9rem !important;
            color: #2A497A !important;
            font-weight: 600 !important;
          }}

          .ripple {{
            position: fixed;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(37,99,235,0.45) 0%, rgba(37,99,235,0.08) 60%, rgba(37,99,235,0) 75%);
            transform: translate(-50%, -50%) scale(0);
            animation: rippleAnim 520ms ease-out forwards;
            pointer-events: none;
            z-index: 9999;
          }}
          @keyframes rippleAnim {{
            to {{
              transform: translate(-50%, -50%) scale(8.8);
              opacity: 0;
            }}
          }}

          .rec-ticker-wrap {{
            border: 1px solid rgba(31,41,55,0.12);
            border-radius: 12px;
            background: linear-gradient(180deg, #FFFFFF, #F2F7FF);
            overflow: hidden;
            margin: 4px 0 10px 0;
            box-shadow: 0 8px 16px rgba(15,23,42,0.06);
          }}
          .rec-ticker {{
            width: 100%;
            overflow: hidden;
            white-space: nowrap;
          }}
          .rec-track {{
            display: inline-block;
            padding: 10px 0;
            animation: recScroll 28s linear infinite;
          }}
          .rec-pill {{
            display: inline-block;
            margin: 0 8px;
            padding: 6px 12px;
            border-radius: 999px;
            background: #E9F2FF;
            border: 1px solid #BFDBFE;
            color: #133265;
            font-size: 13px;
            font-weight: 700;
          }}
          @keyframes recScroll {{
            from {{ transform: translateX(0%); }}
            to {{ transform: translateX(-50%); }}
          }}
          .click-flash {{
            animation: clickFlash .28s ease;
          }}
          @keyframes clickFlash {{
            0% {{ filter: brightness(1.0); }}
            50% {{ filter: brightness(1.12); }}
            100% {{ filter: brightness(1.0); }}
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def bank_logo_svg() -> str:
    return """
    <svg width="62" height="62" viewBox="0 0 256 256" xmlns="http://www.w3.org/2000/svg" aria-label="bank-logo">
      <defs>
        <linearGradient id="gl" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#63A6E8"/>
          <stop offset="100%" stop-color="#8D79D9"/>
        </linearGradient>
      </defs>
      <rect x="18" y="20" width="220" height="214" rx="54" fill="url(#gl)" opacity="0.23"/>
      <path d="M48 102 L128 56 L208 102" fill="none" stroke="#0F172A" stroke-width="14" stroke-linecap="round" stroke-linejoin="round"/>
      <rect x="64" y="108" width="128" height="12" rx="6" fill="#0F172A"/>
      <rect x="72" y="126" width="20" height="62" rx="10" fill="#0F172A"/>
      <rect x="106" y="126" width="20" height="62" rx="10" fill="#0F172A"/>
      <rect x="140" y="126" width="20" height="62" rx="10" fill="#0F172A"/>
      <rect x="174" y="126" width="20" height="62" rx="10" fill="#0F172A"/>
      <rect x="56" y="196" width="144" height="12" rx="6" fill="#0F172A"/>
    </svg>
    """


def inject_click_animation() -> None:
    components.html(
        """
        <script>
        function bindRipple(docRef) {
          if (!docRef || docRef.__churnRippleBound) return;
          docRef.__churnRippleBound = true;
          docRef.addEventListener("click", function(e) {
            const ripple = docRef.createElement("span");
            ripple.style.position = "fixed";
            ripple.style.width = "14px";
            ripple.style.height = "14px";
            ripple.style.borderRadius = "9999px";
            ripple.style.left = e.clientX + "px";
            ripple.style.top = e.clientY + "px";
            ripple.style.transform = "translate(-50%, -50%) scale(0)";
            ripple.style.pointerEvents = "none";
            ripple.style.zIndex = "2147483647";
            ripple.style.background = "radial-gradient(circle, rgba(37,99,235,0.42) 0%, rgba(37,99,235,0.16) 58%, rgba(37,99,235,0) 75%)";
            ripple.style.transition = "transform 520ms ease-out, opacity 520ms ease-out";
            docRef.body.appendChild(ripple);
            requestAnimationFrame(() => {
              ripple.style.transform = "translate(-50%, -50%) scale(8.6)";
              ripple.style.opacity = "0";
            });
            setTimeout(() => ripple.remove(), 560);

            const appRoot = docRef.querySelector(".stApp");
            if (appRoot) {
              appRoot.classList.add("click-flash");
              setTimeout(() => appRoot.classList.remove("click-flash"), 300);
            }
          }, { passive: true });
        }
        try { bindRipple(window.parent.document); } catch (e) {}
        try { bindRipple(document); } catch (e) {}
        try {
          const appRoot = (window.parent.document || document).querySelector(".stApp");
          if (appRoot && !appRoot.__hoverLiveBound) {
            appRoot.__hoverLiveBound = true;
            appRoot.addEventListener("mouseover", function(ev) {
              const t = ev.target.closest('a, button, [role="tab"], [role="option"], [data-baseweb="select"]');
              if (!t) return;
              t.style.transition = "all .18s ease";
              t.style.boxShadow = "0 0 0 3px rgba(37,99,235,0.14)";
            }, true);
          }
        } catch (e) {}
        </script>
        """,
        height=0,
    )


def yn_to_int(value: str) -> int:
    return 1 if value == "Yes" else 0


def simple_fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Beginner-friendly missing value handling: median for numeric, mode for categorical."""
    out = df.copy()
    for col in out.columns:
        if out[col].isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].fillna(out[col].median())
        else:
            mode_vals = out[col].mode(dropna=True)
            fill_val = mode_vals.iloc[0] if len(mode_vals) else "Unknown"
            out[col] = out[col].fillna(fill_val)
    return out


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["BalanceSalaryRatio"] = out["Balance"] / (out["EstimatedSalary"] + 1.0)
    out["ProductDensity"] = out["NumOfProducts"] / (out["Tenure"] + 1.0)
    out["EngagementProduct"] = out["IsActiveMember"] * out["NumOfProducts"]
    out["AgeTenureInteraction"] = out["Age"] * out["Tenure"]
    return out


def align_features(df_features: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    for col in feature_names:
        if col not in df_features.columns:
            df_features[col] = 0
    return df_features[feature_names]


def make_single_feature_row(input_raw: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([input_raw])
    df = simple_fill_missing(df)
    df = add_engineered_features(df)
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)
    return align_features(df, feature_names)


def prepare_model_inputs(raw: pd.DataFrame, feature_names: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Shared preprocessing used by scoring, explainability, and diagnostics."""
    prepared = raw.copy()
    prepared = prepared.drop(columns=["CustomerId", "Surname", "Year"], errors="ignore")
    prepared = simple_fill_missing(prepared)

    y = prepared["Exited"].astype(int) if "Exited" in prepared.columns else pd.Series(np.zeros(len(prepared), dtype=int))
    X = prepared.drop(columns=["Exited"], errors="ignore")
    X = add_engineered_features(X)
    X = pd.get_dummies(X, columns=["Geography", "Gender"], drop_first=True)
    X = align_features(X, feature_names)
    return X, y


@st.cache_data
def score_portfolio(feature_names: list[str], _scaler, _model) -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "raw", "European_Bank_data.csv")
    raw = pd.read_csv(csv_path)

    X, y = prepare_model_inputs(raw, feature_names)

    old_err = np.seterr(all="ignore")
    try:
        probs = _model.predict_proba(_scaler.transform(X))[:, 1]
    finally:
        np.seterr(**old_err)

    scored = raw.copy()
    scored["Exited"] = y.values
    scored["churn_probability"] = probs
    return scored


def compute_customer_value(balance: float, products: int, has_card: int, nim: float, fee_per_product: float, card_fee: float) -> float:
    return balance * nim + products * fee_per_product + has_card * card_fee


def predict_single_probability(raw_row: dict[str, Any], feature_names: list[str], _scaler, _model) -> float:
    x_row = make_single_feature_row(raw_row, feature_names)
    return float(_model.predict_proba(_scaler.transform(x_row))[:, 1][0])


def render_recommendation_ticker(recs: list[str]) -> None:
    if not recs:
        return
    pills = "".join([f'<span class="rec-pill">{r}</span>' for r in recs])
    st.markdown(
        f"""
        <div class="rec-ticker-wrap">
          <div class="rec-ticker">
            <div class="rec-track">{pills}{pills}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def recommendations_for_customer(
    prob: float,
    threshold: float,
    expected_loss: float,
    is_active: int,
    products: int,
    geography: str,
    age: int,
) -> list[str]:
    recs: list[str] = []

    if prob >= threshold + 0.20:
        recs.append("Immediate priority: assign relationship manager and launch a 48-hour retention outreach.")
    elif prob >= threshold:
        recs.append("High-priority segment: trigger personalized retention offer within current campaign cycle.")
    else:
        recs.append("Low risk: place in nurture journey and monitor monthly risk trend.")

    if expected_loss >= 5000:
        recs.append("Potential high revenue impact: offer premium service benefits to reduce switching intent.")
    elif expected_loss >= 2000:
        recs.append("Moderate revenue risk: provide targeted loyalty incentives and product bundle discounts.")

    if is_active == 0:
        recs.append("Re-activation plan: push digital engagement campaign and proactive service check-in.")

    if products <= 1:
        recs.append("Cross-sell opportunity: recommend a second product to improve relationship stickiness.")

    if geography == "Germany":
        recs.append("Region watchlist: prioritize Germany segment due to structurally higher observed churn.")

    if age >= 45:
        recs.append("Lifecycle strategy: design age-tailored advisory offers to reduce attrition in mature cohorts.")

    return recs[:6]


def recommendations_for_portfolio(scored: pd.DataFrame, threshold: float, nim: float, fee_per_product: float, card_fee: float) -> list[str]:
    df = scored.copy()
    annual_value = (
        df["Balance"].astype(float) * nim
        + df["NumOfProducts"].astype(float) * fee_per_product
        + df["HasCrCard"].astype(float) * card_fee
    )
    df["expected_loss_eur"] = df["churn_probability"] * annual_value

    churn_rate = float(df["Exited"].mean())
    high_risk_rate = float((df["churn_probability"] >= threshold).mean())
    top_geo = (
        df.groupby("Geography")["churn_probability"].mean().sort_values(ascending=False).index[0]
        if "Geography" in df.columns
        else "N/A"
    )
    total_loss = float(df["expected_loss_eur"].sum())

    recs = []
    if churn_rate > 0.22:
        recs.append("Portfolio alert: churn is elevated; increase retention budget and weekly monitoring cadence.")
    if high_risk_rate > 0.30:
        recs.append("High-risk concentration is large; deploy tiered campaign targeting top-risk deciles first.")
    if total_loss > 5_000_000:
        recs.append("Revenue-at-risk is material; focus immediate interventions on highest expected-loss customers.")
    if top_geo != "N/A":
        recs.append(f"Geography focus: {top_geo} shows highest predicted risk; launch region-specific retention playbook.")

    recs.append("Operational KPI: track save-rate, offer-acceptance, and post-campaign churn monthly.")
    return recs[:6]


def compute_threshold_analytics(
    df: pd.DataFrame,
    intervention_cost_eur: float = 120.0,
) -> tuple[pd.DataFrame, float]:
    if len(df) == 0:
        return pd.DataFrame(), 0.40

    probs = df["churn_probability"].to_numpy(dtype=float)
    actual = df["Exited"].to_numpy(dtype=int)
    exp_loss = df["expected_loss_eur"].to_numpy(dtype=float)

    rows: list[dict[str, float]] = []
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    for t in thresholds:
        pred = (probs >= t).astype(int)
        tp = int(np.sum((pred == 1) & (actual == 1)))
        fp = int(np.sum((pred == 1) & (actual == 0)))
        fn = int(np.sum((pred == 0) & (actual == 1)))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        missed_loss = float(np.sum(exp_loss[(actual == 1) & (pred == 0)]))
        campaign_cost = float(np.sum(pred == 1) * intervention_cost_eur)
        total_cost = missed_loss + campaign_cost
        rows.append(
            {
                "threshold": float(t),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "predicted_churners": int(np.sum(pred == 1)),
                "missed_churners": fn,
                "missed_loss_eur": missed_loss,
                "campaign_cost_eur": campaign_cost,
                "total_business_cost_eur": total_cost,
            }
        )

    out = pd.DataFrame(rows)
    best_threshold = float(out.loc[out["total_business_cost_eur"].idxmin(), "threshold"])
    return out, best_threshold


def action_for_candidate(row: pd.Series, threshold: float) -> str:
    if float(row["churn_probability"]) >= threshold + 0.20:
        return "Immediate RM outreach + premium retention offer"
    if int(row["IsActiveMember"]) == 0:
        return "Reactivation campaign + service quality callback"
    if int(row["NumOfProducts"]) <= 1:
        return "Cross-sell second product with fee waiver"
    return "Monitor weekly + targeted loyalty communication"


def build_top_save_candidates(df: pd.DataFrame, threshold: float, n: int = 20) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    top = df.sort_values("expected_loss_eur", ascending=False).head(n).copy()
    if "CustomerId" in top.columns:
        top["CustomerRef"] = top["CustomerId"].astype(str)
    else:
        top["CustomerRef"] = ("CUS-" + (top.reset_index().index + 1).astype(str)).values
    top["Active"] = top["IsActiveMember"].map({0: "No", 1: "Yes"})
    top["HasCard"] = top["HasCrCard"].map({0: "No", 1: "Yes"})
    top["Action"] = top.apply(lambda r: action_for_candidate(r, threshold), axis=1)

    return top[
        [
            "CustomerRef",
            "Geography",
            "Gender",
            "Age",
            "Balance",
            "NumOfProducts",
            "Active",
            "HasCard",
            "churn_probability",
            "expected_loss_eur",
            "Action",
        ]
    ].rename(
        columns={
            "NumOfProducts": "Products",
            "churn_probability": "PredictedRisk",
            "expected_loss_eur": "ExpectedLossEUR",
        }
    )


def generate_executive_summary_pdf(
    filtered: pd.DataFrame,
    top_candidates: pd.DataFrame,
    threshold: float,
    best_threshold: float,
    customer_inputs: dict[str, Any],
    portfolio_filters: dict[str, Any],
    threshold_curve: pd.DataFrame,
) -> bytes:
    bio = io.BytesIO()
    with PdfPages(bio) as pdf:
        # Page 1: Summary + selected controls
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.axis("off")

        if len(filtered) == 0:
            lines = [
                "Churn Risk Intelligence System - Executive Summary",
                f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
                "No records available for current filters.",
                "",
                "Selected Portfolio Filters:",
                f"- Geography: {portfolio_filters.get('Geography', [])}",
                f"- Gender: {portfolio_filters.get('Gender', [])}",
                f"- Active Member: {portfolio_filters.get('ActiveMember', [])}",
                f"- Age Range: {portfolio_filters.get('AgeRange', '')}",
            ]
            ax.text(0.03, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11, color="#0F172A", family="DejaVu Sans")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            bio.seek(0)
            return bio.read()

        churn_rate = float(filtered["Exited"].mean())
        avg_prob = float(filtered["churn_probability"].mean())
        total_loss = float(filtered["expected_loss_eur"].sum())
        high_risk_count = int((filtered["churn_probability"] >= threshold).sum())
        top_geo = filtered.groupby("Geography")["expected_loss_eur"].sum().sort_values(ascending=False).index[0]

        lines = [
            "Churn Risk Intelligence System - Executive Summary",
            f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "A) Selected Customer Inputs",
            f"- Geography: {customer_inputs.get('Geography')}",
            f"- Gender: {customer_inputs.get('Gender')}",
            f"- Age: {customer_inputs.get('Age')}",
            f"- CreditScore: {customer_inputs.get('CreditScore')}",
            f"- Tenure: {customer_inputs.get('Tenure')}",
            f"- Balance (EUR): {customer_inputs.get('Balance')}",
            f"- Products: {customer_inputs.get('NumOfProducts')}",
            f"- Has Credit Card: {customer_inputs.get('HasCrCard')}",
            f"- Active Member: {customer_inputs.get('IsActiveMember')}",
            f"- EstimatedSalary (EUR): {customer_inputs.get('EstimatedSalary')}",
            "",
            "B) Selected Portfolio Filters",
            f"- Geography: {portfolio_filters.get('Geography')}",
            f"- Gender: {portfolio_filters.get('Gender')}",
            f"- Active Member: {portfolio_filters.get('ActiveMember')}",
            f"- Age Range: {portfolio_filters.get('AgeRange')}",
            "",
            "C) Portfolio Summary",
            f"- Portfolio Size: {len(filtered):,}",
            f"- Observed Churn Rate: {churn_rate:.2%}",
            f"- Average Predicted Churn Probability: {avg_prob:.2%}",
            f"- Expected Annual Revenue Loss (EUR): {total_loss:,.0f}",
            f"- High-Risk Customers @ Threshold {threshold:.2f}: {high_risk_count:,}",
            f"- Cost-Optimized Threshold Suggestion: {best_threshold:.2f}",
            f"- Highest Revenue-at-Risk Geography: {top_geo}",
        ]
        ax.text(0.03, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10.2, color="#0F172A", family="DejaVu Sans", linespacing=1.45)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: Probability distribution + threshold lines
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        sns.histplot(filtered["churn_probability"], bins=28, kde=True, color=PALETTE["sky"], alpha=0.55, ax=ax)
        ax.axvline(threshold, color=PALETTE["purple"], linestyle="--", linewidth=2, label=f"Current threshold ({threshold:.2f})")
        ax.axvline(best_threshold, color=PALETTE["slate"], linewidth=2, label=f"Recommended threshold ({best_threshold:.2f})")
        style_axis(ax, "Predicted Churn Probability Distribution", xlabel="Predicted probability", ylabel="Customers")
        ax.legend(frameon=False, fontsize=9)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: Churn by geography + expected revenue loss
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
        geo = filtered.groupby("Geography", as_index=False).agg(churn_rate=("Exited", "mean"))
        sns.barplot(data=geo, x="Geography", y="churn_rate", palette=[PALETTE["teal"], PASTELTE, PASTELPURP], ax=axes[0])
        style_axis(axes[0], "Churn Rate by Geography", ylabel="Churn rate")

        loss_geo = filtered.groupby("Geography", as_index=False).agg(expected_loss_eur=("expected_loss_eur", "sum"))
        sns.barplot(data=loss_geo, x="Geography", y="expected_loss_eur", palette=["#ECA4A1", "#8FC9C5", "#AFA0E2"], ax=axes[1])
        style_axis(axes[1], "Expected Revenue Loss by Geography (EUR)", ylabel="Expected loss (EUR)")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: Geography x Gender heatmap
        heat = (
            filtered.groupby(["Geography", "Gender"], as_index=False)
            .agg(churn_rate=("Exited", "mean"))
            .pivot(index="Geography", columns="Gender", values="churn_rate")
        )
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        cmap = sns.blend_palette(["#E7E1F8", "#CFC3EE", "#B3A3E2"], as_cmap=True)
        sns.heatmap(
            heat,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            linewidths=1,
            linecolor="white",
            cbar=True,
            ax=ax,
            vmin=0,
            vmax=max(0.30, float(np.nanmax(heat.values)) if np.size(heat.values) else 0.30),
        )
        style_axis(ax, "Geography x Gender Heatmap (Churn Rate)")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 5: Threshold analytics
        if len(threshold_curve):
            fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
            thr_melt = threshold_curve.melt(
                id_vars="threshold",
                value_vars=["precision", "recall", "f1"],
                var_name="Metric",
                value_name="Score",
            )
            sns.lineplot(
                data=thr_melt,
                x="threshold",
                y="Score",
                hue="Metric",
                marker="o",
                palette=[PALETTE["sky"], PALETTE["coral"], PALETTE["teal"]],
                ax=axes[0],
            )
            axes[0].axvline(threshold, color=PALETTE["purple"], linestyle="--", linewidth=2)
            axes[0].axvline(best_threshold, color=PALETTE["slate"], linewidth=2)
            style_axis(axes[0], "Precision / Recall / F1 vs Threshold", xlabel="Threshold", ylabel="Score")
            axes[0].set_ylim(0, 1)

            sns.lineplot(
                data=threshold_curve,
                x="threshold",
                y="total_business_cost_eur",
                marker="o",
                color=PALETTE["amber"],
                ax=axes[1],
            )
            axes[1].axvline(threshold, color=PALETTE["purple"], linestyle="--", linewidth=2)
            axes[1].axvline(best_threshold, color=PALETTE["slate"], linewidth=2)
            style_axis(axes[1], "Business Cost Curve vs Threshold", xlabel="Threshold", ylabel="Total cost (EUR)")
            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

        # Page 6: Top recommendations from candidate list
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.axis("off")
        lines = [
            "Actionable Retention Notes",
            "",
            "Top 5 Save Candidates:",
        ]
        top5 = top_candidates.head(5) if len(top_candidates) else pd.DataFrame()
        if len(top5):
            for _, r in top5.iterrows():
                lines.append(
                    f"- {r['CustomerRef']}: Risk {r['PredictedRisk']:.1%}, Loss EUR {r['ExpectedLossEUR']:,.0f}"
                )
                lines.append(f"  Recommended Action: {r['Action']}")
        else:
            lines.append("- No candidates after current filters.")

        lines.extend(
            [
                "",
                "Portfolio Recommendations:",
                "- Prioritize high expected-loss customers for immediate outreach.",
                "- Use recommended threshold for campaign targeting if budget allows.",
                "- Track save-rate and post-campaign churn monthly by geography.",
            ]
        )
        ax.text(0.04, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10.5, color="#0F172A", family="DejaVu Sans", linespacing=1.45)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    bio.seek(0)
    return bio.read()


# -----------------------------
# App start
# -----------------------------
inject_css()
inject_click_animation()
model, scaler, feature_names = load_artifacts()

# Sidebar controls
st.sidebar.markdown("## Input Controls")
st.sidebar.markdown(
    """
    <div class="sidebar-note">
      <h4>How to Use Inputs</h4>
      <p>1) Select profile and account details.</p>
      <p>2) Click <b>Analyze Customer Risk</b>.</p>
      <p>3) Use What-If to test engagement/product changes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.caption("Yes/No values are automatically converted to model-ready 0/1.")

threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.40, 0.01)

with st.sidebar.expander("🌍 Geography & Profile", expanded=True):
    geography = st.selectbox("🌍 Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Female", "Male"])
    age = st.slider("Age", 18, 100, 40)

with st.sidebar.expander("Account & Engagement", expanded=True):
    credit_score = st.slider("Credit Score", 300, 900, 600)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    balance = float(st.number_input("Balance", 0, 200000, 50000))
    products = int(st.slider("Number of Products", 1, 4, 2))
    has_card = yn_to_int(st.selectbox("Has Credit Card", ["No", "Yes"], index=1))
    is_active = yn_to_int(st.selectbox("Active Member", ["No", "Yes"], index=1))
    salary = float(st.number_input("Estimated Salary", 0, 200000, 70000))

with st.sidebar.expander("💶 Revenue Assumptions", expanded=False):
    nim = st.slider("Net interest margin on balance (annual %)", 0.0, 10.0, 2.0, 0.1) / 100.0
    fee_per_product = float(st.slider("Annual fee per product (EUR)", 0, 500, 50, 5))
    card_fee = float(st.slider("Annual card fee (EUR)", 0, 200, 20, 5))

input_raw = {
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": products,
    "HasCrCard": has_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": salary,
}

single_X = make_single_feature_row(input_raw, feature_names)
prob = predict_single_probability(input_raw, feature_names, scaler, model)
risk_label = "High Risk" if prob >= threshold else "Low Risk"

annual_value_customer = compute_customer_value(balance, products, has_card, nim, fee_per_product, card_fee)
expected_loss_customer = prob * annual_value_customer

# Hero (rendered in components HTML to avoid markdown parsing issues)
import base64
import textwrap

def get_base64_image(image_path):
    with open(image_path, "rb") as img:
        return base64.b64encode(img.read()).decode()

img_base64 = get_base64_image("assets/hero_bg.png")

hero_html = textwrap.dedent(f"""
<style>
.hero-wrap {{
  position: relative;
  height: 260px;
  width: 100%;
  border-radius: 0;
  overflow: hidden;

  background-image: url("data:image/png;base64,{img_base64}");
  background-size: cover;
  background-position: center right;
  background-repeat: no-repeat;

  display: flex;
  align-items: center;
  padding-left: 40px;
}}

.hero-content {{
  max-width: 55%;
  color: white;
}}

.hero-title {{
  font-size: 42px;
  font-weight: 700;
  margin-bottom: 8px;
  text-shadow: 0px 2px 8px rgba(0,0,0,0.6);
}}

.hero-sub {{
  font-size: 16px;
  color: #cbd5ff;
  text-shadow: 0px 1px 6px rgba(0,0,0,0.5);
}}
</style>

<div class="hero-wrap">
  <div class="hero-content">
    <h1 class="hero-title">Churn Risk Intelligence System</h1>
    <p class="hero-sub">Predict. Prevent. Retain High-Value Customers.</p>
  </div>
</div>

""").strip()

components.html(hero_html, height=220, scrolling=False)

col1, col2, col3 = st.columns([1,2,1])

with col2:
    if st.button("Analyze Customer Risk"):
        st.session_state["show_hint"] = True

if st.session_state.get("show_hint"):
    st.toast("**Please Enter customer inputs from the left sidebar.**", icon="ℹ️")
    st.session_state["show_hint"] = False

st.write("")

# Top cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Churn Probability", f"{prob:.2%}")
c2.metric("Risk Flag", risk_label)
c3.metric("Expected Annual Value (EUR)", f"EUR {annual_value_customer:,.0f}")
c4.metric("Expected Annual Loss (EUR)", f"EUR {expected_loss_customer:,.0f}")

# Tabs (core modules + expanded analytics)
tab_calc, tab_whatif, tab_portfolio, tab_features = st.tabs([
    "Risk Calculator",
    "What-If Simulator",
    "Portfolio Analytics",
    "Feature Dashboard",
])

with tab_calc:
    st.markdown("### Customer Churn Risk Calculator")
    scored_calc = score_portfolio(feature_names, scaler, model)
    portfolio_mean_prob = float(scored_calc["churn_probability"].mean())
    geo_mean_prob = float(scored_calc.loc[scored_calc["Geography"] == geography, "churn_probability"].mean())
    percentile_rank = float((scored_calc["churn_probability"] <= prob).mean())

    left, right = st.columns(2, gap="small")
    with left:
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        ax.pie(
            [prob, 1 - prob],
            labels=["Churn", "No Churn"],
            autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
            colors=["#F6B3B1", "#BFE8C8"],
            startangle=120,
            pctdistance=0.62,
            labeldistance=1.08,
            textprops={"fontsize": 10, "fontweight": "bold", "color": "#1F2937"},
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
        )
        style_axis(ax, "Probability Split")
        fig.tight_layout(pad=1.1)
        st.pyplot(fig)

    with right:
        compare_df = pd.DataFrame(
            {"Metric": ["Probability", "Threshold"], "Value": [prob, threshold]}
        )
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        sns.barplot(data=compare_df, x="Metric", y="Value", palette=[PALETTE["sky"], PASTELTE], ax=ax)
        ax.set_ylim(0, 1)
        style_axis(ax, "Probability vs Threshold", ylabel="Value")
        annotate_bars(ax, percent=True)
        st.pyplot(fig)

    b1, b2, b3 = st.columns(3)
    b1.metric("Portfolio Probability Avg", f"{portfolio_mean_prob:.2%}")
    b2.metric(f"{geography} Probability Avg", f"{geo_mean_prob:.2%}")
    b3.metric("Risk Percentile Position", f"{percentile_rank:.1%}")

    extra1, extra2 = st.columns(2, gap="small")
    with extra1:
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        ax.barh([0], [0.30], left=0.00, color="#BCE7D2", edgecolor="white", label="Low")
        ax.barh([0], [0.30], left=0.30, color="#FDE2B8", edgecolor="white", label="Medium")
        ax.barh([0], [0.40], left=0.60, color="#F6C7C3", edgecolor="white", label="High")
        ax.axvline(prob, color=PALETTE["slate"], linestyle="-", linewidth=3, label="Customer")
        ax.axvline(threshold, color=PALETTE["purple"], linestyle="--", linewidth=2, label="Threshold")
        ax.set_xlim(0, 1)
        ax.set_yticks([])
        style_axis(ax, "Risk Zone Position", xlabel="Probability")
        ax.legend(loc="upper center", ncol=3, fontsize=9, frameon=False)
        st.pyplot(fig)

    with extra2:
        interest_component = balance * nim
        product_component = products * fee_per_product
        card_component = has_card * card_fee
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        ax.pie(
            [interest_component, product_component, card_component],
            labels=["Interest Income", "Product Fees", "Card Fee"],
            colors=[PALETTE["teal"], PALETTE["sky"], PASTELPURP],
            autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
            startangle=90,
            pctdistance=0.72,
            labeldistance=1.06,
            textprops={"fontsize": 9, "fontweight": "bold", "color": "#1F2937"},
            wedgeprops={"width": 0.45, "edgecolor": "white"},
        )
        style_axis(ax, "Annual Value Composition")
        fig.tight_layout(pad=1.0)
        st.pyplot(fig)

    extra3, extra4 = st.columns(2, gap="small")
    with extra3:
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        sns.histplot(
            scored_calc["churn_probability"],
            bins=25,
            kde=True,
            color=PALETTE["sky"],
            alpha=0.55,
            ax=ax,
        )
        ax.axvline(prob, color=PALETTE["slate"], linewidth=2.5, label="Customer")
        ax.axvline(threshold, color=PALETTE["purple"], linestyle="--", linewidth=2, label="Threshold")
        style_axis(ax, "Portfolio Probability Distribution", xlabel="Predicted churn probability", ylabel="Customers")
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig)

    with extra4:
        benchmark_df = pd.DataFrame(
            {
                "Benchmark": ["Customer", f"{geography} Avg", "Portfolio Avg"],
                "Probability": [prob, geo_mean_prob, portfolio_mean_prob],
            }
        )
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        sns.barplot(
            data=benchmark_df,
            x="Benchmark",
            y="Probability",
            palette=[PALETTE["coral"], PALETTE["mint"], PALETTE["sky"]],
            ax=ax,
        )
        ax.set_ylim(0, 1)
        style_axis(ax, "Customer vs Benchmarks", ylabel="Probability")
        annotate_bars(ax, percent=True)
        st.pyplot(fig)

    st.markdown("### Business Recommendation (Customer-Level)")
    recs = recommendations_for_customer(
        prob,
        threshold,
        expected_loss_customer,
        is_active,
        products,
        geography,
        age,
    )
    render_recommendation_ticker(recs)
    for rec in recs:
        st.write(f"- {rec}")

with tab_whatif:
    st.markdown("### What-If Scenario Simulator")
    st.caption("Adjust engagement and product signals to observe churn probability change.")

    w1, w2, w3 = st.columns(3)
    with w1:
        sim_active = yn_to_int(st.selectbox("Scenario: Active Member", ["No", "Yes"], index=is_active, key="sim_active"))
    with w2:
        sim_products = int(st.slider("Scenario: Number of Products", 1, 4, products, key="sim_products"))
    with w3:
        sim_balance = float(st.number_input("Scenario: Balance", 0, 200000, int(balance), key="sim_balance"))

    sim_raw = dict(input_raw)
    sim_raw["IsActiveMember"] = sim_active
    sim_raw["NumOfProducts"] = sim_products
    sim_raw["Balance"] = sim_balance

    sim_prob = predict_single_probability(sim_raw, feature_names, scaler, model)

    s1, s2, s3 = st.columns(3)
    s1.metric("Current", f"{prob:.2%}")
    s2.metric("Scenario", f"{sim_prob:.2%}")
    s3.metric("Delta", f"{(sim_prob - prob):+.2%}")

    sim_annual_value = compute_customer_value(sim_balance, sim_products, has_card, nim, fee_per_product, card_fee)
    sim_expected_loss = sim_prob * sim_annual_value

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    sim_c1.metric("Scenario Annual Value (EUR)", f"EUR {sim_annual_value:,.0f}")
    sim_c2.metric("Scenario Expected Loss (EUR)", f"EUR {sim_expected_loss:,.0f}")
    sim_c3.metric("Loss Change", f"EUR {(sim_expected_loss - expected_loss_customer):+,.0f}")

    sw1, sw2 = st.columns(2)
    with sw1:
        line = pd.DataFrame({"State": ["Current", "Scenario"], "Probability": [prob, sim_prob]})
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        sns.lineplot(data=line, x="State", y="Probability", marker="o", linewidth=3, color=PALETTE["purple"], ax=ax)
        style_axis(ax, "Current vs Scenario Probability", ylabel="Probability", pastel_bg=True)
        st.pyplot(fig)

    with sw2:
        kpi_df = pd.DataFrame(
            {
                "Metric": ["Probability", "Expected Loss (scaled)"],
                "Current": [prob, expected_loss_customer / max(1.0, annual_value_customer)],
                "Scenario": [sim_prob, sim_expected_loss / max(1.0, sim_annual_value)],
            }
        )
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        kpi_melt = kpi_df.melt(id_vars="Metric", value_vars=["Current", "Scenario"], var_name="State", value_name="Value")
        sns.barplot(data=kpi_melt, x="Metric", y="Value", hue="State", palette=[PALETTE["sky"], PALETTE["coral"]], ax=ax)
        style_axis(ax, "Current vs Scenario KPI Comparison", ylabel="Relative value", pastel_bg=True)
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig)

    sw3, sw4 = st.columns(2)
    with sw3:
        delta_df = pd.DataFrame(
            {
                "Driver": ["Balance", "Products", "Active Member"],
                "DeltaPct": [
                    ((sim_balance - balance) / max(1.0, balance)) * 100,
                    ((sim_products - products) / max(1.0, products)) * 100,
                    (sim_active - is_active) * 100,
                ],
            }
        )
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        sns.barplot(
            data=delta_df,
            x="Driver",
            y="DeltaPct",
            palette=[PALETTE["mint"], PASTELPURP, PALETTE["amber"]],
            ax=ax,
        )
        style_axis(ax, "Scenario Input Delta (%)", ylabel="Change (%)", pastel_bg=True)
        annotate_bars(ax)
        st.pyplot(fig)

    with sw4:
        scored_sim = score_portfolio(feature_names, scaler, model)
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        sns.histplot(scored_sim["churn_probability"], bins=25, color=PALETTE["teal"], alpha=0.48, ax=ax)
        ax.axvline(prob, color=PALETTE["slate"], linewidth=2.4, label="Current")
        ax.axvline(sim_prob, color=PALETTE["coral"], linewidth=2.4, label="Scenario")
        style_axis(ax, "Current/Scenario vs Portfolio Distribution", xlabel="Predicted churn probability", ylabel="Customers", pastel_bg=True)
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig)

    sw5, sw6 = st.columns(2)
    with sw5:
        one_at_time = []
        for key, label in [("IsActiveMember", "Active Member"), ("NumOfProducts", "Products"), ("Balance", "Balance")]:
            test_raw = dict(input_raw)
            test_raw[key] = sim_raw[key]
            p_test = predict_single_probability(test_raw, feature_names, scaler, model)
            one_at_time.append({"Driver": label, "DeltaProb": p_test - prob})
        impact_df = pd.DataFrame(one_at_time)
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        sns.barplot(
            data=impact_df,
            x="Driver",
            y="DeltaProb",
            palette=[PALETTE["sky"], PALETTE["purple"], PALETTE["coral"]],
            ax=ax,
        )
        style_axis(ax, "One-at-a-Time Driver Impact", ylabel="Probability delta", pastel_bg=True)
        annotate_bars(ax, percent=True)
        st.pyplot(fig)

    with sw6:
        sim_benchmark = pd.DataFrame(
            {
                "State": ["Current", "Scenario"],
                "Probability": [prob, sim_prob],
                "Expected Loss (EUR)": [expected_loss_customer, sim_expected_loss],
            }
        )
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        sns.scatterplot(
            data=sim_benchmark,
            x="Probability",
            y="Expected Loss (EUR)",
            hue="State",
            s=320,
            palette=[PALETTE["sky"], PALETTE["coral"]],
            ax=ax,
        )
        for _, row in sim_benchmark.iterrows():
            ax.text(row["Probability"] + 0.01, row["Expected Loss (EUR)"], row["State"], fontsize=10, fontweight="bold")
        style_axis(ax, "Risk vs Revenue Loss Positioning", xlabel="Churn probability", ylabel="Expected loss (EUR)", pastel_bg=True)
        ax.legend(frameon=False, fontsize=9)
        st.pyplot(fig)

    st.markdown("### Scenario Recommendation")
    sim_recs = recommendations_for_customer(
        sim_prob,
        threshold,
        sim_expected_loss,
        sim_active,
        sim_products,
        geography,
        age,
    )
    render_recommendation_ticker(sim_recs)
    for rec in sim_recs:
        st.write(f"- {rec}")

with tab_portfolio:
    st.markdown("### Portfolio-Level Analytics")
    scored = score_portfolio(feature_names, scaler, model)

    # Filters
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        geo_f = st.multiselect("🌍 Geography", ["France", "Germany", "Spain"], default=["France", "Germany", "Spain"])
    with f2:
        gender_f = st.multiselect("Gender", ["Female", "Male"], default=["Female", "Male"])
    with f3:
        active_f = st.multiselect("Active Member", ["No", "Yes"], default=["No", "Yes"])
    with f4:
        age_range = st.slider("Age Range", 18, 100, (18, 100))

    active_vals = [1 if v == "Yes" else 0 for v in active_f]
    filtered = scored[
        scored["Geography"].isin(geo_f)
        & scored["Gender"].isin(gender_f)
        & scored["IsActiveMember"].isin(active_vals)
        & (scored["Age"] >= age_range[0])
        & (scored["Age"] <= age_range[1])
    ].copy()

    annual_value = (
        filtered["Balance"].astype(float) * nim
        + filtered["NumOfProducts"].astype(float) * fee_per_product
        + filtered["HasCrCard"].astype(float) * card_fee
    )
    filtered["expected_loss_eur"] = filtered["churn_probability"] * annual_value

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Customers", f"{len(filtered):,}")
    p2.metric("Actual Churn Rate", f"{filtered['Exited'].mean():.2%}")
    p3.metric("Avg Predicted Probability", f"{filtered['churn_probability'].mean():.2%}")
    p4.metric("Expected Revenue Loss (EUR)", f"EUR {filtered['expected_loss_eur'].sum():,.0f}")

    st.markdown("### Comparison Graphs")

    g1, g2 = st.columns(2)
    with g1:
        # Risk band calibration
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        band_df = filtered.copy()
        band_df["risk_band"] = pd.cut(band_df["churn_probability"], bins=bins, labels=labels, include_lowest=True)
        band_stats = (
            band_df.groupby("risk_band", as_index=False)
            .agg(customers=("Exited", "count"), churn_rate=("Exited", "mean"), avg_pred=("churn_probability", "mean"))
            .dropna()
        )

        fig, ax1 = plt.subplots(figsize=CHART_FIGSIZE)
        ax2 = ax1.twinx()
        sns.barplot(data=band_stats, x="risk_band", y="churn_rate", color=PALETTE["mint"], ax=ax1)
        sns.lineplot(data=band_stats, x="risk_band", y="avg_pred", marker="o", color=PALETTE["sky"], ax=ax2)
        style_axis(ax1, "Churn Rate vs Predicted Risk Bands", "Risk Band", "Actual churn rate")
        ax2.set_ylabel("Avg predicted probability", fontsize=LABEL_SIZE, fontweight="bold")
        ax2.tick_params(labelsize=10)
        ax1.tick_params(axis="x", rotation=0)
        annotate_bars(ax1, percent=True)
        st.pyplot(fig)

    with g2:
        # Medium-tone heatmap
        heat = (
            filtered.groupby(["Geography", "Gender"], as_index=False)
            .agg(churn_rate=("Exited", "mean"))
            .pivot(index="Geography", columns="Gender", values="churn_rate")
        )
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        cmap = sns.blend_palette(["#E7E1F8", "#CFC3EE", "#B3A3E2"], as_cmap=True)
        sns.heatmap(
            heat,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            linewidths=1,
            linecolor="white",
            cbar=True,
            ax=ax,
            vmin=0,
            vmax=max(0.30, float(np.nanmax(heat.values)) if np.size(heat.values) else 0.30),
        )
        style_axis(ax, "Geography x Gender Heatmap (Churn Rate)")
        st.pyplot(fig)

    g3, g4 = st.columns(2)
    with g3:
        geo = filtered.groupby("Geography", as_index=False).agg(churn_rate=("Exited", "mean"))
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        sns.barplot(data=geo, x="Geography", y="churn_rate", palette=[PALETTE["teal"], PASTELTE, PASTELPURP], ax=ax)
        style_axis(ax, "Churn Rate by Geography", ylabel="Churn rate")
        annotate_bars(ax, percent=True)
        st.pyplot(fig)

    with g4:
        active = filtered.groupby("IsActiveMember", as_index=False).agg(churn_rate=("Exited", "mean"))
        active["Active"] = active["IsActiveMember"].map({0: "No", 1: "Yes"})
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        sns.barplot(data=active, x="Active", y="churn_rate", palette=[PALETTE["coral"], PASTELTE], ax=ax)
        style_axis(ax, "Churn Rate by Activity", ylabel="Churn rate")
        annotate_bars(ax, percent=True)
        st.pyplot(fig)

    g5, g6 = st.columns(2)
    with g5:
        prod = filtered.groupby("NumOfProducts", as_index=False).agg(churn_rate=("Exited", "mean"))
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        sns.barplot(data=prod, x="NumOfProducts", y="churn_rate", color=PALETTE["amber"], ax=ax)
        style_axis(ax, "Churn Rate by Product Count", "Number of products", "Churn rate")
        annotate_bars(ax, percent=True)
        st.pyplot(fig)

    with g6:
        sample = filtered.sample(n=min(2200, len(filtered)), random_state=42) if len(filtered) > 0 else filtered
        fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
        sns.scatterplot(
            data=sample,
            x="Balance",
            y="churn_probability",
            hue="Exited",
            palette={0: "#5FAFA8", 1: "#6F5BC6"},
            alpha=0.80,
            s=56,
            ax=ax,
            legend=False,
        )
        style_axis(ax, "Balance vs Predicted Churn Probability", ylabel="Predicted probability")
        st.pyplot(fig)

    # Revenue loss by geography
    loss_geo = (
        filtered.groupby("Geography", as_index=False)
        .agg(expected_loss_eur=("expected_loss_eur", "sum"))
        .sort_values("expected_loss_eur", ascending=False)
    )
    rg1, rg2 = st.columns([1.15, 0.85])
    with rg1:
        fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
        sns.barplot(
            data=loss_geo,
            x="Geography",
            y="expected_loss_eur",
            palette=["#ECA4A1", "#8FC9C5", "#AFA0E2"],
            ax=ax,
        )
        style_axis(ax, "Expected Revenue Loss by Geography (EUR)", ylabel="Expected loss (EUR)")
        annotate_bars(ax, currency=True)
        st.pyplot(fig)
    with rg2:
        top_loss = filtered.nlargest(8, "expected_loss_eur")[["Geography", "expected_loss_eur"]].copy()
        top_loss["Customer"] = [f"C{i+1}" for i in range(len(top_loss))]
        fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)
        sns.barplot(data=top_loss, x="Customer", y="expected_loss_eur", color=PALETTE["purple"], ax=ax)
        style_axis(ax, "Top Customers by Expected Loss", ylabel="Expected loss (EUR)")
        st.pyplot(fig)

    st.markdown("### Business Recommendations")
    portfolio_recs = recommendations_for_portfolio(filtered, threshold, nim, fee_per_product, card_fee)
    render_recommendation_ticker(portfolio_recs)
    for rec in portfolio_recs:
        st.write(f"- {rec}")

    st.markdown("### Threshold Optimizer")
    thr_df, best_t = compute_threshold_analytics(filtered, intervention_cost_eur=120.0)
    if len(thr_df):
        t1, t2 = st.columns(2)
        with t1:
            fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
            thr_melt = thr_df.melt(
                id_vars="threshold",
                value_vars=["precision", "recall", "f1"],
                var_name="Metric",
                value_name="Score",
            )
            sns.lineplot(
                data=thr_melt,
                x="threshold",
                y="Score",
                hue="Metric",
                marker="o",
                palette=[PALETTE["sky"], PALETTE["coral"], PALETTE["teal"]],
                ax=ax,
            )
            ax.axvline(threshold, color=PALETTE["purple"], linestyle="--", linewidth=2, label="Current")
            ax.axvline(best_t, color=PALETTE["slate"], linestyle="-", linewidth=2, label="Best Cost")
            style_axis(ax, "Precision/Recall/F1 vs Threshold", xlabel="Threshold", ylabel="Score")
            ax.set_ylim(0, 1)
            ax.legend(frameon=False, fontsize=9)
            st.pyplot(fig)

        with t2:
            fig, ax = plt.subplots(figsize=CHART_FIGSIZE)
            sns.lineplot(
                data=thr_df,
                x="threshold",
                y="total_business_cost_eur",
                marker="o",
                color=PALETTE["amber"],
                ax=ax,
            )
            ax.axvline(threshold, color=PALETTE["purple"], linestyle="--", linewidth=2)
            ax.axvline(best_t, color=PALETTE["slate"], linestyle="-", linewidth=2)
            style_axis(ax, "Business Cost Curve vs Threshold", xlabel="Threshold", ylabel="Total cost (EUR)")
            st.pyplot(fig)

        o1, o2, o3 = st.columns(3)
        o1.metric("Current Threshold", f"{threshold:.2f}")
        o2.metric("Recommended Threshold", f"{best_t:.2f}")
        best_cost = float(thr_df.loc[thr_df["threshold"] == best_t, "total_business_cost_eur"].iloc[0])
        cur_cost = float(thr_df.loc[thr_df["threshold"] == round(threshold, 2), "total_business_cost_eur"].iloc[0]) if round(threshold, 2) in set(thr_df["threshold"]) else best_cost
        o3.metric("Estimated Cost Delta", f"EUR {(best_cost - cur_cost):,.0f}")

    candidates = build_top_save_candidates(filtered, threshold, n=20)

    st.markdown("### Executive Summary Export")
    customer_inputs_for_pdf = {
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "CreditScore": credit_score,
        "Tenure": tenure,
        "Balance": f"{balance:,.0f}",
        "NumOfProducts": products,
        "HasCrCard": "Yes" if has_card == 1 else "No",
        "IsActiveMember": "Yes" if is_active == 1 else "No",
        "EstimatedSalary": f"{salary:,.0f}",
    }
    portfolio_filters_for_pdf = {
        "Geography": ", ".join(geo_f),
        "Gender": ", ".join(gender_f),
        "ActiveMember": ", ".join(active_f),
        "AgeRange": f"{age_range[0]} - {age_range[1]}",
    }
    pdf_bytes = generate_executive_summary_pdf(
        filtered=filtered,
        top_candidates=candidates,
        threshold=threshold,
        best_threshold=best_t if len(thr_df) else threshold,
        customer_inputs=customer_inputs_for_pdf,
        portfolio_filters=portfolio_filters_for_pdf,
        threshold_curve=thr_df if len(thr_df) else pd.DataFrame(),
    )
    st.download_button(
        "Download Executive Summary (PDF)",
        data=pdf_bytes,
        file_name="churn_executive_summary.pdf",
        mime="application/pdf",
    )

with tab_features:
    st.markdown("### Feature Importance Dashboard")

    if hasattr(model, "coef_"):
        imp = pd.DataFrame({"Feature": feature_names, "Importance": np.abs(model.coef_[0])}).sort_values("Importance", ascending=False).head(15)
    elif hasattr(model, "feature_importances_"):
        imp = pd.DataFrame({"Feature": feature_names, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False).head(15)
    else:
        imp = pd.DataFrame({"Feature": feature_names, "Importance": np.zeros(len(feature_names))}).head(15)

    fig, ax = plt.subplots(figsize=FEATURE_FIGSIZE)
    feature_colors = sns.color_palette(["#8FC9C5", "#AFA0E2", "#ECA4A1", "#A8D7F0", "#DAC38D"], n_colors=len(imp))
    sns.barplot(data=imp, y="Feature", x="Importance", palette=feature_colors, ax=ax)
    style_axis(ax, "Top Feature Importance", ylabel="Feature", xlabel="Importance")
    ax.tick_params(axis="y", labelsize=10)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    st.pyplot(fig)

    st.markdown("### Additional Analysis View")
    scored = score_portfolio(feature_names, scaler, model)
    corr_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
        "churn_probability",
    ]
    corr = scored[corr_cols].corr(numeric_only=True)
    ax1, ax2 = st.columns(2)
    with ax1:
        fig, ax = plt.subplots(figsize=FEATURE_HEAT_FIGSIZE)
        sns.heatmap(corr, cmap=sns.light_palette(PALETTE["sky"], as_cmap=True), annot=True, fmt=".2f", linewidths=0.7, ax=ax)
        style_axis(ax, "Correlation Map (Numeric Features)")
        st.pyplot(fig)
    with ax2:
        risk_by_geo = scored.groupby("Geography", as_index=False)["churn_probability"].mean()
        fig, ax = plt.subplots(figsize=FEATURE_HEAT_FIGSIZE)
        sns.barplot(data=risk_by_geo, x="Geography", y="churn_probability", palette=[PALETTE["sky"], PALETTE["purple"], "#EDB7D1"], ax=ax)
        style_axis(ax, "Avg Predicted Risk by Geography", ylabel="Predicted probability")
        annotate_bars(ax, percent=True)
        st.pyplot(fig)

    st.markdown("### Explainability (SHAP + Partial Dependence)")
    st.caption("Optional advanced view for interview discussion. Safe fallbacks are included if a library is unavailable.")

    raw_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "European_Bank_data.csv")
    raw_for_xai = pd.read_csv(raw_path)
    X_xai, _ = prepare_model_inputs(raw_for_xai, feature_names)
    X_xai_scaled = scaler.transform(X_xai)

    try:
        import shap

        sample_n = min(300, len(X_xai_scaled))
        bg = X_xai_scaled[:sample_n]
        explainer = shap.LinearExplainer(model, bg)
        shap_values = explainer(bg)

        mean_abs = np.abs(shap_values.values).mean(axis=0)
        shap_df = (
            pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": mean_abs})
            .sort_values("MeanAbsSHAP", ascending=False)
            .head(10)
        )
        fig, ax = plt.subplots(figsize=FEATURE_FIGSIZE)
        sns.barplot(data=shap_df, y="Feature", x="MeanAbsSHAP", color=PALETTE["sky"], ax=ax)
        style_axis(ax, "Global SHAP Importance (Top 10)", ylabel="Feature", xlabel="Mean |SHAP value|")
        st.pyplot(fig)
    except Exception:
        st.info("SHAP summary is unavailable in this environment, so this section is safely skipped.")

    try:
        from sklearn.inspection import PartialDependenceDisplay

        top_two = imp["Feature"].head(2).tolist()
        top_idx = [feature_names.index(f) for f in top_two if f in feature_names]
        if len(top_idx) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
            PartialDependenceDisplay.from_estimator(
                model,
                X_xai_scaled,
                features=[top_idx[0]],
                feature_names=feature_names,
                ax=axes[0],
            )
            axes[0].set_title(f"PDP: {top_two[0]}", fontsize=11, fontweight="bold")
            PartialDependenceDisplay.from_estimator(
                model,
                X_xai_scaled,
                features=[top_idx[1]],
                feature_names=feature_names,
                ax=axes[1],
            )
            axes[1].set_title(f"PDP: {top_two[1]}", fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
    except Exception:
        st.info("Partial dependence plot is unavailable in this environment, so this section is safely skipped.")
