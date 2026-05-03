import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.express as px
import plotly.graph_objects as go
import os

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="SENTINEL IDS", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .block-container { padding-top: 1.5rem; }
    .alert-box {
        padding: 0.75rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    .alert-attack { background: #3d1a1a; border-left: 4px solid #f85149; color: #f85149; }
    .alert-benign { background: #122d22; border-left: 4px solid #3fb950; color: #3fb950; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 600; padding: 0.5rem 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Load Artifacts
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model    = joblib.load('models/global_ids_model.pkl')
    features = joblib.load('models/global_feature_list.pkl')
    le       = joblib.load('models/label_encoder.pkl')
    stats    = joblib.load('models/scaling_stats.pkl')
    feat_stats = joblib.load('models/feature_medians.pkl')  # has medians, minvals, maxvals

    eval_results = None
    if os.path.exists('models/evaluation_results.pkl'):
        eval_results = joblib.load('models/evaluation_results.pkl')

    return model, features, le, stats, feat_stats, eval_results

try:
    model, top_features, le, scaling_stats, feat_stats, eval_results = load_assets()
except Exception as e:
    st.error(f"**Could not load model files.** Run `model_training.py` first.\n\n`{e}`")
    st.stop()


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ SENTINEL IDS")
    st.caption("ML-Based Intrusion Detection System")
    st.markdown("---")
    st.markdown(f"**Model:** Random Forest\n\n**Features:** {len(top_features)}\n\n**Classes:** {len(le.classes_)}")
    st.markdown("---")
    st.markdown("**Attack Classes Monitored:**")
    for c in le.classes_:
        colour = "#3fb950" if c == "BENIGN" else "#f85149"
        st.markdown(f"<span style='color:{colour}'>● {c}</span>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────
def preprocess_row(row_df):
    row = row_df.reindex(columns=top_features, fill_value=0)
    row = row.replace([np.inf, -np.inf], 0).fillna(0)
    scaled = (row - scaling_stats['mean'][top_features]) / (scaling_stats['std'][top_features] + 1e-9)
    return scaled


# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📡  Live Traffic Monitor",
    "🔬  Manual Packet Inspector",
    "📊  Model Accuracy Report"
])


# ═════════════════════════════════════════════════════════════
# TAB 1 — Live Traffic Simulation
# ═════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Live Network Traffic Classification")
    st.caption("Upload the demo CSV to simulate real-time packet inspection.")

    ctrl1, ctrl2, ctrl3 = st.columns([2, 1, 1])
    with ctrl1:
        uploaded_file = st.file_uploader("Upload demo_traffic.csv", type=["csv"], label_visibility="collapsed")
    with ctrl2:
        sim_speed = st.slider("Speed (sec/packet)", 0.05, 2.0, 0.3, step=0.05)
    with ctrl3:
        start_btn = st.button("▶ Start Simulation", type="primary", use_container_width=True)

    st.markdown("---")
    feed_col, stats_col = st.columns([3, 1])

    with feed_col:
        feed_header = st.empty()
        feed_table  = st.empty()
        alert_box   = st.empty()

    with stats_col:
        st.markdown("**Session Statistics**")
        metric_scanned = st.empty()
        metric_attacks = st.empty()
        metric_rate    = st.empty()
        st.markdown("---")
        st.markdown("**Attack Breakdown**")
        chart_placeholder = st.empty()

    if uploaded_file and start_btn:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()

        alert_count   = 0
        total_scanned = 0
        attack_log    = []
        recent_rows   = []

        for i in range(len(df)):
            row = df.iloc[[i]]
            total_scanned += 1

            row_scaled = preprocess_row(row)
            pred_idx   = model.predict(row_scaled.values)[0]
            prediction = le.inverse_transform([pred_idx])[0]
            is_attack  = prediction != "BENIGN"

            if is_attack:
                alert_count += 1
                attack_log.append(prediction)

            display_row = {'Packet #': total_scanned, 'Prediction': prediction,
                           'Status': '🚨 ATTACK' if is_attack else '✅ Normal'}
            for feat in top_features[:3]:
                if feat in row.columns:
                    display_row[feat] = round(float(row[feat].values[0]), 4)

            recent_rows.append(display_row)
            if len(recent_rows) > 8:
                recent_rows.pop(0)

            feed_header.markdown(f"**Scanning packet #{total_scanned} of {len(df)}**")
            feed_table.dataframe(pd.DataFrame(recent_rows[::-1]), use_container_width=True, hide_index=True)

            if is_attack:
                alert_box.markdown(
                    f'<div class="alert-box alert-attack">🚨 THREAT DETECTED &nbsp;|&nbsp; {prediction} &nbsp;|&nbsp; Packet #{total_scanned}</div>',
                    unsafe_allow_html=True)
            else:
                alert_box.markdown('<div class="alert-box alert-benign">✅ Traffic Normal</div>', unsafe_allow_html=True)

            rate = round((alert_count / total_scanned) * 100, 1)
            metric_scanned.metric("Packets Scanned", total_scanned)
            metric_attacks.metric("Attacks Detected", alert_count)
            metric_rate.metric("Attack Rate", f"{rate}%")

            if attack_log:
                from collections import Counter
                counts = Counter(attack_log)
                fig = px.bar(x=list(counts.values()), y=list(counts.keys()),
                             orientation='h', color=list(counts.values()),
                             color_continuous_scale='Reds', template='plotly_dark')
                fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                                  coloraxis_showscale=False, height=250,
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  yaxis_title=None, xaxis_title="Count")
                chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"simulation_plot_step_{i}")

            time.sleep(sim_speed)

        feed_header.markdown(f"**✅ Simulation complete — {total_scanned} packets analysed.**")
    elif not uploaded_file:
        st.info("Upload `demo_traffic.csv` (from `data/processed/`) using the file picker above, then click Start.")


# ═════════════════════════════════════════════════════════════
# TAB 2 — Manual Packet Inspector
# ═════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Manual Packet Inspector")
    st.caption("Enter feature values to classify a single network packet. Sliders reflect real training-data ranges.")

    st.markdown("---")
    input_col, result_col = st.columns([1, 1])

    medians = feat_stats.get('medians', {})
    minvals = feat_stats.get('minvals', {})
    maxvals = feat_stats.get('maxvals', {})

    with input_col:
        st.markdown("**Feature Input**")
        user_inputs = {}
        for feat in top_features:
            med = float(medians.get(feat, 0.0))
            lo  = float(minvals.get(feat, 0.0))
            hi  = float(maxvals.get(feat, max(med * 10, 1.0)))

            # Ensure lo < hi and med is within range
            if lo >= hi:
                lo = 0.0
                hi = max(med * 2, 1.0)
            med = float(np.clip(med, lo, hi))

            user_inputs[feat] = st.slider(
                label=feat,
                min_value=lo,
                max_value=hi,
                value=med,
                format="%.2f",
                key=f"slider_{feat}"
            )

    with result_col:
        st.markdown("**Classification Result**")
        classify_btn = st.button("🔍 Classify This Packet", type="primary", use_container_width=True)

        if classify_btn:
            input_df     = pd.DataFrame([user_inputs])
            input_scaled = preprocess_row(input_df)

            pred_idx   = model.predict(input_scaled.values)[0]
            prediction = le.inverse_transform([pred_idx])[0]
            is_attack  = prediction != "BENIGN"

            if is_attack:
                st.markdown(
                    f'<div class="alert-box alert-attack" style="font-size:1.3rem;padding:1rem 1.5rem;">🚨 {prediction}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="alert-box alert-benign" style="font-size:1.3rem;padding:1rem 1.5rem;">✅ BENIGN — Normal Traffic</div>',
                    unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Model Confidence**")
            st.caption("Vote fraction across all 100 trees for each class.")

            try:
                proba      = model.predict_proba(input_scaled.values)[0]
                class_names = le.classes_
                proba_df   = pd.DataFrame({'Class': class_names, 'Confidence': proba}).sort_values('Confidence', ascending=True)

                fig = px.bar(proba_df, x='Confidence', y='Class', orientation='h',
                             color='Confidence', color_continuous_scale=['#1f6feb', '#f85149'],
                             template='plotly_dark',
                             text=proba_df['Confidence'].apply(lambda x: f"{x*100:.1f}%"))
                fig.update_traces(textposition='outside')
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
                                  coloraxis_showscale=False, height=420,
                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                  xaxis_title="Confidence", yaxis_title=None, xaxis_tickformat='.0%')
                st.plotly_chart(fig, use_container_width=True)

                top_row = proba_df.sort_values('Confidence', ascending=False).iloc[0]
                st.info(f"**{top_row['Confidence']*100:.1f}% confident** this packet is **{top_row['Class']}**.")

            except Exception as e:
                st.warning(f"Could not generate confidence chart: {e}")

        else:
            st.markdown("""
            <div style='padding:2rem;border:1px dashed #30363d;border-radius:8px;text-align:center;color:#8b949e;'>
                Adjust sliders on the left, then click<br>
                <strong>Classify This Packet</strong> to see the result.
            </div>""", unsafe_allow_html=True)

            with st.expander("📖 What do these features mean?"):
                st.markdown("""
| Feature | Meaning |
|---|---|
| Flow Duration | Total duration of the network flow (microseconds) |
| Total Fwd Packets | Packets sent forward (client → server) |
| Total Backward Packets | Packets sent backward (server → client) |
| Fwd Packet Length Max | Largest forward packet in bytes |
| Flow Bytes/s | Bytes transferred per second |
| Flow Packets/s | Packets transmitted per second |
| Packet Length Mean | Average size of all packets |
| Flow IAT Mean | Mean inter-arrival time between packets |
| PSH Flag Count | Push flag count — high = interactive/urgent traffic |
| SYN Flag Count | SYN flags — high count suggests SYN flood attack |
""")
                st.caption("Values derived from raw PCAP files using CICFlowMeter.")


# ═════════════════════════════════════════════════════════════
# TAB 3 — Model Accuracy Report
# ═════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Model Accuracy Report")
    st.caption("Performance metrics from the trained Random Forest model on the held-out test set (20% of data).")

    if eval_results is None:
        st.warning("No evaluation results found. Run `model_training.py` to generate them.")
    else:
        report      = eval_results['report']
        cm          = eval_results['confusion_matrix']
        class_names = eval_results['class_names']

        # ── Top-level summary metrics ────────────────────────────────────────
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        overall_acc = report.get('accuracy', 0)
        macro_f1    = report.get('macro avg', {}).get('f1-score', 0)
        macro_prec  = report.get('macro avg', {}).get('precision', 0)
        macro_rec   = report.get('macro avg', {}).get('recall', 0)

        m1.metric("Overall Accuracy",  f"{overall_acc*100:.2f}%")
        m2.metric("Macro F1-Score",    f"{macro_f1*100:.2f}%")
        m3.metric("Macro Precision",   f"{macro_prec*100:.2f}%")
        m4.metric("Macro Recall",      f"{macro_rec*100:.2f}%")

        train_col, feat_col = st.columns(2)
        train_col.metric("Training Samples", f"{eval_results['n_train']:,}")
        feat_col.metric("Test Samples",      f"{eval_results['n_test']:,}")

        st.markdown("---")

        # ── Per-class metrics table ──────────────────────────────────────────
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.markdown("#### Per-Class Performance")
            rows = []
            for cls in class_names:
                if cls in report:
                    rows.append({
                        'Class':     cls,
                        'Precision': f"{report[cls]['precision']*100:.1f}%",
                        'Recall':    f"{report[cls]['recall']*100:.1f}%",
                        'F1-Score':  f"{report[cls]['f1-score']*100:.1f}%",
                        'Support':   int(report[cls]['support']),
                    })
            metrics_df = pd.DataFrame(rows)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### F1-Score by Attack Class")
            f1_vals = [report[cls]['f1-score'] for cls in class_names if cls in report]


            # 1. Filter the names to only include what's actually in the report
            available_classes = [c for c in class_names if c in report]

            # 2. Get the values for only those available classes
            f1_vals = [report[c]['f1-score'] for c in available_classes]

            # 3. Update the chart to use the filtered lists
            fig_f1 = px.bar(
                x=available_classes, 
                y=f1_vals,
                labels={'x': 'Attack Class', 'y': 'F1-Score'},
                title="F1-Score per Class",
                text=[f"{v*100:.1f}%" for v in f1_vals]
            )




            fig_f1.update_traces(textposition='outside')
            fig_f1.update_layout(
                showlegend=False, coloraxis_showscale=False,
                height=350, margin=dict(l=0, r=0, t=10, b=80),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_tickangle=-35, yaxis_range=[0, 1.1]
            )
            st.plotly_chart(fig_f1, use_container_width=True)

        with right_col:
            st.markdown("#### Confusion Matrix")
            st.caption("Rows = Actual class, Columns = Predicted class. Diagonal = correct predictions.")

            # Normalise rows to percentages for readability
            cm_norm = cm.astype(float)
            row_sums = cm_norm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # avoid divide by zero
            cm_pct = (cm_norm / row_sums * 100).round(1)

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm_pct,
                x=class_names,
                y=class_names,
                colorscale='Blues',
                text=cm_pct,
                texttemplate="%{text}%",
                textfont={"size": 9},
                hoverongaps=False,
                showscale=True
            ))
            fig_cm.update_layout(
                xaxis_title="Predicted",
                yaxis_title="Actual",
                template='plotly_dark',
                height=500,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_tickangle=-35,
                yaxis_autorange='reversed'
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        # ── Presentation talking points ──────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 💡 Key Talking Points for Presentation")
        col_a, col_b = st.columns(2)
        with col_a:
            st.success(f"""
**Why not just use Accuracy?**

Accuracy of {overall_acc*100:.1f}% sounds great — but with class imbalance
(more BENIGN than attacks), a model that predicts everything as BENIGN would
still score ~60-70% accuracy. That's why we use **F1-Score**, which balances
Precision and Recall.
            """)
        with col_b:
            st.info(f"""
**What the Confusion Matrix shows**

The diagonal shows correct predictions per class.
Off-diagonal cells are misclassifications. A good IDS
wants **high Recall** (catch all attacks) even at the
cost of some false positives — missing an attack is
worse than a false alarm.
            """)
