"""
NeMo Speech Alignment & Text Normalization – Interactive Demo

Launch with:  streamlit run demo/app.py
"""

import sys
import os
import json
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.text_normalizer import TextNormalizer
from src.audio_preprocessor import AudioPreprocessor
from src.forced_aligner import ForcedAligner
from src.quality_validator import QualityValidator
from src.pipeline import SpeechAlignmentPipeline


# ── Page Config ──────────────────────────────────────────────────

st.set_page_config(
    page_title="NeMo Speech Alignment & Text Normalization",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #76B900, #00A86B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1a1a2e;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #333;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #76B900;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #aaa;
        margin-top: 0.3rem;
    }
    .change-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        margin: 2px;
        font-weight: 600;
    }
    .change-number { background: #1a3a5c; color: #5dade2; }
    .change-currency { background: #1a4a2a; color: #58d68d; }
    .change-date { background: #4a1a3a; color: #d291bc; }
    .change-time { background: #4a3a1a; color: #f0b429; }
    .change-abbreviation { background: #3a1a4a; color: #bb8fce; }
    .change-measure { background: #1a4a4a; color: #48c9b0; }
    .word-box {
        display: inline-block;
        margin: 3px;
        padding: 6px 10px;
        border-radius: 6px;
        font-size: 0.9rem;
        cursor: default;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize Components ────────────────────────────────────────

@st.cache_resource
def get_pipeline():
    return SpeechAlignmentPipeline()

@st.cache_resource
def get_normalizer():
    return TextNormalizer()

@st.cache_resource
def get_preprocessor():
    return AudioPreprocessor()

@st.cache_resource
def get_aligner():
    aligner = ForcedAligner()
    return aligner

@st.cache_resource
def get_validator():
    return QualityValidator()


pipeline = get_pipeline()
normalizer = get_normalizer()
preprocessor = get_preprocessor()
aligner = get_aligner()
validator = get_validator()


# ── Header ───────────────────────────────────────────────────────

st.markdown('<div class="main-header">🎙️ NeMo Speech Alignment & Text Normalization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-grade speech data preparation using NVIDIA NeMo • Text normalization • Forced alignment • Quality validation</div>', unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    st.markdown("**Text Normalization**")
    expand_numbers = st.checkbox("Expand Numbers", value=True)
    expand_currency = st.checkbox("Expand Currency", value=True)
    expand_dates = st.checkbox("Expand Dates", value=True)
    expand_time = st.checkbox("Expand Time", value=True)
    expand_abbreviations = st.checkbox("Expand Abbreviations", value=True)
    expand_measures = st.checkbox("Expand Measures", value=True)

    st.markdown("---")
    st.markdown("**Quality Thresholds**")
    wer_threshold = st.slider("WER Threshold", 0.0, 1.0, 0.30, 0.05)
    min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.60, 0.05)

    st.markdown("---")
    st.markdown("**Audio Settings**")
    sample_rate = st.selectbox("Sample Rate", [8000, 16000, 22050, 44100], index=1)
    audio_duration = st.slider("Demo Audio Duration (s)", 1.0, 10.0, 3.0, 0.5)

# Update normalizer settings
normalizer.expand_numbers = expand_numbers
normalizer.expand_currency = expand_currency
normalizer.expand_dates = expand_dates
normalizer.expand_time = expand_time
normalizer.expand_abbreviations = expand_abbreviations
normalizer.expand_measures = expand_measures


# ── Tabs ─────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📝 Text Normalization",
    "🎯 Forced Alignment",
    "✅ Quality Validation",
    "🔄 Full Pipeline",
])


# ══════════════════════════════════════════════════════════════════
# TAB 1: TEXT NORMALIZATION
# ══════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Text Normalization (TN / ITN)")
    st.markdown("Convert written-form text to spoken-form for ASR dataset preparation.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Input Text (Written Form)**")
        default_text = (
            "Dr. Smith spent $3.50 on 2.5kg of apples at the store on 01/15/2024. "
            "He arrived at 3:30 PM driving 65mph on St. Patrick's Ave. "
            "The total including tax was $142,500.75. "
            "Prof. Johnson from MIT measured 15cm of rainfall in Feb. 2024."
        )
        input_text = st.text_area("Enter text to normalize:", value=default_text, height=180)

    if st.button("🔄 Normalize Text", key="normalize_btn", type="primary"):
        with st.spinner("Normalizing..."):
            result = normalizer.normalize(input_text)

        with col2:
            st.markdown("**Output Text (Spoken Form)**")
            st.text_area("Normalized:", value=result.normalized, height=180, disabled=True)

        # Show changes
        if result.changes:
            st.markdown("---")
            st.markdown(f"### 📊 Changes Made: **{result.num_changes}**")

            changes_by_type = {}
            for change in result.changes:
                t = change["type"]
                if t not in changes_by_type:
                    changes_by_type[t] = []
                changes_by_type[t].append(change)

            cols = st.columns(min(len(changes_by_type), 3))
            for i, (change_type, changes) in enumerate(changes_by_type.items()):
                with cols[i % len(cols)]:
                    st.markdown(f"**{change_type.title()}** ({len(changes)})")
                    for c in changes:
                        st.markdown(
                            f'<span class="change-tag change-{change_type}">'
                            f'"{c["original"]}" → "{c["normalized"]}"</span>',
                            unsafe_allow_html=True,
                        )

            # Pie chart of change types
            fig = px.pie(
                names=list(changes_by_type.keys()),
                values=[len(v) for v in changes_by_type.values()],
                title="Change Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

    # Batch normalization section
    with st.expander("📦 Batch Normalization"):
        batch_input = st.text_area(
            "Enter multiple lines (one per line):",
            value="I have $500\nMeeting at 9:00 AM\nRan 10km in 45 min.\nDr. Lee from Dept. of Physics",
            height=120,
        )
        if st.button("Normalize Batch"):
            lines = [l.strip() for l in batch_input.strip().split("\n") if l.strip()]
            results = normalizer.normalize_batch(lines)
            for r in results:
                st.markdown(f"**{r.original}** → {r.normalized} ({r.num_changes} changes)")


# ══════════════════════════════════════════════════════════════════
# TAB 2: FORCED ALIGNMENT
# ══════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Forced Alignment (CTC-Based)")
    st.markdown("Generate word-level timestamps by aligning transcripts to audio.")

    transcript_input = st.text_input(
        "Transcript:",
        value="the quick brown fox jumps over the lazy dog",
    )

    col_a1, col_a2 = st.columns([2, 1])

    with col_a2:
        freq = st.slider("Demo Tone Frequency (Hz)", 100, 1000, 440)
        noise = st.slider("Background Noise Level", 0.0, 0.1, 0.01, 0.005)

    if st.button("🎯 Run Alignment", key="align_btn", type="primary"):
        with st.spinner("Generating audio and aligning..."):
            # Generate demo audio
            audio, sr = preprocessor.generate_synthetic_audio(
                duration_sec=audio_duration,
                frequency=freq,
                noise_level=noise,
            )

            # Run alignment
            alignment = aligner.align(audio, transcript_input, sr, "demo_audio.wav")

        # Word-level timeline visualization
        st.markdown("### 🕐 Word-Level Alignment")

        if alignment.words:
            # Build timeline chart
            fig = go.Figure()

            colors = px.colors.qualitative.Plotly
            for i, word in enumerate(alignment.words):
                color = colors[i % len(colors)]
                opacity = max(0.3, word.confidence)

                fig.add_trace(go.Bar(
                    x=[word.duration],
                    y=[0],
                    base=[word.start_time],
                    orientation='h',
                    name=word.word,
                    marker_color=color,
                    opacity=opacity,
                    text=f"{word.word}<br>conf: {word.confidence:.2f}",
                    hovertemplate=(
                        f"<b>{word.word}</b><br>"
                        f"Start: {word.start_time:.3f}s<br>"
                        f"End: {word.end_time:.3f}s<br>"
                        f"Duration: {word.duration:.3f}s<br>"
                        f"Confidence: {word.confidence:.2f}<br>"
                        "<extra></extra>"
                    ),
                ))

            fig.update_layout(
                title="Word-Level Timeline",
                xaxis_title="Time (seconds)",
                showlegend=False,
                height=200,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
                yaxis=dict(visible=False),
                bargap=0.1,
            )
            st.plotly_chart(fig, use_container_width=True)

            # Word details table
            st.markdown("### 📋 Word Details")
            word_data = []
            for w in alignment.words:
                conf_emoji = "🟢" if w.confidence >= 0.8 else ("🟡" if w.confidence >= 0.6 else "🔴")
                word_data.append({
                    "Word": w.word,
                    "Start (s)": f"{w.start_time:.4f}",
                    "End (s)": f"{w.end_time:.4f}",
                    "Duration (s)": f"{w.duration:.4f}",
                    "Confidence": f"{conf_emoji} {w.confidence:.4f}",
                })
            st.dataframe(word_data, use_container_width=True)

            # Confidence distribution
            confidences = [w.confidence for w in alignment.words]
            fig_conf = go.Figure(data=[go.Histogram(
                x=confidences,
                nbinsx=20,
                marker_color="#76B900",
                opacity=0.8,
            )])
            fig_conf.update_layout(
                title="Confidence Distribution",
                xaxis_title="Confidence",
                yaxis_title="Count",
                height=250,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
            )
            st.plotly_chart(fig_conf, use_container_width=True)

            # Segment grouping
            if alignment.segments:
                st.markdown("### 📑 Segments")
                for i, seg in enumerate(alignment.segments):
                    st.markdown(
                        f"**Segment {i+1}** | "
                        f"`{seg.start_time:.2f}s – {seg.end_time:.2f}s` | "
                        f"Confidence: {seg.avg_confidence:.2f} | "
                        f"Words: {len(seg.words)}"
                    )
                    st.text(f"  \"{seg.text}\"")


# ══════════════════════════════════════════════════════════════════
# TAB 3: QUALITY VALIDATION
# ══════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Quality Validation & Metrics")

    col_q1, col_q2 = st.columns(2)

    with col_q1:
        ref_text = st.text_input("Reference Transcript:", value="the quick brown fox jumps over the lazy dog")
    with col_q2:
        hyp_text = st.text_input("Hypothesis (ASR Output):", value="the quik brown fox jump over a lazy dog")

    if st.button("✅ Validate Quality", key="validate_btn", type="primary"):
        # Compute metrics
        wer_score = validator._compute_wer(ref_text, hyp_text)
        cer_score = validator._compute_cer(ref_text, hyp_text)

        # Simulate confidences
        n_words = len(hyp_text.split())
        confidences = np.random.uniform(0.5, 0.98, n_words).tolist()

        metrics = validator.validate_alignment(
            reference=ref_text,
            hypothesis=hyp_text,
            word_confidences=confidences,
            duration=audio_duration,
            aligned_duration=audio_duration * 0.85,
            silence_ratio=0.15,
        )

        # Metric cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            wer_color = "#76B900" if wer_score <= wer_threshold else "#e74c3c"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {wer_color}">{wer_score:.2%}</div>
                <div class="metric-label">Word Error Rate</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            cer_color = "#76B900" if cer_score <= 0.15 else "#e74c3c"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {cer_color}">{cer_score:.2%}</div>
                <div class="metric-label">Character Error Rate</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m3:
            conf_color = "#76B900" if metrics.avg_confidence >= min_confidence else "#e74c3c"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {conf_color}">{metrics.avg_confidence:.2f}</div>
                <div class="metric-label">Avg Confidence</div>
            </div>
            """, unsafe_allow_html=True)

        with col_m4:
            status_color = "#76B900" if metrics.is_valid else "#e74c3c"
            status_text = "PASS ✓" if metrics.is_valid else "FAIL ✗"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {status_color}">{status_text}</div>
                <div class="metric-label">Quality Check</div>
            </div>
            """, unsafe_allow_html=True)

        # Flags
        if metrics.flags:
            st.markdown("### ⚠️ Quality Flags")
            for flag in metrics.flags:
                st.warning(f"🚩 {flag}")
        else:
            st.success("✅ All quality checks passed!")

        # Word-by-word comparison
        st.markdown("### 🔍 Word-by-Word Comparison")
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()

        html_parts = []
        max_len = max(len(ref_words), len(hyp_words))
        for i in range(max_len):
            ref_w = ref_words[i] if i < len(ref_words) else "—"
            hyp_w = hyp_words[i] if i < len(hyp_words) else "—"
            if ref_w == hyp_w:
                color = "#76B900"
            else:
                color = "#e74c3c"
            html_parts.append(
                f'<span class="word-box" style="background: {color}22; border: 1px solid {color};">'
                f'<b>{hyp_w}</b><br><small style="color:#888">{ref_w}</small></span>'
            )

        st.markdown(" ".join(html_parts), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# TAB 4: FULL PIPELINE
# ══════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### Full Pipeline Demo")
    st.markdown("Run the complete pipeline: Normalize → Align → Validate")

    pipeline_text = st.text_area(
        "Raw Transcript:",
        value=(
            "Dr. Smith reported that on 01/15/2024 at 3:30 PM, "
            "the patient weighed 75kg and had a temperature of 98.6 degrees. "
            "The total bill was $1,250.50 for the procedure at St. Mary's Hospital."
        ),
        height=100,
    )

    if st.button("🚀 Run Full Pipeline", key="pipeline_btn", type="primary"):
        progress = st.progress(0, text="Initializing...")

        # Step 1: Generate demo audio
        progress.progress(10, text="Generating synthetic audio...")
        audio, sr = preprocessor.generate_synthetic_audio(
            duration_sec=audio_duration,
            frequency=440,
            noise_level=0.01,
        )

        # Step 2: Run pipeline
        progress.progress(30, text="Normalizing text...")
        time.sleep(0.3)

        progress.progress(50, text="Running forced alignment...")
        time.sleep(0.3)

        progress.progress(70, text="Validating quality...")
        alignment, norm_result, metrics = pipeline.process_from_audio(
            audio, sr, pipeline_text
        )

        progress.progress(90, text="Generating report...")
        time.sleep(0.2)

        progress.progress(100, text="Complete!")
        time.sleep(0.3)
        progress.empty()

        st.success("Pipeline completed successfully!")

        # Results
        st.markdown("---")

        # Step 1 Result: Normalization
        st.markdown("### 1️⃣ Text Normalization")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("**Original:**")
            st.info(norm_result.original)
        with col_p2:
            st.markdown("**Normalized:**")
            st.success(norm_result.normalized)
        st.markdown(f"*{norm_result.num_changes} changes applied*")

        st.markdown("---")

        # Step 2 Result: Alignment
        st.markdown("### 2️⃣ Forced Alignment")

        if alignment.words:
            fig = go.Figure()
            for i, word in enumerate(alignment.words):
                conf_color = f"rgba(118,185,0,{max(0.3, word.confidence)})"
                fig.add_trace(go.Bar(
                    x=[word.duration],
                    y=[""],
                    base=[word.start_time],
                    orientation='h',
                    marker_color=conf_color,
                    text=word.word,
                    textposition="inside",
                    hovertemplate=f"<b>{word.word}</b> | {word.start_time:.3f}s–{word.end_time:.3f}s | conf={word.confidence:.2f}<extra></extra>",
                    showlegend=False,
                ))

            fig.update_layout(
                xaxis_title="Time (s)",
                height=120,
                margin=dict(l=20, r=20, t=10, b=40),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ccc",
                yaxis=dict(visible=False),
                barmode="overlay",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown(f"**{len(alignment.words)} words** aligned across **{len(alignment.segments)} segments** | "
                        f"Total duration: **{alignment.total_duration:.2f}s** | "
                        f"Avg confidence: **{alignment.avg_confidence:.2f}**")

        st.markdown("---")

        # Step 3 Result: Quality
        st.markdown("### 3️⃣ Quality Validation")

        col_r1, col_r2, col_r3, col_r4, col_r5 = st.columns(5)
        metric_items = [
            (col_r1, "WER", f"{metrics.wer:.2%}", metrics.wer <= wer_threshold),
            (col_r2, "CER", f"{metrics.cer:.2%}", metrics.cer <= 0.15),
            (col_r3, "Confidence", f"{metrics.avg_confidence:.2f}", metrics.avg_confidence >= min_confidence),
            (col_r4, "Duration", f"{metrics.duration:.1f}s", True),
            (col_r5, "Status", "PASS ✓" if metrics.is_valid else "FAIL ✗", metrics.is_valid),
        ]

        for col, label, value, is_good in metric_items:
            color = "#76B900" if is_good else "#e74c3c"
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {color}; font-size: 1.5rem">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

        if metrics.flags:
            st.markdown("**Flags:**")
            for flag in metrics.flags:
                st.warning(f"🚩 {flag}")

        st.markdown("---")

        # Manifest preview
        st.markdown("### 📄 Generated Manifest (NeMo Format)")
        manifest = alignment.to_manifest_entry()
        st.json(manifest)


# ── Footer ───────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666; font-size: 0.8rem;">'
    'NeMo Speech Alignment & Text Normalization Pipeline • '
    'Built with NVIDIA NeMo, Conformer-CTC, Python, Librosa • '
    'Jay Guwalani'
    '</div>',
    unsafe_allow_html=True,
)
