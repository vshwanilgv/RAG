import streamlit as st
import requests
import json

API = "http://localhost:8000"
PAGE_ICON_URL = "https://img.icons8.com/fluency/48/combo-chart.png"

st.set_page_config(
    page_title="FinSight",
    page_icon=PAGE_ICON_URL,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  .block-container { padding-top: 2rem; }
  .source-card {
    background: #1a1d24;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 12px;
  }
  .page-tag {
    color: #c9a84c;
    font-family: monospace;
    font-size: 11px;
    font-weight: 600;
  }
  .score-tag {
    color: #3db87a;
    font-family: monospace;
    font-size: 11px;
    float: right;
  }
  .preview-text { color: #8a8880; margin-top: 4px; line-height: 1.5; }
  .confidence-high {
    background: rgba(61,184,122,0.1);
    border: 1px solid rgba(61,184,122,0.3);
    color: #3db87a;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-family: monospace;
  }
  .confidence-low {
    background: rgba(201,168,76,0.1);
    border: 1px solid rgba(201,168,76,0.3);
    color: #c9a84c;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-family: monospace;
  }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📊 FinSight")
    st.caption("AI-powered financial document analyst")
    st.divider()

    # Upload
    st.markdown("#### Upload document")
    uploaded = st.file_uploader(
        "Drop a financial PDF",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded:
        if st.button("Index document", use_container_width=True):
            with st.spinner(f"Indexing {uploaded.name}..."):
                res = requests.post(
                    f"{API}/ingest",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                )
                if res.ok:
                    data = res.json()
                    st.success(f"Done — {data['chunks_created']} chunks indexed")
                    st.rerun()
                else:
                    st.error("Ingestion failed. Is the API running?")

    st.divider()

    # Metrics
    st.markdown("#### RAG quality scores")
    try:
        metrics = requests.get(f"{API}/metrics", timeout=2).json()
        ev = metrics["evaluation"]
        sys_info = metrics["system"]

        cols = st.columns(2)
        cols[0].metric("Faithfulness",     f"{ev['faithfulness']:.2f}")
        cols[1].metric("Ans. relevancy",   f"{ev['answer_relevancy']:.2f}")
        cols[0].metric("Ctx. precision",   f"{ev['context_precision']:.2f}")
        cols[1].metric("Ctx. recall",      f"{ev['context_recall']:.2f}")

        st.markdown(
            f"**Overall score:** `{ev['overall_score']:.3f}`",
        )

        st.divider()
        st.markdown("#### Indexed documents")
        for doc in sys_info["documents_indexed"]:
            st.markdown(f"📄 `{doc}`")
        st.caption(f"{sys_info['chunks_indexed']} total chunks · {sys_info['model']}")

    except Exception:
        st.warning("API offline — start uvicorn first")


st.markdown("### Ask anything about your financial documents")
st.caption("Every answer is grounded in the document with exact page citations")
st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if not st.session_state.messages:
    st.markdown("**Try asking:**")
    col1, col2, col3 = st.columns(3)
    suggestions = [
        "What was total revenue in 2024 vs 2023?",
        "How did AWS operating income change?",
        "What were the main growth drivers?",
    ]
    for col, suggestion in zip([col1, col2, col3], suggestions):
        with col:
            if st.button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Render stored sources for assistant messages
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander(f"Sources · {msg['confidence']} confidence", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                      <span class="page-tag">Page {s['page']}</span>
                      <span class="score-tag">score: {s['rerank_score']:.3f}</span>
                      <div class="preview-text">{s['preview'][:140]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

question = st.chat_input("Ask a question about the indexed documents...")

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        full_answer = ""
        sources = []
        confidence = "high"

        try:
            with requests.post(
                f"{API}/chat",
                json={"question": question},
                stream=True,
                timeout=60
            ) as res:
                for raw_line in res.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        msg = json.loads(raw_line)

                        if msg["type"] == "metadata":
                            sources = msg["sources"]
                            confidence = msg["confidence"]

                        elif msg["type"] == "token":
                            full_answer += msg["content"]
                            # Show answer as it streams, with cursor
                            answer_placeholder.markdown(full_answer + "▌")

                        elif msg["type"] == "done":
                            # Final render — clean, no cursor
                            answer_placeholder.markdown(full_answer)

                    except json.JSONDecodeError:
                        continue

        except requests.exceptions.ConnectionError:
            answer_placeholder.error("Cannot reach API. Make sure uvicorn is running on port 8000.")
            st.stop()

        if sources:
            with st.expander(
                f"{'✅' if confidence == 'high' else '⚠️'} Sources · {confidence} confidence",
                expanded=confidence == "high"
            ):
                for s in sources:
                    st.markdown(f"""
                    <div class="source-card">
                      <span class="page-tag">Page {s['page']}</span>
                      <span class="score-tag">score: {s['rerank_score']:.3f}</span>
                      <div class="preview-text">{s['preview'][:140]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "sources": sources,
        "confidence": confidence
    })