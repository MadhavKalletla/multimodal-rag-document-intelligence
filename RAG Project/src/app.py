# src/app.py
"""
Streamlit UI dashboard for the Production-Grade Multimodal RAG system.
Features:
- Dark theme with glassmorphism CSS
- Support for PDF, PNG, JPG, JPEG, WEBP, TXT, DOCX
- Tabs: Query Documents and Analytics Dashboard
- Chat-style Q&A with answer + source chunks highlighted + response time shown in ms
- Sidebar document library with live database sync
"""

import sys
import os
import time
from datetime import datetime

# Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np

from src.indexing import index_file
from src.query import run_query
from src.evaluation import evaluate_response
from src.vector_store import get_chroma_collection, clear_bm25_index

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cognitive RAG Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session State Initialization ──────────────────────────────────────────────
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "query_times" not in st.session_state:
    st.session_state.query_times = []
if "last_query_results" not in st.session_state:
    st.session_state.last_query_results = None
if "indexed_files_cache" not in st.session_state:
    st.session_state.indexed_files_cache = []

# ── Custom CSS for Dark Mode & Premium Feel ────────────────────────────────────
st.markdown("""
<style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main app wrapper background */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #38bdf8 !important;
        font-weight: 600 !important;
    }
    
    /* Card Glassmorphism Styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
        margin-bottom: 15px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 5px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(56, 189, 248, 0.1) !important;
    }
    
    /* Tab active styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0px 0px;
        color: #94a3b8;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.15) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        color: #38bdf8 !important;
    }
    
    /* Highlight Source Chunks */
    .source-chunk-card {
        background: rgba(15, 23, 42, 0.6);
        border-left: 4px solid #38bdf8;
        border-radius: 4px 8px 8px 4px;
        padding: 15px;
        margin-bottom: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.03);
        border-right: 1px solid rgba(255, 255, 255, 0.03);
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
    }
    
    .source-chunk-header {
        font-weight: 600;
        color: #38bdf8;
        margin-bottom: 5px;
        font-size: 0.95rem;
    }
    
    .source-chunk-meta {
        font-size: 0.8rem;
        color: #64748b;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Live DB Document Library Fetcher ──────────────────────────────────────────
def get_db_document_library():
    """Queries ChromaDB directly to fetch document list and chunk counts."""
    try:
        col = get_chroma_collection()
        res = col.get(include=["metadatas"])
        metadatas = res.get("metadatas", [])
        
        doc_library = {}
        for meta in metadatas:
            if meta and "source" in meta:
                src = meta["source"]
                doc_library[src] = doc_library.get(src, 0) + 1
        return doc_library
    except Exception as e:
        print(f"Error fetching document library: {e}")
        return {}

def delete_all_documents():
    """Wipes vector store collection and BM25 index."""
    try:
        col = get_chroma_collection()
        # Delete all records by matching all IDs (or deleting and recreating)
        col.delete(where={})
        clear_bm25_index()
        st.session_state.indexed_files_cache = []
        st.success("Successfully cleared all indexed documents from vector store and BM25.")
    except Exception as e:
        st.error(f"Failed to clear collection: {e}")

# ── SIDEBAR: Document Library & File Uploader ───────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🧠 Cognitive Hub</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>FAANG-Grade RAG Pipeline</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload files for indexing", 
        type=["pdf", "png", "jpg", "jpeg", "webp", "txt", "docx"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        file_name = uploaded_file.name
        st.info(f"Selected: `{file_name}`")
        
        # Save temp file
        temp_dir = os.path.join(os.getcwd(), "temp_uploads")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, file_name)
        
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if st.button("⚡ Index Document", use_container_width=True):
            with st.spinner("Processing & indexing document..."):
                try:
                    num_chunks = index_file(temp_path, source_name=file_name)
                    st.success(f"Indexed `{file_name}` successfully! Added {num_chunks} chunks.")
                    st.session_state.indexed_files_cache.append(file_name)
                except Exception as e:
                    st.error(f"Error indexing document: {e}")
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    st.markdown("---")
    
    # Show active documents in DB
    st.subheader("📄 Document Library")
    doc_library = get_db_document_library()
    
    if doc_library:
        for doc_name, chunks_count in doc_library.items():
            st.markdown(f"📁 **{doc_name}** ({chunks_count} chunks)")
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Library", type="secondary", use_container_width=True):
            delete_all_documents()
            st.rerun()
    else:
        st.caption("No documents in active index. Upload above to start.")

# ── MAIN PANEL: Tabs & Interaction ──────────────────────────────────────────
tab_query, tab_analytics = st.tabs(["💬 Query Documents", "📊 Analytics Dashboard"])

# ── TAB 1: Query Documents ────────────────────────────────────────────────────
with tab_query:
    st.markdown("### Ask the Knowledge Base")
    query_input = st.text_input("💬 Enter your question here:", placeholder="e.g., What are the key performance results in the paper?")
    
    if query_input:
        if st.button("🔍 Run Query", type="primary"):
            with st.spinner("Retrieving facts & synthesizing answer..."):
                # Run query engine
                results = run_query(query_input, top_k=5)
                
                # Fetch query outputs
                answer = results["answer"]
                docs = results["docs"]
                metadatas = results["metadatas"]
                scores = results["scores"]
                response_time = results["response_time_ms"]
                llm_used = results["llm_used"]
                
                # Evaluate metrics
                eval_scores = evaluate_response(query_input, answer, docs)
                
                # Save to session state
                st.session_state.total_queries += 1
                st.session_state.query_times.append(response_time)
                st.session_state.last_query_results = {
                    "query": query_input,
                    "answer": answer,
                    "docs": docs,
                    "metadatas": metadatas,
                    "scores": scores,
                    "response_time_ms": response_time,
                    "llm_used": llm_used,
                    "eval": eval_scores
                }
                
    # Display last query results if exists
    if st.session_state.last_query_results:
        res = st.session_state.last_query_results
        
        st.markdown("---")
        st.markdown(f"#### **Answer** (Synthesized by `{res['llm_used']}` in `{res['response_time_ms']} ms`)")
        st.success(res["answer"])
        
        # Display Evaluation metrics in beautiful columns
        st.markdown("##### **Quality Metrics**")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            faith = res["eval"]["faithfulness"]
            st.metric("Faithfulness Score", f"{faith * 100:.1f}%", help="Percentage of answer words supported by retrieved context.")
        with col_m2:
            relev = res["eval"]["relevance"]
            st.metric("Query-Answer Relevance", f"{relev * 100:.1f}%", help="Cosine similarity between the query and generated answer.")
        with col_m3:
            util = res["eval"]["context_utilization"]
            st.metric("Context Utilization", f"{util * 100:.1f}%", help="Percentage of retrieved chunks utilized in the answer.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Highlighted Source Chunks
        st.markdown("##### **Retrieved Source Chunks (Hybrid RRF Rank)**")
        for idx, (doc_text, meta) in enumerate(zip(res["docs"], res["metadatas"])):
            score_val = res["scores"][idx] if idx < len(res["scores"]) else 0.0
            source_file = meta.get("source", "Unknown file")
            page_num = meta.get("page_num", 0)
            chunk_idx = meta.get("chunk_index", 0)
            
            st.markdown(f"""
            <div class="source-chunk-card">
                <div class="source-chunk-header">[{idx}] Source: {source_file} (Page {page_num}, Chunk {chunk_idx})</div>
                <div class="source-chunk-meta">RRF Rank Score: {score_val:.4f}</div>
                <div style="font-size: 0.95rem; line-height: 1.5; color: #cbd5e1;">{doc_text}</div>
            </div>
            """, unsafe_allow_html=True)

# ── TAB 2: Analytics Dashboard ────────────────────────────────────────────────
with tab_analytics:
    st.markdown("### System Analytics")
    
    # High-level KPIs
    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
    
    total_docs = len(doc_library)
    avg_resp = np.mean(st.session_state.query_times) if st.session_state.query_times else 0.0
    
    with col_kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Documents Indexed</div>
            <div class="metric-value">{total_docs}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Queries Run</div>
            <div class="metric-value">{st.session_state.total_queries}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Response Time</div>
            <div class="metric-value">{avg_resp:.1f} ms</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Detailed retrieval scores chart for last query
    if st.session_state.last_query_results:
        res = st.session_state.last_query_results
        st.markdown(f"#### Retrieval Score Analysis for Query: *\"{res['query']}\"*")
        
        # Build score bar chart
        scores = res["scores"]
        doc_names = [f"[{i}] {res['metadatas'][i].get('source', 'Unknown')}" for i in range(min(len(scores), len(res['metadatas'])))]
        if scores:
            import pandas as pd
            min_len = min(len(scores), len(doc_names))
            chart_data = pd.DataFrame({
                "RRF Fusion Score": scores[:min_len]
            }, index=doc_names[:min_len])
            st.bar_chart(chart_data)
            
            # Show table view
            st.markdown("##### Detailed Candidate Scores")
            st.dataframe(chart_data, use_container_width=True)
        else:
            st.info("No scores available for retrieval.")
    else:
        st.info("Run a query in the first tab to view detailed metrics here.")