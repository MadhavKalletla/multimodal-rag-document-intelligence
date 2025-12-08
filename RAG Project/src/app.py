import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from datetime import datetime

from src.ocr import extract_text_from_image
from src.preprocess import clean_and_chunk
from src.query import run_query, answer_from_context
from src.indexing import index_text   # <-- import here


# ----------------------------------------------------
# Streamlit UI setup
# ----------------------------------------------------
st.set_page_config(page_title="Image-based QA System", page_icon="🧠", layout="wide")
st.title("🧠 Image-based QA System")


uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:

    # Show uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Ask user question
    query = st.text_input("💬 Ask your question:")

    # --- Save temp file for OCR ---
    temp_path = "temp_image.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- OCR extraction ---
    with st.spinner("🔍 Extracting text from image..."):
        text = extract_text_from_image(temp_path)

    # --- Show extracted text ---
    with st.expander("📄 Extracted Text", expanded=False):
        st.write(text if text.strip() else "⚠️ No text detected in image.")

    # ----------------------------------------------------
    # 📥 Save extracted text into Knowledge Base
    # ----------------------------------------------------
    st.markdown("---")
    st.subheader("📥 Save this document to Knowledge Base")

    col1, col2 = st.columns([3,1])
    with col1:
        kb_name = st.text_input(
            "Document name (optional)",
            value=f"uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    with col2:
        index_now = st.button("Save to KB", key="save_kb_btn")

    if index_now:
        if not text or not text.strip():
            st.error("⚠️ No extracted text to index.")
        else:
            with st.spinner("Indexing document..."):
                try:
                    count = index_text(text, source_name=kb_name)
                    st.success(f"✅ Indexed {count} chunk(s) as '{kb_name}'.")
                except Exception as e:
                    st.error(f"❌ Indexing failed: {e}")

    # ----------------------------------------------------
    # QA FEATURES
    # ----------------------------------------------------
    chunks = clean_and_chunk(text, max_chars=800, overlap=100)

    use_image_only = st.toggle("Use only this image’s text (no DB)", value=True)

    if query and len(query.strip()) >= 3:
        if st.button("Get Answer"):
            if use_image_only:
                with st.spinner("🤖 Answering from extracted text..."):
                    answer = answer_from_context(chunks, query)
            else:
                with st.spinner("🔎 Retrieving from vector DB / LLM..."):
                    answer = run_query(query)

            st.subheader("🧩 Answer")
            st.success(answer)
    else:
        st.info("Please enter a question.")

    # Clean temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

else:
    st.info("📤 Upload an image to begin.")
 