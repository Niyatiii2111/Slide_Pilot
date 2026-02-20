import os
import time
import queue

import av
import cv2
import fitz
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
from streamlit_autorefresh import st_autorefresh

# ───────────────── PAGE CONFIG ─────────────────
st.set_page_config(page_title="PDF Presentation Assistant", layout="wide")

# ───────────────── CUSTOM CSS ─────────────────
st.markdown("""
<style>
    /* ── Centered title with custom font ── */
    .main-title {
        text-align: center;
        font-family: 'Georgia', serif;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
        letter-spacing: 1px;
    }
    .main-subtitle {
        text-align: center;
        color: #888;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        font-family: 'Georgia', serif;
        font-style: italic;
    }

    /* ── Chat container: fixed height, scrollable ── */
    .chat-scroll-wrapper {
        height: 420px;
        overflow-y: auto;
        padding: 10px 6px;
        border: 1px solid #2e2e2e;
        border-radius: 10px;
        background: #111;
        margin-bottom: 10px;
    }

    /* ── Finger counter badge ── */
    .finger-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-size: 1.4rem;
        font-weight: bold;
        border-radius: 50%;
        width: 52px;
        height: 52px;
        line-height: 52px;
        text-align: center;
        margin: 0 auto;
        box-shadow: 0 2px 8px rgba(102,126,234,0.5);
    }
    .finger-label {
        text-align: center;
        font-size: 0.78rem;
        color: #aaa;
        margin-top: 4px;
    }

    /* ── Clear chat button styling ── */
    .stButton > button[kind="secondary"] {
        background: transparent;
        border: 1px solid #e74c3c;
        color: #e74c3c;
        border-radius: 8px;
        font-size: 0.82rem;
        padding: 4px 14px;
        transition: all 0.2s;
    }
    .stButton > button[kind="secondary"]:hover {
        background: #e74c3c;
        color: white;
    }

    /* ── Fixed chat input stays at top of right col ── */
    .chat-input-area {
        position: sticky;
        top: 0;
        z-index: 99;
        background: #0e1117;
        padding-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────── CENTERED TITLE ─────────────────
st.markdown('<div class="main-title">📚 PDF Presentation Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Upload your PDF · Navigate with gestures · Chat with AI</div>', unsafe_allow_html=True)

# ───────────────── ENV API KEY ─────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment variables.")
    st.stop()

# ───────────────── SESSION STATE ─────────────────
_DEFAULTS = {
    "pages": [],
    "current_page": 0,
    "vector_store": None,
    "chain_cache": {},
    "chat_history": [],
    "finger_count": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource
def get_gesture_queue():
    return queue.Queue(maxsize=1)

@st.cache_resource
def get_finger_count_store():
    """Shared mutable dict to pass finger count from processor to UI."""
    return {"count": None}

if st.session_state.pages and not st.session_state.get("chat_processing", False):
    st_autorefresh(interval=3500, key="gesture_refresh")

# ───────────────── SMALL-TALK DETECTOR ─────────────────
SMALL_TALK = {
    "hi", "hello", "hey", "hiii", "helo", "hiiii", "heyy",
    "how are you", "how are you?", "what's up", "sup",
    "good morning", "good evening", "good afternoon",
}
def is_small_talk(text: str) -> bool:
    return text.strip().lower() in SMALL_TALK

# ───────────────── PDF → IMAGES ─────────────────
@st.cache_data
def convert_pdf_to_images(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [p.get_pixmap(dpi=150).tobytes("png") for p in doc]
    doc.close()
    return pages

# ───────────────── TEXT EXTRACTION ─────────────────
def extract_text(pdf_files):
    parts = []
    for pdf in pdf_files:
        pdf.seek(0)
        reader = PdfReader(pdf)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                parts.append(t)
    return "\n".join(parts)

# ───────────────── VECTOR STORE ─────────────────
@st.cache_resource
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

# ───────────────── GROQ CHAIN ─────────────────
def get_chain():
    if "groq" not in st.session_state.chain_cache:
        template = """You are a helpful assistant for a PDF document.
Use the context below to answer the user's question accurately.
If the question is unrelated to the document, politely say you can only answer questions about the document.
Do not make up information that is not in the context.

Context:
{context}

Conversation history:
{history}

Question: {question}

Answer:"""
        prompt = PromptTemplate.from_template(template)
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.5,
        )
        st.session_state.chain_cache["groq"] = prompt | llm
    return st.session_state.chain_cache["groq"]

# ───────────────── GESTURE PROCESSOR ─────────────────
class GestureProcessor(VideoProcessorBase):

    def __init__(self):
        self.detector = HandDetector(maxHands=1, detectionCon=0.5)
        self.last_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        small = cv2.resize(img, (320, 240))
        small = cv2.flip(small, 1)

        hands, small = self.detector.findHands(small, draw=True)
        finger_store = get_finger_count_store()

        if hands:
            fingers = self.detector.fingersUp(hands[0])
            total = sum(fingers)
            finger_store["count"] = total

            now = time.time()
            if now - self.last_time > 2:
                if total <= 1:
                    try:
                        get_gesture_queue().put_nowait("LEFT")
                    except queue.Full:
                        pass
                    self.last_time = now
                elif total >= 4:
                    try:
                        get_gesture_queue().put_nowait("RIGHT")
                    except queue.Full:
                        pass
                    self.last_time = now

            # Draw finger count on frame
            cv2.putText(
                small,
                f"Fingers: {total}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (102, 126, 234),
                2,
                cv2.LINE_AA,
            )
        else:
            finger_store["count"] = None
            cv2.putText(
                small,
                "No hand detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 180),
                2,
                cv2.LINE_AA,
            )

        return av.VideoFrame.from_ndarray(small, format="bgr24")

# ───────────────── SIDEBAR ─────────────────
with st.sidebar:
    st.markdown("### ⚙️ Setup")

    uploaded_pdfs = st.file_uploader(
        "Upload PDF files",
        accept_multiple_files=True,
        type=["pdf"],
    )

    if st.button("🚀 Process PDFs", use_container_width=True):
        if uploaded_pdfs:
            with st.spinner("Processing PDFs..."):
                text = extract_text(uploaded_pdfs)
                st.session_state.vector_store = build_vector_store(text)
                pages = []
                for pdf in uploaded_pdfs:
                    pdf.seek(0)
                    pages.extend(convert_pdf_to_images(pdf.read()))
                st.session_state.pages = pages
                st.session_state.current_page = 0
                st.success(f"✅ {len(pages)} slides loaded")

    st.divider()
    st.markdown("### 🎥 Gesture Control")
    st.caption("☝️ 1 finger = Previous  |  🖐️ 4+ fingers = Next")

    ctx = webrtc_streamer(
        key="gesture",
        video_processor_factory=GestureProcessor,
        async_processing=True,
        media_stream_constraints={
            "video": {"width": 320, "height": 240},
            "audio": False,
        },
    )

    # ── Finger count display ──
    finger_store = get_finger_count_store()
    count = finger_store.get("count")
    if ctx and ctx.state.playing:
        if count is not None:
            st.markdown(
                f'<div class="finger-badge">{count}</div>'
                f'<div class="finger-label">finger{"s" if count != 1 else ""} detected</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="finger-label" style="text-align:center;margin-top:8px;">🖐️ Show your hand...</div>',
                unsafe_allow_html=True,
            )

    # ── Process gesture queue ──
    try:
        action = get_gesture_queue().get_nowait()
    except queue.Empty:
        action = None

    if action and st.session_state.pages:
        total = len(st.session_state.pages)
        if action == "LEFT":
            st.session_state.current_page = max(0, st.session_state.current_page - 1)
        elif action == "RIGHT":
            st.session_state.current_page = min(total - 1, st.session_state.current_page + 1)
        st.rerun()

# ───────────────── MAIN LAYOUT ─────────────────
left_col, right_col = st.columns([7, 3])

with left_col:
    if st.session_state.pages:
        total = len(st.session_state.pages)
        idx = st.session_state.current_page
        st.subheader(f"📄 Slide {idx+1} / {total}")
        st.image(st.session_state.pages[idx], use_container_width=True)

        # Navigation buttons
        b1, b2, b3 = st.columns([1, 2, 1])
        with b1:
            if st.button("⬅️ Prev", use_container_width=True):
                st.session_state.current_page = max(0, idx - 1)
                st.rerun()
        with b2:
            st.markdown(
                f"<div style='text-align:center;color:#888;padding-top:8px;'>{idx+1} / {total}</div>",
                unsafe_allow_html=True,
            )
        with b3:
            if st.button("Next ➡️", use_container_width=True):
                st.session_state.current_page = min(total - 1, idx + 1)
                st.rerun()
    else:
        st.info("⬆️ Upload and process a PDF to get started.")
# ───────────────── RIGHT COLUMN: CHAT ─────────────────
with right_col:
    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.subheader("🤖 Groq Assistant")
    with hcol2:
        if st.button("🗑️ Clear", type="secondary", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    if st.session_state.vector_store:

        # ── Scrollable chat container ──
        chat_container = st.container(height=520)

        # ── Render existing history FIRST ──
        with chat_container:
            if not st.session_state.chat_history:
                st.caption("💬 No messages yet. Ask something about your document!")
            else:
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        # ── Chat input BELOW history ──
        prompt = st.chat_input("Ask about the document...", key="chat_input")

        # ── Process new message ──
        if prompt:
            # 1. Immediately show user message in container
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            # 2. Generate and stream assistant reply in same container
            with chat_container:
                with st.chat_message("assistant"):
                    if is_small_talk(prompt):
                        reply = "Hello! 👋 I'm here to help you with the document. Feel free to ask me anything about it!"
                        st.markdown(reply)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": reply
                        })
                    else:
                        try:
                            docs = st.session_state.vector_store.similarity_search(prompt, k=3)
                            context = "\n\n".join(d.page_content for d in docs)

                            # Only last 6 messages for history, formatted cleanly
                            history_msgs = st.session_state.chat_history[-6:]
                            history = "\n".join(
                                f"{m['role'].capitalize()}: {m['content']}"
                                for m in history_msgs
                            )

                            # Always build a fresh chain — avoids stale cache bugs
                            template = """You are a helpful assistant for a PDF document.
Use the context below to answer the user's question accurately.
If the question is unrelated to the document, politely say so.
Do not make up information not in the context.

Context:
{context}

Conversation history:
{history}

Question: {question}
Answer:"""
                            prompt_template = PromptTemplate.from_template(template)
                            llm = ChatGroq(
                                groq_api_key=GROQ_API_KEY,
                                model_name="llama-3.1-8b-instant",
                                temperature=0.5,
                            )
                            chain = prompt_template | llm

                            # Stream response — no st.rerun() needed
                            full_response = st.write_stream(
                                chunk.content
                                for chunk in chain.stream({
                                    "context": context,
                                    "question": prompt,
                                    "history": history,
                                })
                            )
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": full_response
                            })

                        except Exception as e:
                            err_msg = f"⚠️ Error: {str(e)}"
                            st.error(err_msg)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": err_msg
                            })
    else:
        st.info("⬅️ Process a PDF first to enable the assistant.")