import os
import time
import queue
import base64

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
st.set_page_config(page_title="SlidePilot", layout="wide")

# ───────────────── CUSTOM CSS ─────────────────
st.markdown("""
<style>
    .mode-badge {
        display: inline-block;
        font-size: 0.82rem;
        font-weight: 700;
        border-radius: 20px;
        padding: 3px 14px;
        letter-spacing: 0.5px;
        margin: 6px auto;
        text-align: center;
    }
    .mode-nav  { background: rgba(102,126,234,0.18); border: 1px solid #667eea; color: #667eea; }
    .mode-zoom { background: rgba(50,200,100,0.18);  border: 1px solid #32c864; color: #32c864; }
    .finger-badge {
        display: inline-block;
        color: white;
        font-size: 1.4rem;
        font-weight: bold;
        border-radius: 50%;
        width: 52px; height: 52px;
        line-height: 52px;
        text-align: center;
        margin: 0 auto;
        box-shadow: 0 2px 8px rgba(102,126,234,0.4);
    }
    .finger-label {
        text-align: center;
        font-size: 0.78rem;
        color: #aaa;
        margin-top: 4px;
    }
    .zoom-pill {
        display: inline-block;
        background: rgba(50,200,100,0.15);
        border: 1px solid rgba(50,200,100,0.4);
        color: #32c864;
        font-size: 0.78rem;
        font-weight: 600;
        border-radius: 20px;
        padding: 2px 10px;
        margin-left: 8px;
        vertical-align: middle;
    }
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
</style>
""", unsafe_allow_html=True)

# ───────────────── HEADING (matches reference image) ─────────────────
st.markdown("""
<div style="
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 18px 0 10px 0;
">
    <div style="
        font-family: 'Georgia', serif;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1;
    ">
        <span style="color: #ffffff;">Slide</span><span style="color: #5b7fe8;">Pilot</span>
    </div>
    <div style="
        color: #6b7280;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        padding-top: 4px;
    ">
        Gesture &nbsp;&middot;&nbsp; Zoom &nbsp;&middot;&nbsp; AI Chat
    </div>
</div>
""", unsafe_allow_html=True)
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
    "zoom_level": 1.0,
    "zoom_step": 0.25,
    "zoom_min": 0.5,
    "zoom_max": 3.0,
    "gesture_mode": "NAV",   # "NAV" or "ZOOM"
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource
def get_gesture_queue():
    return queue.Queue(maxsize=1)

@st.cache_resource
def get_shared_state():
    """Shared mutable dict: processor thread → Streamlit UI thread."""
    return {
        "finger_count":  None,
        "gesture_mode":  "NAV",   # processor mirrors current mode here
        "fist_hold_pct": 0.0,     # 0.0-1.0 for the hold progress bar
    }

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

# ───────────────── PDF HELPERS ─────────────────
@st.cache_data
def convert_pdf_to_images(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = [p.get_pixmap(dpi=150).tobytes("png") for p in doc]
    doc.close()
    return pages

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

@st.cache_resource
def build_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)

# ───────────────── GESTURE PROCESSOR ─────────────────
#
#  MODE SYSTEM  (solves the thumb-misdetection problem)
#  ────────────────────────────────────────────────────
#  Only finger counts 0, 1, and 4+ are ever used.
#  2 and 3 are a deliberate dead zone, so thumb false-positives
#  (e.g. 1 finger briefly read as 2) have zero effect.
#
#  ┌───────────┬──────────────────┬───────────────────┐
#  │ Fingers   │ NAV mode         │ ZOOM mode          │
#  ├───────────┼──────────────────┼───────────────────┤
#  │ 0 – fist  │ hold 2s → enter ZOOM                  │
#  │           │ hold 2s → exit  ZOOM                   │
#  │ 1         │ ⬅️ Prev slide    │ 🔍 Zoom In         │
#  │ 2 – 3     │ (dead zone – no action)                │
#  │ 4+        │ ➡️ Next slide    │ 🔎 Zoom Out        │
#  └───────────┴──────────────────┴───────────────────┘

_FIST_HOLD_SECONDS  = 2.0
_NAV_COOLDOWN       = 2.0
_ZOOM_COOLDOWN      = 1.2
_MODE_SWITCH_BREAK  = 2.5   # seconds to ignore all actions after a mode toggle


class GestureProcessor(VideoProcessorBase):

    def __init__(self):
        self.detector         = HandDetector(maxHands=1, detectionCon=0.5)
        self.mode             = "NAV"
        self.last_time        = 0.0
        self.fist_start       = None
        self.mode_switch_time = 0.0   # timestamp of last mode toggle

    @staticmethod
    def _put(img, text, pos, scale=0.60, color=(200, 200, 200), thickness=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _hbar(img, pct, y=226, color=(102, 126, 234)):
        cv2.rectangle(img, (10, y), (210, y + 10), (50, 50, 50), -1)
        if pct > 0:
            cv2.rectangle(img, (10, y), (10 + int(200 * pct), y + 10), color, -1)

    def recv(self, frame):
        img   = frame.to_ndarray(format="bgr24")
        small = cv2.flip(cv2.resize(img, (320, 240)), 1)

        hands, small = self.detector.findHands(small, draw=True)
        shared       = get_shared_state()

        # Sync mode from UI thread (button press may have changed it)
        self.mode    = shared["gesture_mode"]
        is_zoom_mode = self.mode == "ZOOM"
        mode_color   = (50, 200, 100) if is_zoom_mode else (102, 126, 234)
        now          = time.time()

        if hands:
            fingers = self.detector.fingersUp(hands[0])
            total   = sum(fingers)
            shared["finger_count"] = total

            # ── FIST: mode-toggle hold ──────────────────────────────
            if total == 0:
                if self.fist_start is None:
                    self.fist_start = now

                pct = min((now - self.fist_start) / _FIST_HOLD_SECONDS, 1.0)
                shared["fist_hold_pct"] = pct

                self._put(small, f"Fist  {int(pct*100)}%  hold to switch mode",
                          (10, 30), color=(255, 200, 50))
                self._hbar(small, pct, color=(255, 200, 50))

                if pct >= 1.0:
                    new_mode = "ZOOM" if self.mode == "NAV" else "NAV"
                    self.mode              = new_mode
                    shared["gesture_mode"] = new_mode
                    self.mode_switch_time  = now   # start the break window
                    try:
                        get_gesture_queue().put_nowait(f"MODE_{new_mode}")
                    except queue.Full:
                        pass
                    self.fist_start = None   # reset to avoid repeated toggles
                    self.last_time  = now

            # ── OPEN HAND: nav / zoom actions ──────────────────────
            else:
                self.fist_start         = None
                shared["fist_hold_pct"] = 0.0

                action = None
                if total <= 1:
                    action = "ZOOM_IN"  if is_zoom_mode else "LEFT"
                elif total >= 4:
                    action = "ZOOM_OUT" if is_zoom_mode else "RIGHT"
                # 2 or 3 fingers → dead zone, action stays None

                if action:
                    cooldown = _ZOOM_COOLDOWN if action.startswith("ZOOM") else _NAV_COOLDOWN
                    in_break = (now - self.mode_switch_time) < _MODE_SWITCH_BREAK
                    if not in_break and now - self.last_time > cooldown:
                        try:
                            get_gesture_queue().put_nowait(action)
                        except queue.Full:
                            pass
                        self.last_time = now

                label_map = {
                    "LEFT":     "← PREV",
                    "RIGHT":    "NEXT →",
                    "ZOOM_IN":  "+ ZOOM IN",
                    "ZOOM_OUT": "- ZOOM OUT",
                }
                label = label_map.get(action or "", "dead zone")

                # Show break countdown if we're still in the post-switch window
                remaining = _MODE_SWITCH_BREAK - (now - self.mode_switch_time)
                if remaining > 0:
                    self._put(small, f"Mode switched — ready in {remaining:.1f}s",
                              (10, 30), color=(255, 200, 50))
                    self._hbar(small, remaining / _MODE_SWITCH_BREAK, color=(255, 200, 50))
                else:
                    self._put(small, f"{total} finger{'s' if total!=1 else ''}   {label}",
                              (10, 30), color=mode_color)

        else:
            shared["finger_count"]  = None
            shared["fist_hold_pct"] = 0.0
            self.fist_start         = None
            self._put(small, "No hand detected", (10, 30), color=(160, 160, 160))

        # ── Mode banner at bottom ───────────────────────────────────
        mode_text = "ZOOM MODE" if is_zoom_mode else "NAV MODE"
        cv2.rectangle(small, (0, 210), (320, 240), (20, 20, 20), -1)
        self._put(small, f"[ {mode_text} ]  ✊ fist 2s = switch",
                  (6, 228), scale=0.45, color=mode_color, thickness=1)

        return av.VideoFrame.from_ndarray(small, format="bgr24")


# ───────────────── SIDEBAR ─────────────────
with st.sidebar:
    st.markdown("### ⚙️ Setup")

    uploaded_pdfs = st.file_uploader(
        "Upload PDF files", accept_multiple_files=True, type=["pdf"]
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
                st.session_state.pages        = pages
                st.session_state.current_page = 0
                st.session_state.zoom_level   = 1.0
                st.success(f"✅ {len(pages)} slides loaded")

    st.divider()

    # ── Current mode indicator ─────────────────────────────────────
    st.markdown("### 🎥 Gesture Control")

    # Sync from the processor's shared state BEFORE rendering the guide.
    # The queue is only drained later in the sidebar, so without this sync
    # the guide would always show the previous mode for one full cycle.
    _shared_now = get_shared_state()
    if _shared_now["gesture_mode"] != st.session_state.gesture_mode:
        st.session_state.gesture_mode = _shared_now["gesture_mode"]

    is_zoom = st.session_state.gesture_mode == "ZOOM"
    mode_cls  = "mode-zoom" if is_zoom else "mode-nav"
    mode_icon = "🔍 ZOOM MODE" if is_zoom else "🧭 NAV MODE"
    st.markdown(
        f'<div class="mode-badge {mode_cls}" style="width:100%">{mode_icon}</div>',
        unsafe_allow_html=True
    )

    # Reference table (changes with mode)
    if not is_zoom:
        st.markdown("""
| Gesture | Action |
|---------|--------|
| ✊ Fist 2s | ➜ Switch to Zoom |
| ☝️ 1 finger | ⬅️ Previous |
| ✌️ 2 or 3 | _(dead zone)_ |
| 🖐️ 4+ fingers | ➡️ Next |
""")
    else:
        st.markdown("""
| Gesture | Action |
|---------|--------|
| ✊ Fist 2s | ➜ Switch to Nav |
| ☝️ 1 finger | 🔍 Zoom In |
| ✌️ 2 or 3 | _(dead zone)_ |
| 🖐️ 4+ fingers | 🔎 Zoom Out |
""")

    # Manual mode toggle (keyboard/mouse fallback)
    toggle_label = "🔁 Switch to Zoom mode" if not is_zoom else "🔁 Switch to Nav mode"
    if st.button(toggle_label, use_container_width=True):
        new = "ZOOM" if not is_zoom else "NAV"
        st.session_state.gesture_mode      = new
        get_shared_state()["gesture_mode"] = new
        st.rerun()

    st.divider()

    # ── WebRTC streamer ────────────────────────────────────────────
    ctx = webrtc_streamer(
        key="gesture",
        video_processor_factory=GestureProcessor,
        async_processing=True,
        media_stream_constraints={"video": {"width": 320, "height": 240}, "audio": False},
    )

    # ── Live feedback ──────────────────────────────────────────────
    shared = get_shared_state()
    if ctx and ctx.state.playing:
        count = shared.get("finger_count")
        if count is not None:
            badge_bg = "#32c864" if is_zoom else "#667eea"
            st.markdown(
                f'<div style="display:flex;justify-content:center;">'
                f'  <div class="finger-badge" style="background:{badge_bg};">{count}</div>'
                f'</div>'
                f'<div class="finger-label">finger{"s" if count != 1 else ""} detected</div>',
                unsafe_allow_html=True,
            )
            pct = shared.get("fist_hold_pct", 0.0)
            if pct > 0:
                st.progress(pct, text="Hold fist to toggle mode…")
        else:
            st.markdown(
                '<div class="finger-label" style="text-align:center;margin-top:8px;">'
                '🖐️ Show your hand...</div>',
                unsafe_allow_html=True,
            )

    # ── Manual zoom controls ───────────────────────────────────────
    st.divider()
    st.markdown("### 🔍 Zoom (manual)")
    zc1, zc2, zc3 = st.columns([1, 2, 1])
    with zc1:
        if st.button("➖", use_container_width=True):
            st.session_state.zoom_level = max(
                st.session_state.zoom_min,
                round(st.session_state.zoom_level - st.session_state.zoom_step, 2)
            )
            st.rerun()
    with zc2:
        st.markdown(
            f'<div style="text-align:center;padding-top:6px;color:#32c864;font-weight:600;">'
            f'{st.session_state.zoom_level:.2f}×</div>',
            unsafe_allow_html=True
        )
    with zc3:
        if st.button("➕", use_container_width=True):
            st.session_state.zoom_level = min(
                st.session_state.zoom_max,
                round(st.session_state.zoom_level + st.session_state.zoom_step, 2)
            )
            st.rerun()
    if st.button("↺ Reset Zoom", use_container_width=True):
        st.session_state.zoom_level = 1.0
        st.rerun()

    # ── Process gesture queue ──────────────────────────────────────
    try:
        action = get_gesture_queue().get_nowait()
    except queue.Empty:
        action = None

    if action:
        if action == "MODE_ZOOM":
            st.session_state.gesture_mode = "ZOOM"
        elif action == "MODE_NAV":
            st.session_state.gesture_mode = "NAV"
        elif st.session_state.pages:
            total_pages = len(st.session_state.pages)
            if action == "LEFT":
                st.session_state.current_page = max(0, st.session_state.current_page - 1)
            elif action == "RIGHT":
                st.session_state.current_page = min(total_pages - 1, st.session_state.current_page + 1)
            elif action == "ZOOM_IN":
                st.session_state.zoom_level = min(
                    st.session_state.zoom_max,
                    round(st.session_state.zoom_level + st.session_state.zoom_step, 2)
                )
            elif action == "ZOOM_OUT":
                st.session_state.zoom_level = max(
                    st.session_state.zoom_min,
                    round(st.session_state.zoom_level - st.session_state.zoom_step, 2)
                )
        st.rerun()


# ───────────────── MAIN LAYOUT ─────────────────
left_col, right_col = st.columns([7, 3])

with left_col:
    if st.session_state.pages:
        total = len(st.session_state.pages)
        idx   = st.session_state.current_page
        zoom  = st.session_state.zoom_level

        st.markdown(
            f'<div style="margin-bottom:8px;">'
            f'<span style="font-size:1.1rem;font-weight:600;">📄 Slide {idx+1} / {total}</span>'
            f'<span class="zoom-pill">🔍 {zoom:.2f}×</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        img_b64      = base64.b64encode(st.session_state.pages[idx]).decode()
        overflow_css = "hidden" if zoom > 1.0 else "visible"

        st.markdown(
            f'''<div style="overflow:{overflow_css}; min-height:300px; max-height:680px;
                            display:flex; justify-content:center; align-items:center;
                            border-radius:8px;">
                  <img src="data:image/png;base64,{img_b64}"
                       style="transform: scale({zoom});
                              transform-origin: center center;
                              transition: transform 0.35s cubic-bezier(0.25,0.46,0.45,0.94);
                              width:100%; height:auto;" />
                </div>''',
            unsafe_allow_html=True,
        )

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
        chat_container = st.container(height=520)

        with chat_container:
            if not st.session_state.chat_history:
                st.caption("💬 No messages yet. Ask something about your document!")
            else:
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

        prompt = st.chat_input("Ask about the document...", key="chat_input")

        if prompt:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)

            with chat_container:
                with st.chat_message("assistant"):
                    if is_small_talk(prompt):
                        reply = "Hello! 👋 I'm here to help you with the document. Feel free to ask me anything about it!"
                        st.markdown(reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
                    else:
                        try:
                            docs    = st.session_state.vector_store.similarity_search(prompt, k=3)
                            context = "\n\n".join(d.page_content for d in docs)
                            history_msgs = st.session_state.chat_history[-6:]
                            history = "\n".join(
                                f"{m['role'].capitalize()}: {m['content']}" for m in history_msgs
                            )

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

                            full_response = st.write_stream(
                                chunk.content for chunk in chain.stream({
                                    "context": context,
                                    "question": prompt,
                                    "history": history,
                                })
                            )
                            st.session_state.chat_history.append({
                                "role": "assistant", "content": full_response
                            })

                        except Exception as e:
                            err_msg = f"⚠️ Error: {str(e)}"
                            st.error(err_msg)
                            st.session_state.chat_history.append({
                                "role": "assistant", "content": err_msg
                            })
    else:
        st.info("⬅️ Process a PDF first to enable the assistant.")
