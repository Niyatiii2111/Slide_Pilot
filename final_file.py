import os
import time
import queue
import base64
from collections import deque
from statistics import mode as stat_mode

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
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, RTCConfiguration

# ── ICE servers: STUN + TURN relay fallback.
#
#    WHY THIS MATTERS:
#    • STUN-only works initially but breaks after a few minutes on
#      symmetric NAT (common in home/office routers) because the router
#      rotates its UDP port mapping.
#    • A TURN relay tunnels media through a server when P2P fails,
#      keeping the connection alive indefinitely.
#    • openrelay.metered.ca is a free public TURN service.
#      For production swap in your own coturn / Twilio / Metered creds.
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
})

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
    .fullview-pill {
        display: inline-block;
        background: rgba(255,165,0,0.15);
        border: 1px solid rgba(255,165,0,0.5);
        color: #ffa500;
        font-size: 0.75rem;
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
    .stButton > button[kind="secondary"]:hover { background: #e74c3c; color: white; }
    .sticky-header {
        position: sticky; top: 0; z-index: 999;
        background-color: var(--background-color, #0e1117);
        padding-bottom: 10px;
    }
    body.present-active [data-testid="stSidebar"],
    body.present-active [data-testid="stHeader"],
    body.present-active [data-testid="stToolbar"],
    body.present-active [data-testid="stDecoration"],
    body.present-active footer { display: none !important; }
    body.present-active [data-testid="stAppViewContainer"] { background: #000 !important; }
    body.present-active [data-testid="stMainBlockContainer"],
    body.present-active [data-testid="stMain"] { padding: 0 !important; max-width: 100vw !important; }
    .present-slide-wrap {
        width: 100%; height: calc(100vh - 56px);
        display: flex; align-items: center; justify-content: center;
        background: #000; overflow: hidden;
    }
    .present-slide-wrap img { max-width: 100%; max-height: 100%; object-fit: contain; display: block; }
    .present-nav-bar {
        position: sticky; bottom: 0; z-index: 9999; width: 100%; height: 56px;
        display: flex; align-items: center; justify-content: center; gap: 16px;
        background: rgba(0,0,0,0.85); backdrop-filter: blur(10px);
        border-top: 1px solid rgba(255,255,255,0.08);
        opacity: 0.22; transition: opacity 0.3s ease;
    }
    .present-nav-bar:hover { opacity: 1; }
    .present-nav-bar button {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.35) !important;
        color: #fff !important; border-radius: 8px !important;
        font-weight: 700 !important; transition: background 0.2s !important;
    }
    .present-nav-bar button:hover { background: rgba(255,255,255,0.22) !important; }
    .present-exit button { border-color: rgba(231,76,60,0.6) !important; color: #ff6b6b !important; }
    .present-exit button:hover { background: rgba(231,76,60,0.28) !important; }
    .present-counter {
        color: rgba(255,255,255,0.65); font-size: 0.88rem; font-weight: 600;
        min-width: 90px; text-align: center; letter-spacing: 1px; user-select: none;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────── HEADING ─────────────────
st.markdown("""
<div class="sticky-header">
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
            gap:6px;padding:18px 0 10px 0;text-align:center;">
    <div style="font-family:'Georgia',serif;font-size:3.2rem;font-weight:800;
                letter-spacing:-0.5px;line-height:1;">
        <span style="color:#ffffff;">Slide</span><span style="color:#5b7fe8;">Pilot</span>
    </div>
    <div style="color:#6b7280;font-size:0.75rem;font-weight:600;
                letter-spacing:2.5px;text-transform:uppercase;">
        Gesture &nbsp;&middot;&nbsp; Zoom &nbsp;&middot;&nbsp; AI Chat
    </div>
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
    "gesture_mode": "NAV",
    "full_view": False,
    "present_mode": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource
def get_gesture_queue():
    return queue.Queue(maxsize=1)

@st.cache_resource
def get_shared_state():
    return {
        "finger_count":  None,
        "gesture_mode":  "NAV",
        "fist_hold_pct": 0.0,
        "fsm_state":     "NEUTRAL",
    }

_shared_boot = get_shared_state()
if _shared_boot["gesture_mode"] != st.session_state.gesture_mode:
    st.session_state.gesture_mode = _shared_boot["gesture_mode"]
    st.rerun()

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

# ═══════════════════════════════════════════════════════════════════
#  GESTURE PROCESSOR
# ═══════════════════════════════════════════════════════════════════
_BUFFER_SIZE       = 6
_FIST_HOLD_SECONDS = 2.0
_DEAD_ZONE         = {2, 3}
_NO_HAND_SENTINEL  = -1


class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector     = HandDetector(maxHands=1, detectionCon=0.5)
        self.mode         = "NAV"
        self.frame_buffer: deque[int] = deque(
            [_NO_HAND_SENTINEL] * _BUFFER_SIZE, maxlen=_BUFFER_SIZE
        )
        self._fsm: dict[str, str] = {"NAV": "NEUTRAL", "ZOOM": "NEUTRAL"}
        self.fist_start   = None

    @staticmethod
    def _put(img, text, pos, scale=0.60, color=(200, 200, 200), thickness=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _hbar(img, pct, y=226, color=(102, 126, 234)):
        cv2.rectangle(img, (10, y), (210, y + 10), (50, 50, 50), -1)
        if pct > 0:
            cv2.rectangle(img, (10, y), (10 + int(200 * pct), y + 10), color, -1)

    def _stable_gesture(self) -> int:
        try:
            return stat_mode(self.frame_buffer)
        except Exception:
            return self.frame_buffer[-1]

    def _fsm_step(self, mode: str, stable: int) -> str | None:
        state   = self._fsm[mode]
        is_dead = stable in _DEAD_ZONE or stable == _NO_HAND_SENTINEL
        if state == "WAIT_FOR_RESET":
            if is_dead:
                self._fsm[mode] = "NEUTRAL"
            return None
        if state == "NEUTRAL":
            if is_dead:
                return None
            if mode == "NAV":
                action = "LEFT" if stable <= 1 else "RIGHT" if stable >= 4 else None
            else:
                action = "ZOOM_IN" if stable <= 1 else "ZOOM_OUT" if stable >= 4 else None
            if action:
                self._fsm[mode] = "WAIT_FOR_RESET"
            return action
        return None

    def recv(self, frame):
        img   = frame.to_ndarray(format="bgr24")
        small = cv2.flip(cv2.resize(img, (320, 240)), 1)
        hands, small = self.detector.findHands(small, draw=True)
        shared       = get_shared_state()
        self.mode    = shared["gesture_mode"]
        is_zoom      = self.mode == "ZOOM"
        mode_color   = (50, 200, 100) if is_zoom else (102, 126, 234)
        now          = time.time()

        if hands:
            fingers = self.detector.fingersUp(hands[0])
            total   = sum(fingers)
            self.frame_buffer.append(total)
            shared["finger_count"] = total
        else:
            self.frame_buffer.append(_NO_HAND_SENTINEL)
            shared["finger_count"]  = None
            shared["fist_hold_pct"] = 0.0
            self.fist_start         = None

        stable = self._stable_gesture()

        if stable == 0:
            if self.fist_start is None:
                self.fist_start = now
            pct = min((now - self.fist_start) / _FIST_HOLD_SECONDS, 1.0)
            shared["fist_hold_pct"] = pct
            self._put(small, f"Fist  {int(pct*100)}%  hold to switch mode",
                      (10, 30), color=(255, 200, 50))
            self._hbar(small, pct, color=(255, 200, 50))
            if pct >= 1.0:
                new_mode               = "ZOOM" if self.mode == "NAV" else "NAV"
                self.mode              = new_mode
                shared["gesture_mode"] = new_mode
                self._fsm              = {"NAV": "NEUTRAL", "ZOOM": "NEUTRAL"}
                self.frame_buffer      = deque(
                    [_NO_HAND_SENTINEL] * _BUFFER_SIZE, maxlen=_BUFFER_SIZE
                )
                try:
                    get_gesture_queue().put_nowait(f"MODE_{new_mode}")
                except queue.Full:
                    pass
                self.fist_start = None
        else:
            if stable != _NO_HAND_SENTINEL:
                self.fist_start         = None
                shared["fist_hold_pct"] = 0.0
            action    = self._fsm_step(self.mode, stable)
            fsm_state = self._fsm[self.mode]
            shared["fsm_state"] = fsm_state
            if action:
                try:
                    get_gesture_queue().put_nowait(action)
                except queue.Full:
                    pass
            label_map = {"LEFT": "← PREV", "RIGHT": "NEXT →",
                         "ZOOM_IN": "+ ZOOM IN", "ZOOM_OUT": "- ZOOM OUT"}
            if stable == _NO_HAND_SENTINEL:
                self._put(small, "No hand detected", (10, 30), color=(160, 160, 160))
            elif stable in _DEAD_ZONE:
                self._put(small, f"{stable} fingers  (dead zone)", (10, 30), color=(160, 160, 160))
            elif fsm_state == "WAIT_FOR_RESET":
                self._put(small, "Return to dead zone to re-arm", (10, 30), color=(255, 200, 50))
            else:
                lbl = label_map.get(action or "", "")
                self._put(small, f"{stable} finger{'s' if stable!=1 else ''}   {lbl}",
                          (10, 30), color=mode_color)

        mode_text = "ZOOM MODE" if is_zoom else "NAV MODE"
        cv2.rectangle(small, (0, 210), (320, 240), (20, 20, 20), -1)
        self._put(small, f"[ {mode_text} ]  ✊ fist 2s = switch",
                  (6, 228), scale=0.45, color=mode_color, thickness=1)
        return av.VideoFrame.from_ndarray(small, format="bgr24")


# ═══════════════════════════════════════════════════════════════════
#  PRESENTATION MODE
# ═══════════════════════════════════════════════════════════════════
def _inject_present_body_class(active: bool):
    action = "add" if active else "remove"
    st.markdown(
        f"""<script>(function(){{document.body.classList.{action}('present-active');}})();</script>""",
        unsafe_allow_html=True,
    )


@st.fragment(run_every="0.4s")
def render_present_mode():
    try:
        action = get_gesture_queue().get_nowait()
    except queue.Empty:
        action = None

    shared = get_shared_state()

    if action:
        if action in ("MODE_ZOOM", "MODE_NAV"):
            new_mode = "ZOOM" if action == "MODE_ZOOM" else "NAV"
            st.session_state.gesture_mode = new_mode
            shared["gesture_mode"]        = new_mode
            st.rerun()                    # full — sidebar must update
        elif st.session_state.pages:
            n = len(st.session_state.pages)
            if action == "LEFT":
                st.session_state.current_page = max(0, st.session_state.current_page - 1)
            elif action == "RIGHT":
                st.session_state.current_page = min(n - 1, st.session_state.current_page + 1)
            st.rerun(scope="fragment")    # ← does NOT touch WebRTC

    _inject_present_body_class(True)

    if not st.session_state.pages:
        st.warning("No slides loaded.")
        return

    total   = len(st.session_state.pages)
    idx     = st.session_state.current_page
    img_b64 = base64.b64encode(st.session_state.pages[idx]).decode()

    st.markdown(
        f'<div class="present-slide-wrap">'
        f'<img src="data:image/png;base64,{img_b64}" alt="Slide {idx+1}" /></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="present-nav-bar">', unsafe_allow_html=True)
    c_prev, c_counter, c_next, c_exit = st.columns([1, 2, 1, 1])

    with c_prev:
        if st.button("⬅️ Prev", key="pm_prev", use_container_width=True):
            st.session_state.current_page = max(0, idx - 1)
            st.rerun(scope="fragment")    # ← fragment-scoped

    with c_counter:
        st.markdown(
            f'<div class="present-counter">Slide &nbsp; {idx+1} &nbsp;/&nbsp; {total}</div>',
            unsafe_allow_html=True,
        )

    with c_next:
        if st.button("Next ➡️", key="pm_next", use_container_width=True):
            st.session_state.current_page = min(total - 1, idx + 1)
            st.rerun(scope="fragment")    # ← fragment-scoped

    with c_exit:
        st.markdown('<div class="present-exit">', unsafe_allow_html=True)
        if st.button("✕ Exit", key="pm_exit", use_container_width=True):
            st.session_state.present_mode = False
            _inject_present_body_class(False)
            st.rerun()                    # full — restore chrome
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  SLIDE FRAGMENT
#  KEY RULE: only st.rerun(scope="fragment") for slide/zoom changes.
#  Full st.rerun() only on mode switch (sidebar must redraw).
#  This means the sidebar — and the webrtc_streamer widget inside it —
#  is NEVER torn down by normal navigation, so the peer connection
#  stays alive for the entire session.
# ═══════════════════════════════════════════════════════════════════
@st.fragment(run_every="0.4s")
def slide_fragment(full_view: bool = False):
    try:
        action = get_gesture_queue().get_nowait()
    except queue.Empty:
        action = None

    shared = get_shared_state()

    if action:
        if action in ("MODE_ZOOM", "MODE_NAV"):
            new_mode = "ZOOM" if action == "MODE_ZOOM" else "NAV"
            st.session_state.gesture_mode = new_mode
            shared["gesture_mode"]        = new_mode
            st.rerun()                    # full — sidebar badge must flip

        elif st.session_state.pages:
            total_pages = len(st.session_state.pages)
            changed = False
            if action == "LEFT":
                new = max(0, st.session_state.current_page - 1)
                if new != st.session_state.current_page:
                    st.session_state.current_page = new
                    changed = True
            elif action == "RIGHT":
                new = min(total_pages - 1, st.session_state.current_page + 1)
                if new != st.session_state.current_page:
                    st.session_state.current_page = new
                    changed = True
            elif action == "ZOOM_IN":
                new = min(st.session_state.zoom_max,
                          round(st.session_state.zoom_level + st.session_state.zoom_step, 2))
                if new != st.session_state.zoom_level:
                    st.session_state.zoom_level = new
                    changed = True
            elif action == "ZOOM_OUT":
                new = max(st.session_state.zoom_min,
                          round(st.session_state.zoom_level - st.session_state.zoom_step, 2))
                if new != st.session_state.zoom_level:
                    st.session_state.zoom_level = new
                    changed = True

            if changed:
                st.rerun(scope="fragment")   # ← WebRTC stays alive

    _render_slide(full_view=full_view)


def _render_slide(full_view: bool = False):
    if not st.session_state.pages:
        st.info("⬆️ Upload and process a PDF to get started.")
        return

    total = len(st.session_state.pages)
    idx   = st.session_state.current_page
    zoom  = st.session_state.zoom_level

    ctrl_prev, ctrl_info, ctrl_next, ctrl_fv = st.columns([1, 5, 1, 1])

    with ctrl_prev:
        if st.button("⬅️", use_container_width=True,
                     key=f"prev_{'fv' if full_view else 'nv'}",
                     help="Previous slide"):
            st.session_state.current_page = max(0, idx - 1)
            st.rerun(scope="fragment")    # ← fragment-scoped

    with ctrl_info:
        fv_badge = '<span class="fullview-pill">⛶ full</span>' if full_view else ""
        st.markdown(
            f'<div style="padding-top:6px;">'
            f'<span style="font-size:1.05rem;font-weight:600;">📄 Slide {idx+1} / {total}</span>'
            f'<span class="zoom-pill">🔍 {zoom:.2f}×</span>'
            f'{fv_badge}</div>',
            unsafe_allow_html=True,
        )

    with ctrl_next:
        if st.button("➡️", use_container_width=True,
                     key=f"next_{'fv' if full_view else 'nv'}",
                     help="Next slide"):
            st.session_state.current_page = min(total - 1, idx + 1)
            st.rerun(scope="fragment")    # ← fragment-scoped

    with ctrl_fv:
        btn_label = "✕" if full_view else "⛶"
        btn_help  = "Exit full view" if full_view else "Expand to full width"
        if st.button(btn_label, use_container_width=True, help=btn_help,
                     key=f"fv_toggle_{'fv' if full_view else 'nv'}"):
            st.session_state.full_view = not full_view
            st.rerun()   # full — main column layout must change

    container_h = "82vh" if full_view else "600px"
    img_b64     = base64.b64encode(st.session_state.pages[idx]).decode()

    st.markdown(
        f'''<div style="overflow:auto;height:{container_h};display:flex;
                        justify-content:center;align-items:center;
                        border-radius:8px;background:rgba(0,0,0,0.04);">
              <img src="data:image/png;base64,{img_b64}"
                   style="transform:scale({zoom});transform-origin:center center;
                          transition:transform 0.35s cubic-bezier(0.25,0.46,0.45,0.94);
                          max-width:100%;height:auto;" />
            </div>''',
        unsafe_allow_html=True,
    )


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
                st.session_state.full_view    = False
                st.success(f"✅ {len(pages)} slides loaded")

    if st.session_state.pages:
        if st.button("🗑️ Unload Document", use_container_width=True,
                     help="Clear all slides, embeddings and chat history"):
            build_vector_store.clear()
            st.session_state.pages         = []
            st.session_state.vector_store  = None
            st.session_state.current_page  = 0
            st.session_state.zoom_level    = 1.0
            st.session_state.full_view     = False
            st.session_state.present_mode  = False
            st.session_state.chat_history  = []
            st.toast("✅ Document unloaded — vectors freed from memory.", icon="🗑️")
            st.rerun()

        if st.button("▶ Present", use_container_width=True,
                     help="Enter full-screen presentation mode"):
            st.session_state.present_mode = True
            st.session_state.full_view    = False
            st.rerun()

    st.divider()
    st.markdown("### 🎥 Gesture Control")

    is_zoom   = st.session_state.gesture_mode == "ZOOM"
    mode_cls  = "mode-zoom" if is_zoom else "mode-nav"
    mode_icon = "🔍 ZOOM MODE" if is_zoom else "🧭 NAV MODE"
    st.markdown(
        f'<div class="mode-badge {mode_cls}" style="width:100%">{mode_icon}</div>',
        unsafe_allow_html=True
    )

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

    toggle_label = "🔁 Switch to Zoom mode" if not is_zoom else "🔁 Switch to Nav mode"
    if st.button(toggle_label, use_container_width=True):
        new = "ZOOM" if not is_zoom else "NAV"
        st.session_state.gesture_mode      = new
        get_shared_state()["gesture_mode"] = new
        st.rerun()

    st.divider()

    ctx = webrtc_streamer(
        key="gesture",
        video_processor_factory=GestureProcessor,
        async_processing=True,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"width": 320, "height": 240}, "audio": False},
    )

    shared = get_shared_state()
    if ctx and ctx.state.playing:
        count = shared.get("finger_count")
        if count is not None:
            badge_bg = "#32c864" if is_zoom else "#667eea"
            st.markdown(
                f'<div style="display:flex;justify-content:center;">'
                f'<div class="finger-badge" style="background:{badge_bg};">{count}</div></div>'
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


# ───────────────── MAIN LAYOUT ─────────────────
present_mode = st.session_state.present_mode
full_view    = st.session_state.full_view

if present_mode:
    render_present_mode()
elif full_view:
    slide_fragment(full_view=True)
else:
    left_col, right_col = st.columns([7, 3])

    with left_col:
        slide_fragment(full_view=False)

    with right_col:
        hcol1, hcol2 = st.columns([3, 1])
        with hcol1:
            st.subheader("🤖 Groq Assistant")
        with hcol2:
            if st.button("🗑️ Clear", type="secondary", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if st.session_state.vector_store:
            chat_container = st.container(height=390)

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
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": reply}
                            )
                        else:
                            try:
                                docs    = st.session_state.vector_store.similarity_search(prompt, k=3)
                                context = "\n\n".join(d.page_content for d in docs)
                                history_msgs = st.session_state.chat_history[-6:]
                                history = "\n".join(
                                    f"{m['role'].capitalize()}: {m['content']}"
                                    for m in history_msgs
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
                                        "context":  context,
                                        "question": prompt,
                                        "history":  history,
                                    })
                                )
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": full_response}
                                )
                            except Exception as e:
                                err_msg = f"⚠️ Error: {str(e)}"
                                st.error(err_msg)
                                st.session_state.chat_history.append(
                                    {"role": "assistant", "content": err_msg}
                                )
        else:
            st.info("⬅️ Process a PDF first to enable the assistant.")
