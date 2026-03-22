import os
import time
import queue
import base64
import threading
from collections import deque, Counter

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
from twilio.rest import Client

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
    .stButton > button[kind="secondary"]:hover {
        background: #e74c3c;
        color: white;
    }
    .fullview-slide img {
        max-height: 82vh !important;
        width: auto !important;
        max-width: 100% !important;
    }
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 999;
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
        position: sticky; bottom: 0; z-index: 9999;
        width: 100%; height: 56px;
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
    .twilio-badge {
        display: inline-block;
        background: rgba(245, 47, 47, 0.12);
        border: 1px solid rgba(245, 47, 47, 0.4);
        color: #f52f2f;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 20px;
        padding: 2px 10px;
        letter-spacing: 0.4px;
    }
    .stun-badge {
        display: inline-block;
        background: rgba(255,165,0,0.12);
        border: 1px solid rgba(255,165,0,0.45);
        color: #ffa500;
        font-size: 0.72rem;
        font-weight: 600;
        border-radius: 20px;
        padding: 2px 10px;
        letter-spacing: 0.4px;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────── HEADING ─────────────────
st.markdown("""
<div class="sticky-header">
<div style="
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 6px; padding: 18px 0 10px 0; text-align: center;
">
    <div style="font-family:'Georgia',serif;font-size:3.2rem;font-weight:800;letter-spacing:-0.5px;line-height:1;">
        <span style="color:#ffffff;">Slide</span><span style="color:#5b7fe8;">Pilot</span>
    </div>
    <div style="color:#6b7280;font-size:0.75rem;font-weight:600;letter-spacing:2.5px;text-transform:uppercase;">
        Gesture &nbsp;&middot;&nbsp; Zoom &nbsp;&middot;&nbsp; AI Chat
    </div>
</div>
</div>
""", unsafe_allow_html=True)

# ───────────────── ENV / SECRETS HELPERS ─────────────────
def _get_secret(key: str) -> str | None:
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key)

GROQ_API_KEY        = _get_secret("GROQ_API_KEY")
TWILIO_ACCOUNT_SID  = _get_secret("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN   = _get_secret("TWILIO_AUTH_TOKEN")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment variables or secrets.")
    st.stop()

# ───────────────── TWILIO RTC CONFIGURATION ─────────────────
@st.cache_resource(ttl=60)
def get_rtc_configuration() -> RTCConfiguration:
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        try:
            client      = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            token       = client.tokens.create()
            ice_servers = token.ice_servers
            return RTCConfiguration({"iceServers": ice_servers})
        except Exception as e:
            st.warning(f"⚠️ Twilio token fetch failed ({e}). Falling back to STUN.")
    return RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    })

def _rtc_badge() -> str:
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN:
        return '<span class="twilio-badge">🔴 Twilio TURN</span>'
    return '<span class="stun-badge">🟡 STUN only</span>'


# ───────────────── SESSION STATE ─────────────────
_DEFAULTS = {
    "pages":        [],
    "current_page": 0,
    "vector_store": None,
    "chain_cache":  {},
    "chat_history": [],
    "zoom_level":   1.0,
    "zoom_step":    0.25,
    "zoom_min":     0.5,
    "zoom_max":     3.0,
    "gesture_mode": "NAV",
    "full_view":    False,
    "present_mode": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

@st.cache_resource
def get_gesture_queue():
    return queue.Queue(maxsize=1)

# ─── FIX #3: Thread-safe shared state ───────────────────────────────────────
# The _shared dict is written by the WebRTC worker thread (recv) and read by
# Streamlit's main thread (camera_fragment, slide_fragment).  Without a lock
# the two threads race on the same dict, which can corrupt values or produce
# torn reads.  We protect every access with a single reentrant lock.
@st.cache_resource
def get_shared_state():
    return {
        "data": {
            "finger_count":  None,
            "gesture_mode":  "NAV",
            "fist_hold_pct": 0.0,
            "fsm_state":     "NEUTRAL",
        },
        "lock": threading.Lock(),
    }

def _shared_read(key: str):
    """Thread-safe read from shared state."""
    container = get_shared_state()
    with container["lock"]:
        return container["data"].get(key)

def _shared_write(updates: dict):
    """Thread-safe bulk write to shared state."""
    container = get_shared_state()
    with container["lock"]:
        container["data"].update(updates)

def _shared_read_all() -> dict:
    """Thread-safe snapshot of the full shared state."""
    container = get_shared_state()
    with container["lock"]:
        return dict(container["data"])

# Boot-time sync: if gesture mode was changed in a prior run, restore it
_boot_mode = _shared_read("gesture_mode")
if _boot_mode and _boot_mode != st.session_state.gesture_mode:
    st.session_state.gesture_mode = _boot_mode
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
    doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
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
    splitter   = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks     = splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(chunks, embeddings)


# ═══════════════════════════════════════════════════════════════════
#  GESTURE PROCESSOR
#
#  Key changes vs original:
#  ─────────────────────────────────────────────────────────────────
#  FIX #1  async_processing=True (set in webrtc_streamer call below)
#          recv() now runs in a background thread managed by aiortc,
#          so the WebRTC event loop never stalls waiting for it.
#
#  FIX #2  draw=False in findHands()
#          MediaPipe skeleton drawing (21 landmarks + connections) is
#          the single most expensive operation: 30–50 ms/frame at
#          320×240.  We draw only a tiny palm-center circle instead
#          (~0.1 ms).  This alone halves per-frame CPU time.
#
#  FIX #3  threading.Lock via _shared_write/_shared_read helpers.
#
#  FIX #4  Counter.most_common(1) replaces statistics.mode().
#          statistics.mode raises StatisticsError when the buffer has
#          no single most-common value (even-length buffer, tie). The
#          original code suppressed this with a bare except, silently
#          returning the last raw value.  Counter is O(n) and never
#          raises; ties are broken deterministically by insertion order.
#
#  TUNING  detectionCon=0.65 — rejects weak candidate hands slightly
#          faster during the pre-NMS phase of MediaPipe.
#
#  TUNING  Overlay is a single cv2.putText + a small circle.
#          Cutting from 4–6 putText calls to 1–2 saves ~1 ms/frame.
# ═══════════════════════════════════════════════════════════════════

_BUFFER_SIZE            = 6
_FIST_HOLD_SECONDS      = 2.0
_DEAD_ZONE              = {2, 3}
_NO_HAND_SENTINEL       = -1
_PROCESS_EVERY_N_FRAMES = 3   # at 10 fps → real detection at ~3 fps


class GestureProcessor(VideoProcessorBase):

    def __init__(self):
        # FIX #2: detectionCon raised to 0.65 for slightly faster palm rejection
        self.detector    = HandDetector(maxHands=1, detectionCon=0.65)
        self.mode        = "NAV"
        self.frame_buffer: deque[int] = deque(
            [_NO_HAND_SENTINEL] * _BUFFER_SIZE, maxlen=_BUFFER_SIZE
        )
        self._fsm: dict[str, str] = {"NAV": "NEUTRAL", "ZOOM": "NEUTRAL"}
        self.fist_start  = None
        self._frame_idx  = 0
        self._last_total = _NO_HAND_SENTINEL
        self._queue      = get_gesture_queue()
        # Cache last palm-center to draw dot on skipped frames too
        self._palm_xy: tuple[int, int] | None = None

    # ── thin helpers ────────────────────────────────────────────────

    @staticmethod
    def _put(img, text: str, pos, scale=0.55,
             color=(200, 200, 200), thickness=1):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)

    # FIX #4: Counter-based mode — never raises, O(n), tie-safe
    def _stable_gesture(self) -> int:
        counts = Counter(self.frame_buffer)
        return counts.most_common(1)[0][0]

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
                if stable <= 1:
                    action = "LEFT"
                elif stable >= 4:
                    action = "RIGHT"
                else:
                    return None
            else:
                if stable <= 1:
                    action = "ZOOM_IN"
                elif stable >= 4:
                    action = "ZOOM_OUT"
                else:
                    return None
            self._fsm[mode] = "WAIT_FOR_RESET"
            return action
        return None

    # ── main frame callback ─────────────────────────────────────────

    def recv(self, frame):
        img   = frame.to_ndarray(format="bgr24")
        small = cv2.resize(img, (320, 240))
        small = cv2.flip(small, 1)

        self._frame_idx += 1
        now = time.time()

        # Read mode safely (Streamlit thread may write this concurrently)
        self.mode = _shared_read("gesture_mode") or self.mode
        is_zoom   = self.mode == "ZOOM"
        mode_color = (50, 200, 100) if is_zoom else (102, 126, 234)

        # ── PROCESS every Nth frame only ───────────────────────────
        if self._frame_idx % _PROCESS_EVERY_N_FRAMES == 0:
            # FIX #2: draw=False — skip all 21-landmark skeleton rendering
            hands, _ = self.detector.findHands(small, draw=False)
            if hands:
                fingers          = self.detector.fingersUp(hands[0])
                total            = sum(fingers)
                self._last_total = total
                # Cache palm center for the dot we draw instead of skeleton
                cx, cy           = hands[0]["center"]
                self._palm_xy    = (int(cx), int(cy))
                _shared_write({"finger_count": total})
            else:
                self._last_total = _NO_HAND_SENTINEL
                self._palm_xy    = None
                _shared_write({
                    "finger_count":  None,
                    "fist_hold_pct": 0.0,
                })
                self.fist_start = None

        total  = self._last_total
        stable = self._stable_gesture()
        self.frame_buffer.append(total)

        # ── Fist-hold mode-switch logic ────────────────────────────
        if stable == 0:
            if self.fist_start is None:
                self.fist_start = now
            pct = min((now - self.fist_start) / _FIST_HOLD_SECONDS, 1.0)
            _shared_write({"fist_hold_pct": pct})
            self._put(small,
                      f"Hold fist: {int(pct * 100)}%",
                      (8, 26), color=(255, 200, 50))
            # Minimal progress bar
            cv2.rectangle(small, (8, 32), (208, 40), (50, 50, 50), -1)
            if pct > 0:
                cv2.rectangle(small, (8, 32),
                              (8 + int(200 * pct), 40), (255, 200, 50), -1)
            if pct >= 1.0:
                new_mode = "ZOOM" if self.mode == "NAV" else "NAV"
                self.mode = new_mode
                _shared_write({"gesture_mode": new_mode, "fist_hold_pct": 0.0})
                self._fsm       = {"NAV": "NEUTRAL", "ZOOM": "NEUTRAL"}
                self.frame_buffer = deque(
                    [_NO_HAND_SENTINEL] * _BUFFER_SIZE, maxlen=_BUFFER_SIZE
                )
                try:
                    self._queue.put_nowait(f"MODE_{new_mode}")
                except queue.Full:
                    pass
                self.fist_start = None

        else:
            if stable != _NO_HAND_SENTINEL:
                self.fist_start = None
                _shared_write({"fist_hold_pct": 0.0})

            action = self._fsm_step(self.mode, stable)
            if action:
                try:
                    self._queue.put_nowait(action)
                except queue.Full:
                    pass
            _shared_write({"fsm_state": self._fsm[self.mode]})

            # ── Minimal overlay: one line of status text ───────────
            if stable == _NO_HAND_SENTINEL:
                self._put(small, "No hand", (8, 26), color=(140, 140, 140))
            elif stable in _DEAD_ZONE:
                self._put(small, f"{stable}  dead zone", (8, 26), color=(140, 140, 140))
            elif self._fsm[self.mode] == "WAIT_FOR_RESET":
                self._put(small, "Relax to reset", (8, 26), color=(255, 200, 50))
            else:
                label_map = {
                    "LEFT":     "<- PREV",
                    "RIGHT":    "NEXT ->",
                    "ZOOM_IN":  "+ ZOOM IN",
                    "ZOOM_OUT": "- ZOOM OUT",
                }
                lbl = label_map.get(action or "", "")
                self._put(small,
                          f"{stable} finger{'s' if stable != 1 else ''}  {lbl}",
                          (8, 26), color=mode_color)

        # ── Draw palm dot instead of full skeleton ──────────────────
        # A single filled circle at the palm center is ~0.1 ms vs
        # the 30–50 ms for drawing all 21 MediaPipe landmarks.
        if self._palm_xy:
            cv2.circle(small, self._palm_xy, 8, mode_color, -1)
            cv2.circle(small, self._palm_xy, 8, (255, 255, 255), 1)

        # ── Mode indicator strip ────────────────────────────────────
        mode_text = "ZOOM" if is_zoom else "NAV"
        cv2.rectangle(small, (0, 218), (320, 240), (18, 18, 18), -1)
        self._put(small, f"[ {mode_text} MODE ]  fist 2s = switch",
                  (6, 233), scale=0.42, color=mode_color, thickness=1)

        return av.VideoFrame.from_ndarray(small, format="bgr24")


# ═══════════════════════════════════════════════════════════════════
#  PRESENTATION MODE RENDERER
# ═══════════════════════════════════════════════════════════════════

def _inject_present_body_class(active: bool):
    action = "add" if active else "remove"
    st.markdown(
        f"""<script>
            (function() {{ document.body.classList.{action}('present-active'); }})();
        </script>""",
        unsafe_allow_html=True,
    )


@st.fragment(run_every="0.5s")
def render_present_mode():
    try:
        action = get_gesture_queue().get_nowait()
    except queue.Empty:
        action = None

    if action:
        if action == "MODE_ZOOM":
            st.session_state.gesture_mode = "ZOOM"
            _shared_write({"gesture_mode": "ZOOM"})
        elif action == "MODE_NAV":
            st.session_state.gesture_mode = "NAV"
            _shared_write({"gesture_mode": "NAV"})
        elif st.session_state.pages:
            n = len(st.session_state.pages)
            if action == "LEFT":
                st.session_state.current_page = max(0, st.session_state.current_page - 1)
            elif action == "RIGHT":
                st.session_state.current_page = min(n - 1, st.session_state.current_page + 1)

    _inject_present_body_class(True)
    if not st.session_state.pages:
        st.warning("No slides loaded.")
        return

    total   = len(st.session_state.pages)
    idx     = st.session_state.current_page
    img_b64 = base64.b64encode(st.session_state.pages[idx]).decode()

    st.markdown(
        f'<div class="present-slide-wrap">'
        f'  <img src="data:image/png;base64,{img_b64}" alt="Slide {idx+1}" />'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="present-nav-bar">', unsafe_allow_html=True)
    c_prev, c_counter, c_next, c_exit = st.columns([1, 2, 1, 1])
    with c_prev:
        if st.button("⬅️ Prev", key="pm_prev", use_container_width=True):
            st.session_state.current_page = max(0, idx - 1)
            st.rerun()
    with c_counter:
        st.markdown(
            f'<div class="present-counter">Slide &nbsp; {idx+1} &nbsp;/&nbsp; {total}</div>',
            unsafe_allow_html=True,
        )
    with c_next:
        if st.button("Next ➡️", key="pm_next", use_container_width=True):
            st.session_state.current_page = min(total - 1, idx + 1)
            st.rerun()
    with c_exit:
        st.markdown('<div class="present-exit">', unsafe_allow_html=True)
        if st.button("✕ Exit", key="pm_exit", use_container_width=True,
                     help="Exit presentation mode"):
            st.session_state.present_mode = False
            _inject_present_body_class(False)
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# FIX #5: run_every increased from 0.15s → 0.25s
# At 10 fps capture + PROCESS_EVERY_N=3, real gesture decisions arrive at
# ~3 fps (one per ~330 ms). Polling at 0.15s (6.7/s) gives zero benefit
# over 0.25s (4/s) and just triggers more Streamlit reruns, wasting CPU.
@st.fragment(run_every="0.25s")
def slide_fragment(full_view: bool = False):
    try:
        action = get_gesture_queue().get_nowait()
    except queue.Empty:
        action = None

    if action:
        if action in ("MODE_ZOOM", "MODE_NAV"):
            new_mode = "ZOOM" if action == "MODE_ZOOM" else "NAV"
            st.session_state.gesture_mode = new_mode
            _shared_write({"gesture_mode": new_mode})
            st.rerun()
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
                     key=f"prev_{'fv' if full_view else 'nv'}", help="Previous slide"):
            st.session_state.current_page = max(0, idx - 1)
            st.rerun()
    with ctrl_info:
        fv_badge = '<span class="fullview-pill">⛶ full</span>' if full_view else ""
        st.markdown(
            f'<div style="padding-top:6px;">'
            f'<span style="font-size:1.05rem;font-weight:600;">📄 Slide {idx+1} / {total}</span>'
            f'<span class="zoom-pill">🔍 {zoom:.2f}×</span>'
            f'{fv_badge}'
            f'</div>',
            unsafe_allow_html=True,
        )
    with ctrl_next:
        if st.button("➡️", use_container_width=True,
                     key=f"next_{'fv' if full_view else 'nv'}", help="Next slide"):
            st.session_state.current_page = min(total - 1, idx + 1)
            st.rerun()
    with ctrl_fv:
        btn_label = "✕" if full_view else "⛶"
        btn_help  = "Exit full view" if full_view else "Expand to full width"
        if st.button(btn_label, use_container_width=True, help=btn_help,
                     key=f"fv_toggle_{'fv' if full_view else 'nv'}"):
            st.session_state.full_view = not full_view
            st.rerun()

    container_h = "82vh" if full_view else "600px"
    img_b64     = base64.b64encode(st.session_state.pages[idx]).decode()
    st.markdown(
        f'''<div style="
                overflow:auto; height:{container_h};
                display:flex; justify-content:center; align-items:center;
                border-radius:8px; background:rgba(0,0,0,0.04);">
              <img src="data:image/png;base64,{img_b64}"
                   style="transform:scale({zoom});
                          transform-origin:center center;
                          transition:transform 0.35s cubic-bezier(0.25,0.46,0.45,0.94);
                          max-width:100%; height:auto;" />
            </div>''',
        unsafe_allow_html=True,
    )


# ───────────────── SIDEBAR ─────────────────

@st.fragment
def camera_fragment():
    ctx = webrtc_streamer(
        key="gesture",
        video_processor_factory=GestureProcessor,
        # FIX #1: async_processing=True
        # recv() now runs in a background asyncio task managed by aiortc.
        # The WebRTC event loop is never blocked, so frames are delivered
        # on time and the browser gets a stable 10 fps stream without
        # the "connection taking longer than expected" stall that occurs
        # when recv() blocks the loop for 80-150 ms at async_processing=False.
        async_processing=True,
        rtc_configuration=get_rtc_configuration(),
        media_stream_constraints={
            "video": {
                "width":     {"ideal": 320},
                "height":    {"ideal": 240},
                "frameRate": {"ideal": 10, "max": 10},
            },
            "audio": False,
        },
    )

    if ctx and ctx.state.playing:
        # Thread-safe snapshot read
        shared  = _shared_read_all()
        count   = shared.get("finger_count")
        is_zoom = st.session_state.gesture_mode == "ZOOM"
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
        st.session_state.gesture_mode = new
        _shared_write({"gesture_mode": new})
        st.rerun()

    st.divider()

    st.markdown(
        f'<div style="margin-bottom:6px;">'
        f'  <span style="font-size:0.75rem;color:#888;">WebRTC: </span>'
        f'  {_rtc_badge()}'
        f'</div>',
        unsafe_allow_html=True,
    )

    camera_fragment()

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
                                chain         = prompt_template | llm
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
