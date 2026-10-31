"""
Eye State Detector — open vs. closed eyes (images, video, webcam snapshot)

Run:   streamlit run app.py
Needs: eye_model.keras  (or best_model.h5) next to this file.
"""
import os
import shutil
import subprocess
import tempfile
import urllib.request
from collections import deque
from pathlib import Path

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ───────────────────────────── Config ─────────────────────────────
BASE = Path(__file__).resolve().parent
MODEL_FILES = [BASE / "eye_model.keras", BASE / "best_model.h5"]
YUNET_PATH = BASE / "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
IMG_SIZE = 224          # same as training
LONG_CLOSURE_S = 1.0    # closures longer than this are reported separately from blinks
GREEN, RED = (94, 197, 34), (68, 68, 239)  # BGR

st.set_page_config(page_title="Eye State Detector", page_icon="👁️", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1250px; }
.hero { padding: 28px 34px; border-radius: 20px; margin-bottom: 22px; color: #fff;
        background: linear-gradient(135deg, #4f46e5 0%, #0891b2 100%); }
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; color: #fff; }
.hero p  { margin: 6px 0 0; opacity: .92; font-size: 1.02rem; }
[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.28); border-radius: 14px;
        padding: 14px 18px; background: rgba(128,128,128,.07); }
.card { border: 1px solid rgba(128,128,128,.28); border-radius: 14px; padding: 14px 18px;
        background: rgba(128,128,128,.07); margin-bottom: 12px; }
.badge { display: inline-block; padding: 4px 14px; border-radius: 999px; font-weight: 700;
         font-size: .85rem; letter-spacing: .03em; }
.badge.open   { background: rgba(34,197,94,.18);  color: #16a34a; }
.badge.closed { background: rgba(239,68,68,.18);  color: #dc2626; }
.bar { height: 8px; border-radius: 6px; background: rgba(128,128,128,.25); overflow: hidden; }
.bar > div { height: 100%; border-radius: 6px; }
.row { display: flex; justify-content: space-between; font-size: .9rem; margin: 8px 0 4px; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
<div class="hero">
  <h1>👁️ Eye State Detector</h1>
  <p>CNN-based open / closed eye detection for photos, videos and webcam snapshots.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ───────────────────────────── Model & detector ─────────────────────────────
def build_model():
    """Same architecture as the notebook — used to load weights when the saved file
    was made with a newer Keras than the one installed here."""
    from tensorflow.keras import Sequential, layers

    return Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"), layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2), layers.Flatten(),
        layers.Dense(128, activation="relu"), layers.Dense(1, activation="sigmoid"),
    ])


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    import tensorflow as tf

    for p in MODEL_FILES:
        if not p.exists():
            continue
        try:
            return tf.keras.models.load_model(p, compile=False), p.name
        except Exception:  # Keras version mismatch → rebuild architecture + load weights only
            m = build_model()
            m.load_weights(str(p))
            return m, f"{p.name} (weights)"
    return None, None


class FaceDetector:
    """YuNet (OpenCV DNN) → accurate, gives eye centres even when eyes are closed.
    Falls back to Haar cascade + facial proportions if YuNet can't be downloaded."""

    def __init__(self):
        self.yunet = None
        try:
            if not YUNET_PATH.exists() or YUNET_PATH.stat().st_size < 100_000:
                urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
            self.yunet = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.7, 0.3, 50)
        except Exception:
            YUNET_PATH.unlink(missing_ok=True)
        try:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        except Exception:
            self.haar = None
        self.name = "YuNet" if self.yunet else "Haar (fallback)"

    def detect(self, bgr, max_side=1280):
        h, w = bgr.shape[:2]
        k = min(1.0, max_side / max(h, w))
        img = cv2.resize(bgr, None, fx=k, fy=k) if k < 1 else bgr
        faces = []
        if self.yunet:
            self.yunet.setInputSize((img.shape[1], img.shape[0]))
            _, det = self.yunet.detect(img)
            for d in (det if det is not None else []):
                faces.append((d[:4] / k, (d[4:6] / k, d[6:8] / k)))
        else:
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for x, y, fw, fh in self.haar.detectMultiScale(g, 1.1, 5, minSize=(50, 50)):
                re = np.array([x + .31 * fw, y + .40 * fh]) / k
                le = np.array([x + .69 * fw, y + .40 * fh]) / k
                faces.append((np.array([x, y, fw, fh]) / k, (re, le)))
        return faces


@st.cache_resource
def load_detector():
    return FaceDetector()


# ───────────────────────────── Core pipeline ─────────────────────────────
def eye_crop(gray, center, angle, side):
    """Rotate so the eye-line is horizontal, crop a square around the eye, resize to 224×224."""
    M = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, IMG_SIZE / side)
    M[0, 2] += IMG_SIZE / 2 - center[0]
    M[1, 2] += IMG_SIZE / 2 - center[1]
    return cv2.warpAffine(gray, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def predict_open(model, crops):
    """Returns P(open) per crop. Same preprocessing as training: gray, 224×224, /255."""
    x = np.stack(crops).astype("float32")[..., None] / 255.0
    return model(x, training=False).numpy().ravel()


def analyze(bgr, model, detector, crop_scale, mode="auto", allow_fallback=True):
    """→ list of dicts {kind, box, eyes, side, crops, p_eyes, p, p_disp}"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    faces = [] if mode == "eye" else detector.detect(bgr)
    results, crops = [], []

    for box, (re, le) in faces:
        iod = float(np.linalg.norm(le - re))
        if iod < 8:
            continue
        side = crop_scale * iod
        angle = float(np.degrees(np.arctan2(le[1] - re[1], le[0] - re[0])))
        ec = [eye_crop(gray, c, angle, side) for c in (re, le)]
        results.append(dict(kind="face", box=tuple(map(float, box)), eyes=[tuple(re), tuple(le)],
                            side=side, crops=ec))

    if not results and (mode == "eye" or (mode == "auto" and allow_fallback)):
        results.append(dict(kind="eye", box=(0, 0, w, h), eyes=[(w / 2, h / 2)], side=min(w, h),
                            crops=[cv2.resize(gray, (IMG_SIZE, IMG_SIZE))]))

    for r in results:
        crops += r["crops"]
    if crops:
        probs = predict_open(model, crops)
        i = 0
        for r in results:
            n = len(r["crops"])
            r["p_eyes"] = probs[i:i + n].tolist()
            r["p"] = float(np.mean(r["p_eyes"]))
            r["p_disp"] = r["p"]
            i += n
    return results


def draw(bgr, results, thr):
    out = bgr.copy()
    t = max(2, round(max(out.shape[:2]) / 450))
    for r in results:
        col = GREEN if r["p_disp"] >= thr else RED
        x, y, w, h = map(int, r["box"])
        if r["kind"] == "face":
            cv2.rectangle(out, (x, y), (x + w, y + h), col, t)
            for (ex, ey), pe in zip(r["eyes"], r["p_eyes"]):
                s = int(r["side"] / 2)
                cv2.rectangle(out, (int(ex) - s, int(ey) - s), (int(ex) + s, int(ey) + s),
                              GREEN if pe >= thr else RED, max(1, t - 1))
        else:
            cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), col, t * 2)
            x, y = t * 2, t * 2 + 28
        conf = r["p_disp"] if r["p_disp"] >= thr else 1 - r["p_disp"]
        label = f"{'OPEN' if r['p_disp'] >= thr else 'CLOSED'} {conf:.0%}"
        fs = max(0.5, out.shape[1] / 1400 * 1.1)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        ty = max(y, th + 10)
        cv2.rectangle(out, (x, ty - th - 10), (x + tw + 12, ty + 4), col, -1)
        cv2.putText(out, label, (x + 6, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def to_h264(src, dst):
    """OpenCV writes mp4v which browsers can't play → re-encode to H.264."""
    exe = shutil.which("ffmpeg")
    if not exe:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return False
    r = subprocess.run([exe, "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                        "-crf", "23", "-movflags", "+faststart", "-an", dst], capture_output=True)
    return r.returncode == 0 and os.path.exists(dst)


def closure_stats(p, dt, thr):
    """p: smoothed P(open) per analysed sample (NaN = no face)."""
    p = np.asarray(p, dtype=float)
    valid = ~np.isnan(p)
    closed = valid & (p < thr)
    runs, n = [], 0
    for c in list(closed) + [False]:
        if c:
            n += 1
        elif n:
            runs.append(n * dt)
            n = 0
    blinks = sum(d < LONG_CLOSURE_S for d in runs)
    long_ = sum(d >= LONG_CLOSURE_S for d in runs)
    dur = valid.sum() * dt
    return dict(
        blinks=blinks, long=long_, longest=max(runs, default=0.0),
        closed_pct=100 * closed.sum() / max(valid.sum(), 1),
        rate=blinks / (dur / 60) if dur > 0 else 0.0, valid=int(valid.sum()),
    )


# ───────────────────────────── UI helpers ─────────────────────────────
def eye_row(label, p, thr):
    is_open = p >= thr
    color = "#22c55e" if is_open else "#ef4444"
    conf = p if is_open else 1 - p
    st.markdown(
        f"""<div class="row"><span>{label}</span>
        <span class="badge {'open' if is_open else 'closed'}">{'OPEN' if is_open else 'CLOSED'} · {conf:.1%}</span></div>
        <div class="bar"><div style="width:{p*100:.1f}%;background:{color}"></div></div>
        <div style="font-size:.75rem;opacity:.6;margin-top:3px">P(open) = {p:.3f}</div>""",
        unsafe_allow_html=True,
    )


def show_results(bgr, results, thr, name=""):
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.image(cv2.cvtColor(draw(bgr, results, thr), cv2.COLOR_BGR2RGB), caption=name, use_container_width=True)
    with right:
        if not results:
            st.warning("No face detected. Try a clearer photo, or switch **Input type** to *Eye close-ups* in the sidebar.")
            return
        if results[0]["kind"] == "eye":
            st.info("No face found → image treated as an eye close-up.")
        for i, r in enumerate(results, 1):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            title = f"Face {i}" if r["kind"] == "face" else "Eye close-up"
            st.markdown(f"**{title}**")
            names = ["Right eye (left in image)", "Left eye (right in image)"] if r["kind"] == "face" else ["Eye"]
            for nm, pe in zip(names, r["p_eyes"]):
                eye_row(nm, pe, thr)
            st.markdown("</div>", unsafe_allow_html=True)
        with st.expander("What the model sees (224×224 grayscale)"):
            crops = [c for r in results for c in r["crops"]]
            st.image(crops[:8], width=110, clamp=True)


# ───────────────────────────── Sidebar ─────────────────────────────
model, model_name = load_model()
detector = load_detector()

with st.sidebar:
    st.header("⚙️ Settings")
    mode_label = st.radio("Input type", ["Auto (recommended)", "Face photos", "Eye close-ups"],
                          help="*Eye close-ups* feeds the whole image to the CNN — the same kind of images it was trained on.")
    mode = {"Auto (recommended)": "auto", "Face photos": "face", "Eye close-ups": "eye"}[mode_label]
    thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                    help="P(open) ≥ threshold → OPEN. Lower it if open eyes get flagged as closed (e.g. glasses, squinting).")
    crop_scale = st.slider("Eye crop size", 0.40, 1.00, 0.60, 0.05,
                           help="Crop side as a fraction of the distance between the eyes. Tune this if accuracy "
                                "looks off — it should frame one eye with a little skin around it.")
    st.subheader("🎞️ Video")
    stride = st.slider("Analyse every N-th frame", 1, 10, 1, help="1 = most accurate, higher = faster.")
    smooth = st.slider("Smoothing window", 1, 9, 3, 2, help="Median filter over analysed frames to remove flicker.")

    st.divider()
    if model is None:
        st.error("Model file not found.\n\nPut **eye_model.keras** (or **best_model.h5**) next to app.py.")
    else:
        st.success(f"Model: {model_name}")
    st.caption(f"Face detector: {detector.name}")
    with st.expander("About"):
        st.markdown(
            "- Architecture: 3×Conv-Pool + Dense (11.2 M params)\n"
            "- Input: 224×224 grayscale, scaled to 0-1\n"
            "- Test accuracy in the notebook: **99 %** (600 images)\n"
            "- Faces → eyes are located, aligned and cropped before classification.\n\n"
            "Not intended for safety-critical use (e.g. driver-drowsiness decisions)."
        )

if model is None:
    st.stop()

tab_img, tab_vid, tab_cam = st.tabs(["🖼️ Images", "🎬 Video", "📷 Camera"])

# ───────────────────────────── Images ─────────────────────────────
with tab_img:
    files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png", "webp", "bmp"],
                             accept_multiple_files=True)
    for f in files:
        bgr = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            st.error(f"Couldn't read {f.name}")
            continue
        show_results(bgr, analyze(bgr, model, detector, crop_scale, mode), thr, f.name)
        st.divider()
    if not files:
        st.info("Drop photos here — faces are detected automatically, or use *Eye close-ups* for cropped eye images.")

# ───────────────────────────── Camera ─────────────────────────────
with tab_cam:
    shot = st.camera_input("Take a photo")
    if shot:
        bgr = cv2.imdecode(np.frombuffer(shot.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        show_results(bgr, analyze(bgr, model, detector, crop_scale, mode), thr, "Camera")

# ───────────────────────────── Video ─────────────────────────────
def process_video(path, mode, thr, crop_scale, stride, smooth):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    writer = cv2.VideoWriter(tmp_out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    bar = st.progress(0.0, text="Analysing video…")
    hist, last, rows, i = deque(maxlen=smooth), [], [], 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            last = analyze(frame, model, detector, crop_scale, mode, allow_fallback=False)
            p_raw = p_s = np.nan
            if last:
                prim = max(last, key=lambda r: r["box"][2] * r["box"][3])
                hist.append(prim["p"])
                prim["p_disp"] = float(np.median(hist))
                p_raw, p_s = prim["p"], prim["p_disp"]
            rows.append(dict(frame=i, time_s=round(i / fps, 3), faces=len(last), p_open_raw=p_raw, p_open=p_s))
        writer.write(draw(frame, last, thr))
        i += 1
        if i % 10 == 0:
            bar.progress(min(i / total, 1.0), text=f"Analysing video… {i}/{total} frames")
    cap.release()
    writer.release()
    bar.progress(1.0, text="Encoding video…")

    final = tmp_out.replace(".mp4", "_h264.mp4")
    encoded = to_h264(tmp_out, final)
    data = Path(final if encoded else tmp_out).read_bytes()
    for p in (tmp_out, final):
        Path(p).unlink(missing_ok=True)
    bar.empty()

    df = pd.DataFrame(rows)
    return dict(video=data, playable=encoded, df=df, fps=fps, dt=stride / fps, frames=i, thr=thr)


with tab_vid:
    vf = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv", "webm"])
    if vf:
        key = (vf.name, vf.size, mode, thr, crop_scale, stride, smooth)
        if st.session_state.get("vkey") != key:
            st.session_state.pop("vres", None)
        if st.button("▶ Analyse video", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(vf.name).suffix) as t:
                t.write(vf.getvalue())
            try:
                st.session_state["vres"] = process_video(t.name, mode, thr, crop_scale, stride, smooth)
                st.session_state["vkey"] = key
            finally:
                Path(t.name).unlink(missing_ok=True)

    res = st.session_state.get("vres")
    if vf and res:
        df, s = res["df"], closure_stats(res["df"]["p_open"], res["dt"], res["thr"])
        c = st.columns(5)
        c[0].metric("Duration", f"{res['frames'] / res['fps']:.1f} s")
        c[1].metric("Blinks", s["blinks"])
        c[2].metric("Blink rate", f"{s['rate']:.0f} /min")
        c[3].metric("Eyes closed", f"{s['closed_pct']:.1f} %")
        c[4].metric(f"Closures ≥ {LONG_CLOSURE_S:.0f}s", s["long"], help=f"Longest: {s['longest']:.2f} s")
        if s["valid"] == 0:
            st.warning("No faces were detected in this video.")

        v, g = st.columns([3, 2], gap="large")
        with v:
            st.video(res["video"])
            if not res["playable"]:
                st.caption("⚠️ ffmpeg not found — preview may not play in the browser; download the file instead.")
            st.download_button("⬇️ Download annotated video", res["video"], "eye_state_annotated.mp4", "video/mp4")
        with g:
            line = alt.Chart(df).mark_line(color="#6366f1").encode(
                x=alt.X("time_s:Q", title="Time (s)"),
                y=alt.Y("p_open:Q", title="P(open)", scale=alt.Scale(domain=[0, 1])),
            )
            rule = alt.Chart(pd.DataFrame({"y": [res["thr"]]})).mark_rule(color="#ef4444", strokeDash=[5, 4]).encode(y="y:Q")
            st.altair_chart((line + rule).properties(height=260), use_container_width=True)
            st.caption("Statistics use the largest face in the frame. Dashed line = decision threshold.")
            st.download_button("⬇️ Download per-frame CSV", df.to_csv(index=False), "eye_state_frames.csv", "text/csv")
    elif not vf:
        st.info("Upload a clip with a visible face (frontal or slightly angled works best).")