import streamlit as st
import cv2
import torch
import torch.nn as nn
import timm
import numpy as np
import librosa
import sounddevice as sd
from torchvision import transforms
import threading
import queue
import time

# ─── CONFIG ───────────────────────────────────────────────
FACE_EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
AUDIO_EMOTIONS = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']
SAMPLE_RATE = 22050
AUDIO_DURATION = 3

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ─── MODELS ───────────────────────────────────────────────
class FaceEmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.base = timm.create_model('efficientnet_b0', pretrained=False)
        in_features = self.base.classifier.in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        return self.base(x)

class AudioEmotionLSTM(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_layers=2, num_classes=8):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

@st.cache_resource
def load_models():
    device = torch.device('cpu')
    face_model = FaceEmotionModel()
    face_model.load_state_dict(torch.load('face_model.pth', map_location=device))
    face_model.eval()
    audio_model = AudioEmotionLSTM()
    audio_model.load_state_dict(torch.load('audio_model.pth', map_location=device))
    audio_model.eval()
    return face_model, audio_model

face_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def detect_face_emotion(frame, face_model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, None, frame
    x, y, bw, bh = faces[0]
    face = frame[y:y+bh, x:x+bw]
    if face.size == 0:
        return None, None, frame
    tensor = face_transform(face).unsqueeze(0)
    with torch.no_grad():
        output = face_model(tensor)
        probs = torch.softmax(output, dim=1)
        idx = probs.argmax().item()
        confidence = probs.max().item()
    emotion = FACE_EMOTIONS[idx]
    cv2.rectangle(frame, (x, y), (x+bw, y+bh), (30, 30, 30), 2)
    cv2.putText(frame, f"{emotion} {confidence:.0%}",
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)
    return emotion, confidence, frame

audio_result_queue = queue.Queue()

def record_and_predict(audio_model):
    while True:
        audio = sd.rec(int(AUDIO_DURATION * SAMPLE_RATE),
                      samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=40)
        max_len = 130
        if mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, ((0,0),(0, max_len - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :max_len]
        tensor = torch.FloatTensor(mfcc).unsqueeze(0)
        with torch.no_grad():
            output = audio_model(tensor)
            idx = output.argmax().item()
        audio_result_queue.put(AUDIO_EMOTIONS[idx])

EMOTION_MAP = {
    'angry': 'Angry', 'disgust': 'Disgust', 'fear': 'Fearful',
    'happy': 'Happy', 'neutral': 'Neutral', 'sad': 'Sad', 'surprise': 'Surprised',
}

def fuse_emotions(face_emotion, audio_emotion):
    if face_emotion is None and audio_emotion is None:
        return "Awaiting signal", False
    if face_emotion is None:
        return audio_emotion, False
    if audio_emotion is None:
        return face_emotion, False
    face_mapped = EMOTION_MAP.get(face_emotion.lower(), face_emotion)
    if face_mapped.lower() == audio_emotion.lower():
        return face_mapped, True
    return f"{face_emotion} / {audio_emotion}", False

# ─── STYLING ──────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: #F9F9F7;
    }

    /* Hide streamlit default elements */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        padding: 1.5rem 2rem;
        max-width: 960px;
    }

    /* App header */
    .app-header {
        display: flex;
        align-items: baseline;
        gap: 12px;
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid #E5E5E3;
    }
    .app-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1A1A1A;
        letter-spacing: -0.03em;
    }
    .app-subtitle {
        font-size: 0.85rem;
        color: #999;
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    /* Cards */
    .card {
        background: #FFFFFF;
        border: 1px solid #E5E5E3;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .card-label {
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #999;
        margin-bottom: 0.75rem;
    }

    /* Emotion display */
    .emotion-value {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1A1A1A;
        letter-spacing: -0.03em;
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    .emotion-sub {
        font-size: 0.8rem;
        color: #999;
    }

    /* Fused output */
    .fused-card {
        background: #1A1A1A;
        border: none;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .fused-label {
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #666;
        margin-bottom: 0.75rem;
    }
    .fused-value {
        font-size: 1.8rem;
        font-weight: 600;
        color: #FFFFFF;
        letter-spacing: -0.04em;
        line-height: 1;
    }
    .fused-confirmed {
        display: inline-block;
        font-size: 0.7rem;
        font-weight: 500;
        color: #4CAF50;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-top: 0.5rem;
    }

    /* Confidence bar */
    .conf-bar-bg {
        height: 3px;
        background: #F0F0EE;
        border-radius: 2px;
        margin-top: 0.75rem;
    }
    .conf-bar-fill {
        height: 3px;
        background: #1A1A1A;
        border-radius: 2px;
        transition: width 0.3s ease;
    }

    /* Status dot */
    .status-row {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-top: 0.5rem;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #4CAF50;
    }
    .status-dot.inactive {
        background: #CCC;
    }
    .status-text {
        font-size: 0.75rem;
        color: #999;
    }

    /* Stop button */
    .stButton button {
        background: #F0F0EE !important;
        color: #1A1A1A !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.25rem !important;
        letter-spacing: 0.01em !important;
        transition: background 0.2s !important;
    }
    .stButton button:hover {
        background: #E5E5E3 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ─── MAIN ─────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Emotive", layout="wide", initial_sidebar_state="collapsed")
    inject_css()

    # Header
    st.markdown("""
    <div class="app-header">
        <span class="app-title">Emotive</span>
        <span class="app-subtitle">Real-time multimodal emotion recognition</span>
    </div>
    """, unsafe_allow_html=True)

    face_model, audio_model = load_models()

    # Layout
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown('<div class="card-label">Live Feed</div>', unsafe_allow_html=True)
        frame_placeholder = st.empty()

    with col2:
        st.markdown('<div class="card-label" style="margin-bottom:0.75rem">Face</div>', unsafe_allow_html=True)
        face_emotion_placeholder = st.empty()

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

        st.markdown('<div class="card-label" style="margin-bottom:0.75rem">Voice</div>', unsafe_allow_html=True)
        audio_emotion_placeholder = st.empty()

        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

        st.markdown('<div class="card-label" style="margin-bottom:0.75rem">Fused Output</div>', unsafe_allow_html=True)
        fused_placeholder = st.empty()

        st.markdown('<div style="height:1.5rem"></div>', unsafe_allow_html=True)
        stop = st.button("Stop session")

    # Start audio thread
    audio_thread = threading.Thread(target=record_and_predict, args=(audio_model,), daemon=True)
    audio_thread.start()

    cap = cv2.VideoCapture(0)
    current_audio_emotion = None
    current_face_confidence = 0.0

    while not stop:
        ret, frame = cap.read()
        if not ret:
            st.error("Cannot access webcam.")
            break

        result = detect_face_emotion(frame, face_model)
        face_emotion, face_confidence, annotated_frame = result

        try:
            current_audio_emotion = audio_result_queue.get_nowait()
        except queue.Empty:
            pass

        fused, confirmed = fuse_emotions(face_emotion, current_audio_emotion)

        # Webcam feed
        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Face card
        face_conf_pct = int((face_confidence or 0) * 100)
        face_emotion_placeholder.markdown(f"""
        <div class="card">
            <div class="emotion-value">{face_emotion or "—"}</div>
            <div class="emotion-sub">{"Detected" if face_emotion else "No face detected"}</div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{face_conf_pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Audio card
        audio_emotion_placeholder.markdown(f"""
        <div class="card">
            <div class="emotion-value">{current_audio_emotion or "—"}</div>
            <div class="status-row">
                <div class="status-dot {'inactive' if not current_audio_emotion else ''}"></div>
                <span class="status-text">{"Analyzing 3s windows" if not current_audio_emotion else "Active"}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Fused card
        confirmed_badge = '<div class="fused-confirmed">Both modalities agree</div>' if confirmed else ''
        fused_placeholder.markdown(f"""
        <div class="fused-card">
            <div class="fused-value">{fused}</div>
            {confirmed_badge}
        </div>
        """, unsafe_allow_html=True)

        time.sleep(0.03)

    cap.release()

if __name__ == "__main__":
    main()
