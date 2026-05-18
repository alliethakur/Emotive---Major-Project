# Emotive — Real-Time Multimodal Emotion Recognition

Real-time emotion recognition from live webcam and microphone using EfficientNet-B0 (face) and Bidirectional LSTM (audio) with feature-level fusion.

## Authors
CMR Institute of Technology, Bengaluru 

[Allie Thakur], [Kenesha Watt Q], [Syed Anshu], [Dileep M N]

Guide: [Ms. Sandhya Kumari]

## Run the App
```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## Models
| Model | Dataset | Accuracy |
|---|---|---|
| EfficientNet-B0 | FER-2013 | 65.81% |
| Bidirectional LSTM | RAVDESS | 59.03% |

## Base Paper
Salas-Caceres et al., Springer Multimedia Tools and Applications, 2024.
