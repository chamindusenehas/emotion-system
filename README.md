# 📘 Emotion System

A real-time multimodal emotion detection system that recognizes human emotions by analyzing both **visual input (facial expressions)** and **audio input (speech)**. Combines camera and microphone signals for robust emotion predictions.

---

## 🚀 Features

- Real-time webcam face emotion recognition  
- Continuous audio recording and speech emotion analysis  
- Confidence-aware fusion of visual and audio predictions  
- Temporal smoothing of final emotion outputs  

---


## 📦 Installation

1. **Clone the repo**  
   ```bash
    git clone https://github.com/chamindusenehas/emotion-system.git
    cd emotion-system

2. **Create a virtual environment (optional)**
    ```bash
        python -m venv venv
        venv\Scripts\activate  # Windows
        source venv/bin/activate  # macOS/Linux

3. **Install dependencies**
    ```bash
        pip install -r requirements.txt

# 🔧 How It Works

**🎥 Visual Emotion Detection**

- Captures frames from webcam

- Detects faces

- Classifies facial emotion

**🎤 Audio Emotion Detection**

- Continuously records audio

- Detects speech segments

- Maps audio emotion labels to visual emotion space

**🤝 Fusion Logic**

- Combines visual and audio predictions

- Weights each modality based on confidence

- Produces a smoothed final emotion

## 📌 Running the System

Run the main script:

    python main.py


Example console output:

        Visual: neutral (0.82), Audio: happy (0.59), Fused: happy (0.72)

Press q to exit.