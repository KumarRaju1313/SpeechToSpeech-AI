# 🎤 Speech-to-Speech LLM Bot

A real-time voice-based chatbot that listens to your voice, understands your question using **LLaMA 2**, and talks back using **text-to-speech** — creating an interactive assistant experience.

---

## 🔍 What This Project Does

- 🎙️ Takes input from:
  - Microphone
  - Webcam (audio)
  - OR Manual text
- 🎧 Converts voice to text using **OpenAI Whisper**
- 🧠 Understands your message using **Meta LLaMA 2** (via Hugging Face Transformers)
- 🗣️ Speaks the response using **pyttsx3** (text-to-speech)
- 🪄 Feels like you’re talking to a smart assistant!

---

## 🧠 Technologies Used

| Function               | Tool/Library                |
|------------------------|-----------------------------|
| Speech Recognition     | `SpeechRecognition`, `Whisper` |
| Natural Language Model | `LLaMA 2` via `transformers` |
| Text-to-Speech         | `pyttsx3`                   |
| Audio Input Handling   | `OpenCV`, `pyaudio`         |
| Model Hosting          | `Hugging Face`              |
| Interface              | Python CLI                  |

---

## ⚙️ How to Set It Up

> ⚠️ This project must be run **locally** — microphones and webcams are not supported in Google Colab. However, **text mode** will still work there.  
> ℹ️ *This is a work in progress and not yet fully optimized.*

---

### 🔧 Install Dependencies

Copy and paste this into your terminal:

```bash
pip install SpeechRecognition
pip install soundfile
pip install opencv-python-headless
pip install torch
pip install transformers
pip install huggingface_hub
pip install pyttsx3

# OS-specific dependencies (for TTS and mic access)
sudo apt install espeak
sudo apt-get install portaudio19-dev
pip install pyaudio
