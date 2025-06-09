# 🎤 Speech-to-Speech LLM Bot

This is a voice-based chatbot project where users can talk to a bot, and it talks back! It listens to your voice using your microphone or webcam, understands your message using a powerful language model, and responds out loud — all in real time.

This project combines **speech recognition**, **natural language processing**, and **text-to-speech** in a single pipeline using state-of-the-art models like **Whisper** and **LLaMA 2**.

---

## 🔍 What this project does

- Takes input from the **microphone**, **webcam (audio)**, or **manual text**
- Converts your voice into text using **Whisper**
- Uses **LLaMA 2** to generate a smart and relevant reply
- Converts that reply into voice using **pyttsx3**
- Plays the response back to you like a real conversation

This makes it feel like you’re talking to an actual assistant.

---

## 🧠 Technologies used

| Function              | Tool/Library                      |
|-----------------------|-----------------------------------|
| Speech Recognition    | OpenAI Whisper                    |
| Natural Language Model| Meta LLaMA 2                      |
| Text-to-Speech        | pyttsx3                           |
| Interface             | Python CLI (Command Line)         |
| Audio Input Handling  | SpeechRecognition, OpenCV         |
| Model Hosting         | Hugging Face                      |

---

## ⚙️ How to set it up

> ⚠️ This project needs to be run **locally**, not in Google Colab — because Colab doesn't support microphones or webcams properly.

### 🔧 Install dependencies

You can copy-paste these into your terminal (Ubuntu/Debian):

```bash
pip install SpeechRecognition
pip install soundfile
pip install opencv-python-headless
pip install torch
pip install transformers
pip install huggingface_hub
pip install pyttsx3

sudo apt install espeak
sudo apt-get install portaudio19-dev
pip install pyaudio
