import streamlit as st
import speech_recognition as sr
import pyttsx3
import torch
from huggingface_hub import login
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM

# Hugging Face login
hugging_face_token_id = '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'
login(hugging_face_token_id)

# Load models
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-base")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

tts_engine = pyttsx3.init()
recognizer = sr.Recognizer()

# Store conversation history
conversation_history = []

# Function for generating responses
def generate_response(input_text):
    conversation_history.append(f"User: {input_text}")
    history_text = "\n".join(conversation_history[-5:])
    inputs = llama_tokenizer(history_text, return_tensors="pt")
    outputs = llama_model.generate(inputs["input_ids"], max_new_tokens=50)
    response_text = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    conversation_history.append(f"Bot: {response_text}")
    return response_text

# Streamlit UI
st.title("Speech-to-Speech Chatbot")
st.write("Choose your input method:")

# Input methods
input_method = st.radio("Select an option", ("Text Input", "Upload Audio"))

if input_method == "Text Input":
    user_input = st.text_input("Enter your message:")
    if user_input:
        response = generate_response(user_input)
        st.write("Bot:", response)

        # Optionally use TTS to speak the response
        tts_engine.say(response)
        tts_engine.runAndWait()

elif input_method == "Upload Audio":
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file:
        audio_data = recognizer.record(uploaded_file)
        inputs = whisper_processor(audio_data, return_tensors="pt", sampling_rate=16000)
        outputs = whisper_model.generate(inputs["input_ids"])
        text = whisper_processor.decode(outputs[0], skip_special_tokens=True)
        st.write("You said:", text)
        
        response = generate_response(text)
        st.write("Bot:", response)
        
        tts_engine.say(response)
        tts_engine.runAndWait()
