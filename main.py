import os
import streamlit as st
from streamlit_chat import message
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from vosk import Model, KaldiRecognizer
import pyttsx3
import json
from utils import *
import os
from dotenv import load_dotenv

load_dotenv()

# Set your Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

st.subheader("V-Serve")

# Initialize session states
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'mode' not in st.session_state:
    st.session_state['mode'] = 'type'

if 'voice_output' not in st.session_state:
    st.session_state['voice_output'] = False

# Initialize Vosk model
@st.cache_resource
def load_vosk_model():
    # Download model from https://alphacephei.com/vosk/models
    # and unpack as 'model' in the current folder
    model = Model("model")
    return model

# Initialize TTS engine
@st.cache_resource
def init_tts_engine():
    engine = pyttsx3.init()
    return engine

try:
    vosk_model = load_vosk_model()
    tts_engine = init_tts_engine()
except Exception as e:
    st.error(f"Failed to initialize voice models: {e}")

# Initialize GroqChatModel
try:
    llm = ChatGroq(model="llama3-8b-8192")
except Exception as e:
    st.error(f"Failed to initialize Groq Chat Model: {e}")

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Your existing system message template and conversation setup remains the same
system_msg_template = SystemMessagePromptTemplate.from_template(template=""" You are Emma, a friendly and professional customer service representative at our company. Your role is to assist customers with their inquiries in a natural, conversational manner.

Essential Guidelines:
1. ALWAYS maintain a warm, empathetic, and human-like tone in your responses
2. Use natural language and occasional conversational expressions like "I understand how frustrating this must be" or "I'd be happy to help you with that"
3. ONLY answer questions that are related to the customer service information provided in the knowledge base
4. For any questions outside the scope of the provided customer service database:
   - Politely apologize
   - Explain that you can only assist with customer service-related inquiries
   - Guide them back to relevant topics
   - Example: "I apologize, but I can only assist with customer service-related questions about our products and services. Could you please let me know if you have any questions about [mention 2-3 relevant topics from your database]?"

Response Style:
- Begin responses with friendly greetings when appropriate
- Use "I" statements to sound more personal
- Show active listening by briefly acknowledging the customer's concern
- Keep responses clear and concise
- End interactions professionally and warmly

Remember:
- Never make up information
- Never attempt to answer questions outside your customer service knowledge base
- Always stay within the scope of your training data
- If unsure, ask for clarification rather than making assumptions

Example interaction:
Customer: "What's the weather like today?"
You: "I apologize, but I'm specifically here to help with customer service matters related to our products and services. I'd be happy to assist you with questions about our return policy, product features, or account management instead. What can I help you with?""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([
    system_msg_template,
    MessagesPlaceholder(variable_name="history"),
    human_msg_template
])

conversation = ConversationChain(
    memory=st.session_state.buffer_memory,
    prompt=prompt_template,
    llm=llm,
    verbose=True
)

# Function to record and transcribe audio using Vosk
def get_audio_input():
    try:
        # Create placeholder for recording status
        status_placeholder = st.empty()
        status_placeholder.info("Recording... Speak now!")
        
        # Record audio
        duration = 5  # seconds
        sample_rate = 16000
        audio = sd.rec(int(duration * sample_rate),
                      samplerate=sample_rate,
                      channels=1,
                      dtype=np.int16)
        sd.wait()
        status_placeholder.empty()
        
        # Process with Vosk
        rec = KaldiRecognizer(vosk_model, sample_rate)
        rec.AcceptWaveform(audio.tobytes())
        result = json.loads(rec.FinalResult())
        
        transcription = result.get('text', '')
        
        if transcription:
            st.success(f"You said: {transcription}")
            return transcription
        else:
            st.warning("No speech detected. Please try again.")
            return None
            
    except Exception as e:
        st.error(f"Error during voice input: {e}")
        return None

# Function for text-to-speech using pyttsx3
def speak_response(text):
    try:
        output_file = "response.wav"
        tts_engine.save_to_file(text, output_file)
        tts_engine.runAndWait()
        
        if os.path.exists(output_file):
            st.audio(output_file)
            os.remove(output_file)
    except Exception as e:
        st.error(f"Error during speech synthesis: {e}")

# Container for chat history
response_container = st.container()

# Sidebar controls
with st.sidebar:
    st.title("Settings")
    # Button to toggle between typing and speaking modes
    if st.button("Switch to " + ("Speak" if st.session_state['mode'] == 'type' else "Type")):
        st.session_state['mode'] = 'speak' if st.session_state['mode'] == 'type' else 'type'
    
    # Toggle for voice output
    st.session_state['voice_output'] = st.checkbox("Enable Voice Output", value=st.session_state['voice_output'])

# Handle input based on the selected mode
if st.session_state['mode'] == 'speak':
    if st.button("🎤 Start Recording"):
        query = get_audio_input()
else:
    query = st.text_input("Query: ", key="input")

# Handle the query submission
if query:
    with st.spinner("Processing..."):
        context = find_match(query)
        response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        
        # Generate voice response if enabled
        if st.session_state['voice_output']:
            speak_response(response)
        
        # Append to session state
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

# Display chat history
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')