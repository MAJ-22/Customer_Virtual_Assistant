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
import json
from gtts import gTTS
from deep_translator import GoogleTranslator
from utils import *
from dotenv import load_dotenv

load_dotenv()

# Set your Groq API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

st.subheader("V-Serve")
# Supported languages configuration
SUPPORTED_LANGUAGES = {
    'English': {'code': 'en', 'vosk_model': 'vosk-model-small-en-us-0.15'},
    'Hindi': {'code': 'hi', 'vosk_model': 'vosk-model-small-hi-0.22'},
    'Spanish': {'code': 'es', 'vosk_model': 'vosk-model-small-es-0.42'},
    'French': {'code': 'fr', 'vosk_model': 'vosk-model-small-fr-0.22'},
    'German': {'code': 'de', 'vosk_model': 'vosk-model-small-de-0.15'},
    # Add more languages as needed
}

# Initialize session states
if 'language' not in st.session_state:
    st.session_state['language'] = 'English'
if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]
if 'requests' not in st.session_state:
    st.session_state['requests'] = []
if 'mode' not in st.session_state:
    st.session_state['mode'] = 'type'
if 'voice_output' not in st.session_state:
    st.session_state['voice_output'] = False

# Initialize voice models
@st.cache_resource
def load_vosk_models():
    models = {}
    for lang, details in SUPPORTED_LANGUAGES.items():
        model_path = f"models/{details['vosk_model']}"
        if os.path.exists(model_path):
            models[lang] = Model(model_path)
    return models

# Initialize translation
def translate_text(text, source_lang='auto', target_lang='en'):
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)
        return translator.translate(text)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# Function to record and transcribe audio
def get_audio_input(vosk_models):
    try:
        status_placeholder = st.empty()
        status_placeholder.info(f"Recording... Speak in {st.session_state['language']}")
        
        duration = 5
        sample_rate = 16000
        audio = sd.rec(int(duration * sample_rate),
                      samplerate=sample_rate,
                      channels=1,
                      dtype=np.int16)
        sd.wait()
        status_placeholder.empty()
        
        # Get appropriate model for selected language
        current_model = vosk_models.get(st.session_state['language'])
        if not current_model:
            st.error(f"Model for {st.session_state['language']} not found")
            return None
            
        rec = KaldiRecognizer(current_model, sample_rate)
        rec.AcceptWaveform(audio.tobytes())
        result = json.loads(rec.FinalResult())
        
        transcription = result.get('text', '')
        
        if transcription:
            # Translate to English for processing if not already in English
            if st.session_state['language'] != 'English':
                lang_code = SUPPORTED_LANGUAGES[st.session_state['language']]['code']
                english_text = translate_text(transcription, lang_code, 'en')
                st.info(f"Original: {transcription}")
                st.success(f"Translated: {english_text}")
                return english_text
            else:
                st.success(f"You said: {transcription}")
                return transcription
        else:
            st.warning("No speech detected. Please try again.")
            return None
            
    except Exception as e:
        st.error(f"Error during voice input: {e}")
        return None

# Function for text-to-speech using gTTS
def speak_response(text, language):
    try:
        # Translate response if not in user's selected language
        if language != 'English':
            lang_code = SUPPORTED_LANGUAGES[language]['code']
            text = translate_text(text, 'en', lang_code)
            
        output_file = "response.mp3"
        tts = gTTS(text=text, lang=SUPPORTED_LANGUAGES[language]['code'])
        tts.save(output_file)
        
        st.audio(output_file)
        
        if os.path.exists(output_file):
            os.remove(output_file)
    except Exception as e:
        st.error(f"Error during speech synthesis: {e}")

# Sidebar controls
with st.sidebar:
    st.title("Settings")
    
    # Language selector
    st.session_state['language'] = st.selectbox(
        "Select Language",
        options=list(SUPPORTED_LANGUAGES.keys()),
        index=list(SUPPORTED_LANGUAGES.keys()).index(st.session_state['language'])
    )
    
    # Mode toggle
    if st.button("Switch to " + ("Speak" if st.session_state['mode'] == 'type' else "Type")):
        st.session_state['mode'] = 'speak' if st.session_state['mode'] == 'type' else 'type'
    
    # Voice output toggle
    st.session_state['voice_output'] = st.checkbox("Enable Voice Output", 
                                                 value=st.session_state['voice_output'])

# Initialize models
try:
    vosk_models = load_vosk_models()
    llm = ChatGroq(model="llama3-8b-8192")
except Exception as e:
    st.error(f"Failed to initialize models: {e}")

# Set up conversation chain
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""You are Emma, a friendly and professional customer service representative at our company. Your role is to assist customers with their inquiries in a natural, conversational manner.

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
You: "I apologize, but I'm specifically here to help with customer service matters related to our products and services. I'd be happy to assist you with questions about our return policy, product features, or account management instead. What can I help you with?
""")

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

# Handle input based on mode
if st.session_state['mode'] == 'speak':
    if st.button("🎤 Start Recording"):
        query = get_audio_input(vosk_models)
else:
    query = st.text_input("Query: ", key="input")
    # Translate query to English if not in English
    if query and st.session_state['language'] != 'English':
        lang_code = SUPPORTED_LANGUAGES[st.session_state['language']]['code']
        query = translate_text(query, lang_code, 'en')

# Process query and generate response
if query:
    with st.spinner("Processing..."):
        context = find_match(query)
        response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        
        if st.session_state['voice_output']:
            speak_response(response, st.session_state['language'])
        
        # Translate response if not in English
        if st.session_state['language'] != 'English':
            lang_code = SUPPORTED_LANGUAGES[st.session_state['language']]['code']
            displayed_response = translate_text(response, 'en', lang_code)
        else:
            displayed_response = response
            
        st.session_state.requests.append(query)
        st.session_state.responses.append(displayed_response)

# Display chat history
with st.container():
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')