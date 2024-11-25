from sentence_transformers import SentenceTransformer
import streamlit as st


# Initialize the model and Pinecone
model = SentenceTransformer('all-MiniLM-L6-v2')

import os
from pinecone import Pinecone, ServerlessSpec


from dotenv import load_dotenv

load_dotenv()

# Create an instance of Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if the index exists, create if necessary
if 'test3' not in pc.list_indexes().names():
    pc.create_index(
        name='test3', 
        dimension=1536, 
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

# Access the index
index = pc.Index('test3')

def find_match(input):
    input_em = model.encode(input).tolist()
    # Use keyword arguments for the query method
    result = index.query(vector=input_em, top_k=2, include_metadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string

#