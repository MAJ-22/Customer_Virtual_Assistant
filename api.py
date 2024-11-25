import getpass
import os

os.environ["GROQ_API_KEY"] = getpass.getpass()

from langchain_groq import ChatGroq

model = ChatGroq(model="llama3-8b-8192")