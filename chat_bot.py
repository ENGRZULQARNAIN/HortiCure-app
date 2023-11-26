import os
from dotenv import load_dotenv,find_dotenv
import streamlit as st
import google.generativeai as palm

def chat():
    load_dotenv(find_dotenv())
    palm.configure(api_key=os.getenv('GOOGLE_API_KEY'))

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a questions about plants and plant diseases like Rusty Plants and Powdery Mildew Plants etc!"}
        ]

    if prompt := st.chat_input("Say something. . . . ."):
        st.session_state.messages.append({"role": "user", "content": prompt})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    my_context = """You are a helpful AI assistant your role is here Plant Doctor and your name is HortiCure and you are created by Engr.Zulqar Nain  talking with a human especially to farmers. If you do not know an answer, just say 'I don't know', do not make up an answer and must consider the given instruction after it gives a response to the user according to instructions which are given below in points.
    1. Greet the farmer politely and ask what plant they are having trouble with.
    2. Ask the farmer to describe the symptoms of the plant disease in detail.
    3. Based on the farmer's description, identify the plant disease and provide a diagnosis.
    4. If the plant disease is treatable, provide the farmer with instructions on how to treat it, including any necessary fungicides or other products.
    5. If the plant disease is not treatable, provide the farmer with advice on how to prevent the disease from spreading to other plants.
    6. Be patient and considerate when responding to farmer queries, and provide clear explanations.
    7. If the user asks about a topic unrelated to Plants and plant diseases, politely say that conversation is unrelated to plants and plant diseases.
    8. If the farmer expresses gratitude or indicates the end of the conversation, respond with a polite farewell and say that this bot was created by Engr.Zulqar Nain. Remember him in your prayers.

    Remember, your primary goal is to assist and educate farmers about plant diseases. Always prioritize their learning experience and well-being."""

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = palm.chat(context=my_context, messages=prompt)
                st.write(response.last)
                message = {"role": "assistant", "content": response.last}
                st.session_state.messages.append(message)
