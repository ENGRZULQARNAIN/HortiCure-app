import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import google.generativeai as palm
from langchain.llms import google_palm
import os
from dotenv import load_dotenv
import base64


#Streamlit page setup and configuration.
st.set_page_config(page_title="HortiCure🍀",page_icon="✅",layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {
  padding-top: 0.5rem;
}
.sidebar-header {
  position: fixed;
  top:;
  left: ;
  right: 5;
  z-index: 0;
}
</style>
""", unsafe_allow_html=True)


import google.generativeai as palm
import streamlit as st

def palm_chat():
    load_dotenv()
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







#class of My Model which consist of related things to my model loading predicting etc..
class MyModel:
    def __init__(self) -> None:
        pass

    # Load the saved model using tf.keras.models.load_model
    def load_saved_model(self, model_path="./model"):
        model = tf.keras.models.load_model(model_path)
        return model

    # Your single_predict function
    def single_predict(self, model, image):
        classes = ["HEALTHY ✅", "POWDERY ❎", "RUST ❎"]

        if image is not None:
            # Convert the PIL Image to a NumPy array
            image = np.array(image)
            image = image / 255.0  # Normalize the image
            input_arr = np.array([image])

            # Predict using the model
            predictions = model.predict(input_arr)

            index = np.argmax(predictions)
            confidence = np.max(predictions)

            return classes[index], confidence * 100
        else:
            return "Still No Image uploaded", 0.0



#This ou UI class which is consist of related things to UI 
# ...

class Ui:
    def __init__(self):
        st.title("🌿HortiCure🍀")

    def ui_image_load(self):
        st.sidebar.title("Disease Detection 🔎")


        img_file = st.sidebar.file_uploader("Upload your image here :floppy_disk:", type=["jpg", "png"])
        if img_file is not None:
            image = self.read_file_as_image(img_file)  # Use self here
            st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
            return image


    def read_file_as_image(self, data):
        image = Image.open(data)

        if image.mode == 'RGBA':
            # Convert RGBA to RGB
            image = image.convert('RGB')

        image = image.resize((256, 256))  # Resize the image to the expected dimensions
        image = np.array(image)
        return image 
    def show_result(self,label,confidence):
        st.sidebar.header(f"HEALTH STATUS: {label}")
        st.sidebar.header(f"CONFEDINCE: {int(confidence)}%")

    ####chatbot-Section#######
    def chat_bot_interface(self):
        palm_chat()  
    

    
if __name__ == "__main__":
    # class instances
    model_obj = MyModel()
    ui = Ui()

    # to load the saved model
    model = model_obj.load_saved_model()

    # to load the image from the source
    imagee = ui.ui_image_load()

    # to predict the loaded image....
    label, confidence = model_obj.single_predict(model, imagee)
    ui.show_result(label, confidence)

    # Run the chat_bot function
    ui.chat_bot_interface()

    st.sidebar.write("Powerd by Zulqar Nain")
    
    st.write("Copyright © @zulqarnainhumbly258@gmail.com")


