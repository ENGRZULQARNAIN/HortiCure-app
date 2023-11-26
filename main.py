import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import os
import base64
from chat_bot import chat


#Streamlit page setup and configuration.
st.set_page_config(page_title="HortiCure🍀",page_icon="✅",layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
.block-container {
  padding-top: 0.5rem;
}
.sidebar .sidebar-content {
  padding-top: 0;
}

.sidebar .sidebar-header {
  position: absolute;
  top: 0;
  width: 100%;
  background-color: #fff;
  z-index: 1;
}
</style>
""", unsafe_allow_html=True)



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
        st.sidebar.header("Detect your Disease here🔎")


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
        st.sidebar.success(f"HEALTH STATUS: {label}")
        st.sidebar.error(f"CONFIDENCE: {int(confidence)}%")

    ####chatbot-Section#######
    def chat_bot_interface(self):
        #palm_chat()
        for i in range(5):
             st.header("This area is under development!")
    

    
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


