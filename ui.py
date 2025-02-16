import gradio as gr
import numpy as np, tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model('animal_species_model.h5')

def classify_image(image):
    image = np.resize(image, (224, 224, 3))
    image = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(image)
    label = np.argmax(prediction)
    classes = ['Cat', 'Dog', 'Wildlife']
    return f"Predicted Label: {classes[label]}, Confidence: {np.max(prediction):.2f}"

demo = gr.Interface(fn=classify_image, inputs="image", outputs="text")
demo.launch()