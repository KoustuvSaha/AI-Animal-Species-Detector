from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io, json, numpy as np, tensorflow as tf

model = tf.keras.models.load_model('animal_species_model.h5')
app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).resize((224, 224))
    image = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(image)
    label = np.argmax(prediction)
    confidence = np.max(prediction)
    return json.dumps({'label': int(label), 'confidence': float(confidence)})