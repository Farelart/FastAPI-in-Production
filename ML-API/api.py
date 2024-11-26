from fastapi import FastAPI, UploadFile
from tensorflow.keras.models import load_model
import numpy as np
import io
from PIL import Image

app = FastAPI()

@app.get("/")
def greeting():
    return {"message": "Hello, World!"}

def load():
    model_path = "best_model.h5"
    model = load_model(model_path, compile=False)
    return model

model = load()

def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img



@app.post("/predict")
async def predict(file: UploadFile):
    image_data = await file.read()

    #open image
    img = Image.open(io.BytesIO(image_data))

    #preprocess image
    img = preprocess(img)

    #predict
    predictions = model.predict(img)
    rec = predictions[0][0].tolist()

    return {"predictions": rec}