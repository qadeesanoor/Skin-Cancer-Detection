from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import io
 
app = FastAPI()
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
model = tf.keras.models.load_model(
    '../model/best_skin_cancer_model.keras'
)
 
classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
 
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB').resize((224, 224))
    x = np.array(img, dtype=np.float32)
    x = preprocess_input(x)          # scales to [-1, 1], matches training
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)[0]
    idx = int(preds.argmax())
    return {
        "class": classes[idx],
        "confidence": float(preds[idx])
    }
 
@app.get("/")
def root():
    return {"status": "running"}