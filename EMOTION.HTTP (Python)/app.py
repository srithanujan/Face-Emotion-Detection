from fastapi import FastAPI, File, UploadFile, Form
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import uvicorn
import os
import io

# Setup Global Variables
emotion_pred_model = None
emotion_pred_model_file = './models/emotion_model_face.keras'

# Define constants
IMG_HEIGHT = 48
IMG_WIDTH = 48
emotion_face_labels = {
    'ANGRY': 0,
    'DISGUSTED': 1,
    'FEARFUL': 2,
    'HAPPY': 3,
    'NEUTRAL': 4,
    'SAD': 5,
    'SURPRISED': 6
}

def setup_models():
    global emotion_pred_model
    if emotion_pred_model is None:
        if os.path.exists(emotion_pred_model_file):
            # Load the model
            emotion_pred_model = load_model(emotion_pred_model_file, compile=True)
            print("Model loaded successfully.")
        else:
            print(f"Model file {emotion_pred_model_file} not found.")

def predict_emotion_func(image_file):
    try:
        image_content = image_file.file.read()
        image_stream = io.BytesIO(image_content)
        
        img = load_img(image_stream, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = emotion_pred_model.predict(img_array)
        predicted_emotion = np.argmax(predictions, axis=1)
        return predicted_emotion[0]
    except Exception as e:
        print(f"Error in predict_emotion_func: {e}")
        return ''

def pred_face_emotion(image_file):
    pred_face_emotion = predict_emotion_func(image_file)
    if pred_face_emotion == '':
        return '', {'message': 'face function failed!!!'}, 400
    index_to_emotion = {v: k for k, v in emotion_face_labels.items()}
    emotion_name = index_to_emotion[pred_face_emotion]
    return emotion_name, '', 200

setup_models()
app = FastAPI(docs_url=None, redoc_url=None)

@app.get('/info')
def index():
    return {'result': {'message': 'Emotion Detection Service'}, 'code': 200, 'error': ''}

@app.get('/health')
def health_check():
    return {'result': {'message': 'Emotion Detection Service'}, 'code': 200, 'error': ''}

@app.post('/detect/emotion/face')
def predict_sides_faces(user: str = Form(...), file: UploadFile = File(...)):
    if not user or not file:
        return {'result': '', 'code': 400, 'error': {'message': 'Detection failed! Please send proper request.'}}
    result, error, status = pred_face_emotion(file)
    return {'result': result, 'code': status, 'error': error}

if __name__ == '__main__':
    uvicorn.run(app, host=HOST, port=PORT)


