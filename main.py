from fastapi import FastAPI, File, UploadFile

from data_model import PredictionResult, ImageModel
from inference import predict_image_model

app = FastAPI()


@app.get("/")
def health():
    return {"status": "OK"}


@app.post("/predict_image/", response_model=PredictionResult, status_code=200)
def predict_image(file: UploadFile = File(...),
                  model: ImageModel = ImageModel.first_custom_cnn.value):
    image = file.filename
    prediction_class = predict_image_model(model, image)
    return PredictionResult(prediction_class=prediction_class, image_name=image)
