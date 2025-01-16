import os
import logging
import time
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
import torch
from google.cloud.aiplatform.prediction.predictor import Predictor
from google.cloud.aiplatform.utils import prediction_utils
import base64
from PIL import Image
import io
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

IMAGE_BUCKET = "jk-sandbox_temp"

def get_bucket():
    from google.cloud import storage
    storage_client = storage.Client()
    return storage_client.bucket(IMAGE_BUCKET)

class VAIGroundingDINO(Predictor):
    def __init__(self):
        return

    def load(self, artifacts_uri: str) -> None:
        # Load model
        logger.info(f"Starting predictor using {artifacts_uri}")
        self.bucket = get_bucket()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        origin_path = os.getcwd()
        model_path = f"{origin_path}/model"
        os.makedirs(model_path)
        os.chdir(model_path)
        prediction_utils.download_model_artifacts(artifacts_uri)
        os.chdir(origin_path)
        logger.debug('Start model loading...')
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self.device)
        logger.debug('Transformer model loaded successfully')

    def predict(self, prediction_input):
        print("Start predicting")
        start_time = time.time()
        #raw_image = base64.b64decode(prediction_input["instances"][0]['image'])
        #image = Image.open(raw_image).convert("RGB")
        blob = self.bucket.get_blob(prediction_input["instances"][0]['image_uri'])
        image = Image.open(io.BytesIO(blob.download_as_bytes())).convert("RGB")
        prompt = prediction_input["instances"][0]['prompt']
        box_threshold = prediction_input["instances"][0]['box_threshold']
        text_threshold = prediction_input["instances"][0]['text_threshold']
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, box_threshold=box_threshold, 
            text_threshold=text_threshold, target_sizes=[image.size[::-1]])
        logger.info("--- %s seconds ---" % (time.time() - start_time))
        return {"predictions": [{"boxes": results[0]['boxes'].tolist(),
                                 "labels": results[0]['labels'],
                                 "scores": results[0]['scores'].tolist()}]}