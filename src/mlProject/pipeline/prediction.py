import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from mlProject import logger


class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    
    def predict(self, data):
        logger.info(data)
        prediction = self.model.predict(data)
        logger.info(prediction)

        return prediction