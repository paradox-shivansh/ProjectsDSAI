from flask import Flask,request, render_template
from src.pipeline.predict_pipeline import PredictPipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from src.logger import logger
from src.exception import CustomException


application = Flask(__name__)

app = application
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

