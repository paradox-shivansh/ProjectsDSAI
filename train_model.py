"""
Training script to retrain the model and preprocessor with current scikit-learn version
"""
import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logger

def main():
    try:
        logger.info("Data Ingestion started")
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        
        logger.info("Data Transformation started")
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path=train_data_path,
            test_path=test_data_path
        )
        
        logger.info("Model Training started")
        model_trainer = ModelTrainer()
        r2_score = model_trainer.initiate_model_trainer(
            train_array=train_arr,
            test_array=test_arr,
            preprocessor_path=preprocessor_path
        )
        
        logger.info(f"Model training completed with R2 score: {r2_score}")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
