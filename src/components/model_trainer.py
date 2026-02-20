from catboost import CatBoostRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

import os
import sys
from src.logger import logger
from src.exception import CustomException

from src.utils import save_object,evaluate_models
from dataclasses import dataclass
from src.components.data_transformation import DataTransformationConfig


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array,preprocessor_path):
        try:
            logger.info("Split training and testing input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False)
            }
            
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2']
                },
                "Random Forest": {
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting": {
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2']
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    'n_neighbors': [5,7,9,11],
                    'weights': ['uniform', 'distance'],
                    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "CatBoosting Regressor": {
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [100, 200]
                }
            }
            
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
            
            # to get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))
            
            # to get the best model name from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found with score greater than 0.6", sys)
            
            logger.info(f"Best model found on both training and testing dataset: {best_model_name} with r2 score: {best_model_score}")
            
            # preprocessing_obj = load_object(file_path=preprocessor_path)
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square
            
            
            # for i in range(len(models)):
            #     model = list(models.values())[i]
            #     model.fit(X_train, y_train)
                
            #     y_train_pred = model.predict(X_train)
            #     y_test_pred = model.predict(X_test)
                
            #     train_model_score = r2_score(y_train, y_train_pred)
            #     test_model_score = r2_score(y_test, y_test_pred)
                
            #     model_report[list(models.keys())[i]] = test_model_score
            
            # best_model_score = max(sorted(model_report.values()))
            # best_model_name = list(model_report.keys())[
            #     list(model_report.values()).index(best_model_score)
            # ]
            # best_model = models[best_model_name]
            
            # logger.info(f"Best model found on both training and testing dataset: {best_model_name} with r2 score: {best_model_score}")
            
            # save_object(
            #     file_path=self.model_trainer_config.trained_model_file_path,
            #     obj=best_model
            # )
            
        except Exception as e:
            raise CustomException(e, sys)