from configs.config import logging
from typing import Any, Dict, List

import catboost as cb
import pandas as pd


class Ranker:
    def __init__(
            self,
            model_path: str = "./artefacts/catboost_clf.cbm"
    ):
        logging.info('Loading the CatBoostClassifier model...')
        self.ranker = cb.CatBoostClassifier().load_model(fname=model_path)

    @staticmethod
    def fit(
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_test: pd.DataFrame = None,
        y_test: pd.DataFrame = None,
        ranker_params: Dict[str, Any] = {},
        categorical_cols: List[str] = []
    ) -> None:
        """
        Trains and saves CatBoostClassifier model for further inference
        :param X_train: pd.DataFrame, train dataframe without target column
        :param y_train: pd.DataFrame, train dataframe with only target column
        :param X_test: pd.DataFrame, evaluation dataframe without target column
        :param y_test: pd.DataFrame, evaluation dataframe with only target column
        :param ranker_params: Dict[str, Any], dictionary with hyperparameter specification
        :param categorical_cols: List[str], list with specification of categorical columns
        :return: None
        """

        logging.info(f'CatBoostClassifier ranker model initialization with params {ranker_params}...')
        cbm_classifier = cb.CatBoostClassifier(
            loss_function=ranker_params.get("loss_function", "CrossEntropy"),
            iterations=ranker_params.get("iterations", 5000),
            learning_rate=ranker_params.get("lr", 0.1),
            depth=ranker_params.get("depth", 6),
            random_state=ranker_params.get("random_state", 1234),
            verbose=ranker_params.get("verbose", True),
        )

        logging.info('Fitting CatBoostClassifier model...')
        cbm_classifier.fit(
            X_train,
            y_train,
            eval_set=(X_test, y_test),
            early_stopping_rounds=100,  # to avoid overfitting,
            cat_features=categorical_cols,
        )

        cbm_classifier.save_model(
            ranker_params.get("ranker_path", "./artefacts/catboost_clf.cbm")
        )

    def infer(
            self,
            ranker_input: List
    ) -> List:
        """
        Method to make inference for the output from LightFM model
        :param ranker_input: List dict with ranks {"item_id": 1, ...}
        :return: List, containing CatBoost probability predictions ordered by LightFM rank
        """

        logging.info('Making CatBoostClassifier predictions...')
        preds = self.ranker.predict_proba(ranker_input)[:, 1]
        logging.info('Final CatBoostClassifier predictions formed.')

        return preds
