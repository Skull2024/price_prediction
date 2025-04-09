import sqlite3
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", message="X has feature names, but")
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, QuantileTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    median_absolute_error, mean_absolute_percentage_error
)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2
from sklearn.linear_model import Ridge, ElasticNet
import joblib
import os
import json
import logging
from config import MIN_PRICE, MAX_PRICE, get_cleaned_db, get_models_folder
from src.feature_engineering import add_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class ModelTrainer:
    def __init__(self, db_path, selected_city):
        self.db_path = db_path
        self.selected_city = selected_city
        self.models_folder = get_models_folder(self.selected_city)
        os.makedirs(self.models_folder, exist_ok=True)

    def load_data(self):
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql("SELECT * FROM cleaned_flats;", conn)
        conn.close()

        df = df[df["city"] == self.selected_city]
        df = df[(df["price"] >= MIN_PRICE) & (df["price"] <= MAX_PRICE)]

        df["square"] = df["square"].replace(0, np.nan)
        df["room"] = df["room"].replace(0, np.nan)
        df.fillna(df.median(numeric_only=True), inplace=True)

        df = add_features(df)

        y_original = df["price"]

        if y_original.empty:
            raise ValueError("Нет данных для обучения: колонка 'price' пуста после фильтрации.")

        qt = QuantileTransformer(output_distribution='normal', random_state=42)
        y_transformed = qt.fit_transform(y_original.values.reshape(-1, 1)).flatten()

        X_num = df[[
            "square", "room", "room_density",
            "year", "month", "day", "weekday", "days_from_start"
        ]]

        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly.fit_transform(X_num)
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_poly)

        X_final = np.hstack([X_scaled, df[["is_studio"]].values])

        return X_final, y_transformed, y_original, qt, df["city"].values, poly, scaler, df

    def create_neural_network(self, input_shape):
        model = Sequential([
            Input(shape=(input_shape,)),
            Dense(256, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(0.3),
            Dense(128, kernel_regularizer=l2(1e-4)),
            BatchNormalization(),
            LeakyReLU(),
            Dropout(0.15),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.0003), loss=Huber(), metrics=['mae'])
        return model

    def compute_metrics(self, y_true, y_pred):
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
            "R2": r2_score(y_true, y_pred),
            "MedianAE": median_absolute_error(y_true, y_pred),
            "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100
        }

    def train_classic_models(self, X_train, X_test, y_train_trans, y_test_trans, y_train_orig, y_test_orig, qt):
        models = {
            "Ridge": Ridge(alpha=10, fit_intercept=True),
            "ElasticNet": ElasticNet(alpha=0.01, fit_intercept=True, l1_ratio=0.1, max_iter=10000),
            "Random Forest": RandomForestRegressor(
                bootstrap=False, max_depth=10, max_features='sqrt',
                max_samples=None, min_samples_leaf=1, min_samples_split=5,
                n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(
                learning_rate=0.03, max_depth=7, max_features='sqrt',
                min_samples_leaf=2, min_samples_split=2, n_estimators=300,
                subsample=1.0, random_state=42),
            "XGBoost": XGBRegressor(
                learning_rate=0.03, max_depth=7, min_child_weight=2,
                n_estimators=200, subsample=1.0, colsample_bytree=0.6,
                gamma=0.01, reg_alpha=0, reg_lambda=0.5, random_state=42, verbosity=0),
            "LightGBM": LGBMRegressor(
                learning_rate=0.05, colsample_bytree=0.8, max_depth=8,
                n_estimators=300, num_leaves=50, subsample=0.8,
                random_state=42, verbosity=-1),
            "CatBoost": CatBoostRegressor(
                learning_rate=0.05, depth=8, l2_leaf_reg=1,
                n_estimators=300, verbose=0, random_state=42)
        }

        results = {}
        trained_models = {}
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        for name, model in models.items():
            model.fit(X_train, y_train_trans)
            trained_models[name] = model

            y_pred_train_trans = model.predict(X_train)
            y_pred_train_orig = qt.inverse_transform(y_pred_train_trans.reshape(-1, 1)).flatten()
            train_metrics = self.compute_metrics(y_train_orig, y_pred_train_orig)

            y_pred_trans = model.predict(X_test)
            y_pred_orig = qt.inverse_transform(y_pred_trans.reshape(-1, 1)).flatten()
            test_metrics = self.compute_metrics(y_test_orig, y_pred_orig)

            y_cv_pred_trans = cross_val_predict(model, X_train, y_train_trans, cv=cv)
            y_cv_pred_orig = qt.inverse_transform(y_cv_pred_trans.reshape(-1, 1)).flatten()
            cv_metrics = self.compute_metrics(y_train_orig, y_cv_pred_orig)

            print(f"{name}:")
            print(f"  Train MAE: {train_metrics['MAE']:.0f}")
            print(f"  Test  MAE: {test_metrics['MAE']:.0f}")
            print(f"  CV    MAE: {cv_metrics['MAE']:.0f}\n")

            results[name] = {
                "y_true": y_test_orig,
                "y_pred": y_pred_orig,
                "metrics": test_metrics,
                "train_metrics": train_metrics,
                "cv_metrics": cv_metrics
            }
        return results, trained_models

    def train_neural_network(self, X_train, X_test, y_train_trans, y_test_trans, y_test_orig, qt):
        nn_model = self.create_neural_network(X_train.shape[1])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7)
        ]
        history = nn_model.fit(
            X_train, y_train_trans,
            validation_data=(X_test, y_test_trans),
            epochs=200,
            batch_size=32,
            callbacks=callbacks,
            verbose=2
        )
        y_pred_nn_trans = nn_model.predict(X_test)
        y_pred_nn_orig = qt.inverse_transform(np.clip(y_pred_nn_trans, -5, 5)).flatten()
        nn_metrics = self.compute_metrics(y_test_orig, y_pred_nn_orig)
        print("Neural Network:")
        print(f"  Test MAE: {nn_metrics['MAE']:.0f}\n")
        return {
            "y_true": y_test_orig,
            "y_pred": y_pred_nn_orig,
            "history": history.history,
            "metrics": nn_metrics,
            "model": nn_model
        }

    def explain_model_with_shap(self, model, X_sample, model_name, poly, include_is_studio=True):
        try:
            # Получаем имена признаков после PolynomialFeatures
            feature_names = list(poly.get_feature_names_out([
                "square", "room", "room_density", "year", "month", "day", "weekday", "days_from_start"
            ]))
            if include_is_studio:
                feature_names.append("is_studio")

            # Преобразуем X_sample обратно в DataFrame с колонками
            X_df = pd.DataFrame(X_sample, columns=feature_names)

            explainer = shap.Explainer(model.predict, X_df)
            shap_values = explainer(X_df)

            shap.plots.beeswarm(shap_values, max_display=10)
            plt.title(f"SHAP summary for {model_name}")
            plt.tight_layout()

            # shap_output_path = os.path.join(self.models_folder, f"shap_{model_name.replace(' ', '_').lower()}.png")
            # plt.savefig(shap_output_path)
            # plt.close()
            # logging.info("SHAP-график для %s сохранён в %s", model_name, shap_output_path)
        except Exception as e:
            logging.warning("Не удалось построить SHAP-график для %s: %s", model_name, e)


    def train_models(self):
        X, y_trans, y_orig, qt, city_values, poly, scaler, df = self.load_data()
        X_train, X_test, y_train_trans, y_test_trans = train_test_split(X, y_trans, test_size=0.2, random_state=42)
        y_test_orig = qt.inverse_transform(y_test_trans.reshape(-1, 1)).flatten()
        y_train_orig = qt.inverse_transform(y_train_trans.reshape(-1, 1)).flatten()

        classic_results, trained_models = self.train_classic_models(X_train, X_test, y_train_trans, y_test_trans, y_train_orig, y_test_orig, qt)
        nn_result = self.train_neural_network(X_train, X_test, y_train_trans, y_test_trans, y_test_orig, qt)
        trained_models["Neural Network"] = nn_result["model"]
        classic_results["Neural Network"] = nn_result

        print("\n🔍 Интерпретация моделей через SHAP:")
        for name in ["Gradient Boosting", "XGBoost", "Random Forest"]:
            model = trained_models.get(name)
            if model:
                print(f"\n➡ {name}")
                # self.explain_model_with_shap(model, X_train[:100], name)
                self.explain_model_with_shap(model, X_train[:100], name, poly)

        for name, model in trained_models.items():
            if name == "Neural Network":
                model.save(os.path.join(self.models_folder, "neural_network_model.h5"))
                logging.info("Нейронная сеть сохранена в %s", os.path.join(self.models_folder, "neural_network_model.h5"))
            else:
                joblib.dump(model, os.path.join(self.models_folder, f"{name.replace(' ', '_').lower()}_model.pkl"))
                logging.info("Модель %s сохранена в %s", name, os.path.join(self.models_folder, f"{name.replace(' ', '_').lower()}_model.pkl"))

        joblib.dump(poly, os.path.join(self.models_folder, "polynomial_features.pkl"))
        joblib.dump(scaler, os.path.join(self.models_folder, "robust_scaler.pkl"))
        joblib.dump(qt, os.path.join(self.models_folder, "quantile_transformer.pkl"))
        logging.info("Параметры преобразования сохранены.")

        mae_ranking = {}
        sorted_models = sorted(classic_results.items(), key=lambda x: x[1]["metrics"]["MAE"])
        for i, (name, result) in enumerate(sorted_models, 1):
            mae = result["metrics"]["MAE"]
            mae_ranking[name] = {
                "rank": i,
                "mae": round(mae, 2)
            }

        with open(os.path.join(self.models_folder, "model_ranking.json"), "w", encoding="utf-8") as f:
            json.dump(mae_ranking, f, indent=4, ensure_ascii=False)

        return classic_results

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Укажите город как аргумент командной строки.")
        sys.exit(1)

    selected_city = sys.argv[1]
    db_path = get_cleaned_db(selected_city)

    trainer = ModelTrainer(db_path=db_path, selected_city=selected_city)
    trainer.train_models()
