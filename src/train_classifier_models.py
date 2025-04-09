import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_classifier_models(df, selected_city, models_folder):
    # Классификация: дорого/дешево
    df = df[df["city"] == selected_city]
    df = df[(df["price"] >= 30000) & (df["price"] <= 1000000)]

    # Добавим метку класса: 1 — выше медианы, 0 — ниже или равно
    threshold = df["price"].median()
    df["high_price"] = (df["price"] > threshold).astype(int)

    # Выбор признаков и цели
    features = ["square", "room", "year", "month", "day", "weekday"]
    X = df[features]
    y = df["high_price"]

    # Обработка пропусков и масштабирование
    X = X.fillna(X.median(numeric_only=True))  # Убираем предупреждение SettingWithCopyWarning
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "Report": classification_report(y_test, y_pred)
        }

        joblib.dump(model, os.path.join(models_folder, f"{name.replace(' ', '_').lower()}_classifier.pkl"))

    joblib.dump(scaler, os.path.join(models_folder, "classifier_scaler.pkl"))
    logging.info("Классификационные модели обучены и сохранены.")

    return results
