# src.data_preprocessing.py
import sqlite3
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib
import os
from config import get_pipeline_file
from src.feature_engineering import add_features
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_data(db_path: str) -> pd.DataFrame:
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql("SELECT * FROM cleaned_flats;", conn)
        logging.info("Данные успешно загружены из %s", db_path)
        return df
    except Exception as e:
        logging.error("Ошибка при загрузке данных: %s", e)
        raise

def preprocess_data(df: pd.DataFrame, selected_city: str, save_pipeline: bool = False) -> Tuple[pd.DataFrame, Pipeline]:
    required_cols = {"city", "price", "square", "room", "date"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Отсутствуют необходимые столбцы: {required_cols - set(df.columns)}")

    # Фильтрация по городу и удаление нулевых значений
    df = df[df["city"] == selected_city].copy()
    df = df[(df["square"] > 0) & (df["room"] > 0)]
    df.drop_duplicates(inplace=True)

    # Добавление новых признаков
    df = add_features(df)

    # Явно задаём порядок признаков
    numeric_features = [
        "square", "room", "room_density",
        "year", "month", "day", "weekday", "days_from_start",
        "is_studio"
    ]
    X = df[numeric_features]
    y = df["price"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = Pipeline(steps=[
        ("num", numeric_transformer)
    ])

    X_processed = preprocessor.fit_transform(X)
    X_processed_df = pd.DataFrame(X_processed, columns=numeric_features)
    processed_df = pd.concat([X_processed_df, y.reset_index(drop=True)], axis=1)

    logging.info("Данные успешно обработаны")

    if save_pipeline:
        pipeline_path = get_pipeline_file(selected_city)
        os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
        joblib.dump(preprocessor, pipeline_path)
        logging.info("Pipeline сохранён в %s", pipeline_path)
        
    return processed_df, preprocessor

def split_data(df: pd.DataFrame):
    X = df.drop("price", axis=1)
    y = df["price"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    import sys
    from config import get_cleaned_db

    if len(sys.argv) < 2:
        print("Укажите город как аргумент командной строки.")
        sys.exit(1)

    selected_city = sys.argv[1]
    db_path = get_cleaned_db(selected_city)
    df = load_data(db_path)
    processed_df, _ = preprocess_data(df, selected_city=selected_city, save_pipeline=True)
    X_train, X_test, y_train, y_test = split_data(processed_df)
    logging.info("Данные успешно разделены на train/test")