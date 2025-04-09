import joblib 
import pandas as pd
import numpy as np
import os
import json
import subprocess
import warnings
from datetime import datetime
import sys
from tabulate import tabulate
from tensorflow.keras.models import load_model

# Подавление лишних предупреждений
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

# Добавление пути к корню проекта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import MODELS_FOLDER

# --- Получение города из аргументов ---
if len(sys.argv) < 2:
    print("Ошибка: город должен быть передан как аргумент.")
    sys.exit(1)

city = sys.argv[1]
city_model_path = os.path.join(MODELS_FOLDER, city)

# --- Список доступных моделей ---
available_models = {
    "1": "ridge_model.pkl",
    "2": "elasticnet_model.pkl",
    "3": "random_forest_model.pkl",
    "4": "gradient_boosting_model.pkl",
    "5": "xgboost_model.pkl",
    "6": "lightgbm_model.pkl",
    "7": "catboost_model.pkl",
    "8": "neural_network_model.h5"
}

# --- Проверка наличия хотя бы одной обученной модели ---
has_models = any(os.path.exists(os.path.join(city_model_path, fname)) for fname in available_models.values())
if not has_models:
    print(f"\n❗ Для города '{city}' нет обученных моделей.")
    print("Пожалуйста, сначала запустите обучение моделей (выберите 'n' при старте).")
    sys.exit(1)

# --- Загрузка рейтинга моделей ---
model_ranking = {}
ranking_file = os.path.join(city_model_path, "model_ranking.json")
if os.path.exists(ranking_file):
    with open(ranking_file, "r", encoding="utf-8") as f:
        model_ranking = json.load(f)

def display_model_ranking():
    print("\n📊 Доступные модели с рейтингом MAE:")
    for k, v in available_models.items():
        label = v.replace("_model.pkl", "").replace("_", " ").title().replace(".H5", "")
        model_key = v.split("_model")[0].replace(".h5", "").replace("_", " ").title()
        top_tag = ""
        mae_tag = ""
        for rank_name, data in model_ranking.items():
            if rank_name.lower().replace(" ", "") == model_key.lower().replace(" ", ""):
                top_tag = f" (Топ {data['rank']}"
                mae_tag = f", MAE: {data['mae']:,.0f})"
                break
        print(f"{k}: {label}{top_tag}{mae_tag if top_tag else ''}")

# Внешний цикл для возможности повторного прогноза
while True:
    print("\nВведите данные для прогноза аренды жилья:")

    while True:
        try:
            square = float(input("Площадь квартиры (в м²): "))
            if not (20 <= square <= 200):
                print("Ошибка: допустимая площадь — от 20 до 200 м².")
                continue
            break
        except ValueError:
            print("Ошибка: введите корректное число для площади.")

    while True:
        try:
            room = int(input("Количество комнат: "))
            if not (1 <= room <= 8):
                print("Ошибка: допустимое количество комнат — от 1 до 8.")
                continue
            break
        except ValueError:
            print("Ошибка: введите целое число для количества комнат.")


    print("Выберите режим даты:")
    print("1 - Указать конкретную дату")
    print("2 - Указать диапазон месяцев текущего года")
    while True:
        date_choice = input("Ваш выбор (1/2): ").strip()
        if date_choice in ["1", "2"]:
            break
        else:
            print("Ошибка: введите 1 или 2.")

    dates = []
    if date_choice == "1":
        while True:
            date_str = input("Введите дату (в формате ГГГГ-ММ-ДД): ").strip()
            date = pd.to_datetime(date_str, format="%Y-%m-%d", errors='coerce')
            if pd.isnull(date):
                print("Ошибка: неверный формат даты. Попробуйте ещё раз.")
            else:
                dates = [date]
                break
    elif date_choice == "2":
        while True:
            print("Введите диапазон месяцев (например: 5 9 или 5-9):")
            range_str = input("Диапазон месяцев (формат 5 9 или 5-9): ").strip()
            parts = range_str.replace('-', ' ').split()
            if len(parts) != 2:
                print("Ошибка: укажите два месяца.")
                continue
            try:
                start_month, end_month = map(int, parts)
                if not (1 <= start_month <= 12 and 1 <= end_month <= 12 and start_month <= end_month):
                    print("Ошибка: введите месяцы от 1 до 12.")
                    continue
                break
            except ValueError:
                print("Ошибка: введите числа.")
        year = datetime.now().year
        for month in range(start_month, end_month + 1):
            try:
                dates.append(datetime(year, month, 15))
            except Exception as e:
                print(f"Ошибка при формировании даты: {e}")

    display_model_ranking()

    while True:
        model_indices_str = input("Ваш выбор моделей (например: 2,3): ").strip()
        if not model_indices_str:
            print("Ошибка: введите хотя бы один номер.")
            continue
        model_indices = model_indices_str.split(",")
        selected_models = []
        for idx in model_indices:
            filename = available_models.get(idx.strip())
            if filename:
                model_path = os.path.join(city_model_path, filename)
                if os.path.exists(model_path):
                    try:
                        if filename.endswith(".h5"):
                            selected_models.append(load_model(model_path))
                        else:
                            selected_models.append(joblib.load(model_path))
                    except Exception as e:
                        print(f"Ошибка при загрузке модели {filename}: {e}")
                else:
                    print(f"Модель {filename} не найдена.")
            else:
                print(f"Неверный номер модели: {idx.strip()}")
        if not selected_models:
            print("Ошибка: ни одна модель не выбрана.")
            continue
        break

    try:
        poly = joblib.load(os.path.join(city_model_path, "polynomial_features.pkl"))
        scaler = joblib.load(os.path.join(city_model_path, "robust_scaler.pkl"))
        qt = joblib.load(os.path.join(city_model_path, "quantile_transformer.pkl"))
    except Exception as e:
        print(f"Ошибка при загрузке преобразователей: {e}")
        sys.exit(1)

    new_data = pd.DataFrame()
    for date in dates:
        row = {
            "city": city,
            "square": square,
            "room": room,
            "date": date,
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "weekday": date.weekday(),
        }
        new_data = pd.concat([new_data, pd.DataFrame([row])], ignore_index=True)

    new_data["days_from_start"] = (new_data["date"] - new_data["date"].min()).dt.days
    new_data["room_density"] = new_data["room"] / (new_data["square"] + 1)
    new_data["is_studio"] = (new_data["room"] <= 1).astype(int)

    features_order = [
        "square", "room", "room_density",
        "year", "month", "day", "weekday", "days_from_start"
    ]

    X_num = new_data[features_order]
    X_poly = poly.transform(X_num)
    X_scaled = scaler.transform(X_poly)
    X_final = np.hstack([X_scaled, new_data[["is_studio"]].values])

    predictions = []
    for model in selected_models:
        try:
            preds_trans = model.predict(X_final)
            preds_orig = qt.inverse_transform(np.array(preds_trans).reshape(-1, 1)).flatten()
            predictions.append(preds_orig)
        except Exception as e:
            print(f"Ошибка при прогнозировании моделью: {e}")

    if predictions:
        final_preds = np.mean(predictions, axis=0)
    else:
        print("Ошибка: прогноз не получен.")
        sys.exit(1)

    output = []
    for d, price in zip(new_data["date"], final_preds):
        output.append([d.strftime("%Y-%m-%d"), room, square, city, f"{price:,.0f} ₸"])

    print("\nРезультаты прогноза:")
    print(tabulate(output, headers=["Дата", "Комнаты", "Площадь (м²)", "Город", "Прогноз (₸)"], tablefmt="grid"))

    new_data["predicted_price"] = final_preds
    os.makedirs("predictions", exist_ok=True)
    filename = f"predictions/predict_{city}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    try:
        new_data.to_csv(filename, index=False)
        print(f"\nРезультаты сохранены в файл: {filename}")
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")

    while True:
        again = input("\nХотите сделать ещё один прогноз? (y/n): ").strip().lower()
        if again in ["y", "n"]:
            break
        else:
            print("Ошибка: введите 'y' или 'n'.")
    if again == "y":
        continue
    else:
        sys.exit(0)
