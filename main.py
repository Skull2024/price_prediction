# main.py
import sys
import os
import logging
import subprocess

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Добавляем путь к родительской директории в sys.path, чтобы избежать ошибок импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_cleaned_db
from src.cleaned_data_v3 import clean_and_save_data
from src.data_preprocessing import load_data, preprocess_data
from src.model_trainer_v5 import ModelTrainer
from src.visualization import plot_summary_graphs, plot_metrics_comparison, plot_overfitting_comparison
from src.train_classifier_models import train_classifier_models

def main():
    # Проверка корректности ввода города
    valid_cities = ["Алматы", "Астана", "Шымкент"]
    while True:
        selected_city = input("Выберите город (Алматы, Астана, Шымкент): ").strip()
        if selected_city not in valid_cities:
            print(f"Ошибка: введите один из следующих городов: {', '.join(valid_cities)}")
        else:
            break

    # Проверка ввода выбора режима модели
    while True:
        choice = input("Использовать последнюю сохранённую модель? (y/n): ").strip().lower()
        if choice not in ["y", "n"]:
            print("Ошибка: пожалуйста, введите 'y' или 'n'.")
        else:
            break

    # Подавим лишние TF-логи
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if choice == "y":
        logging.info("Запуск прогноза на ранее обученной модели.")
        subprocess.run(["python", os.path.join("src", "direct_predict.py"), selected_city], env=env)

    elif choice == "n":
        logging.info("Переобучение модели с нуля для города %s.", selected_city)

        # Этап 1: Очистка данных
        db_path = get_cleaned_db(selected_city)
        clean_and_save_data(output_file=db_path, selected_city=selected_city)

        # Этап 2: Предобработка
        df = load_data(db_path=db_path)
        processed_df, pipeline = preprocess_data(
            df,
            selected_city=selected_city,
            save_pipeline=True
        )

        # Этап 3: Обучение моделей
        trainer = ModelTrainer(db_path=db_path, selected_city=selected_city)
        results = trainer.train_models()

        # Этап 4: Классификация
        X, y_trans, y_orig, qt, city_values, poly, scaler, df = trainer.load_data()
        train_classifier_models(df=df, selected_city=trainer.selected_city, models_folder=trainer.models_folder)

        # Этап 5: Визуализация результатов
        plot_summary_graphs(results)
        plot_metrics_comparison(results)
        plot_overfitting_comparison(results)

        # Этап 6: Запуск прямого прогнозирования
        logging.info("Запускаем direct_predict.py для прогноза...")

        script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "direct_predict.py"))
        subprocess.run(["python", script_path, selected_city], env=env)

if __name__ == "__main__":
    main()