# cleaned_data_v3.py
import sqlite3
import pandas as pd
import os
import logging
import numpy as np
from config import DATA_FOLDER, MIN_PRICE, MAX_PRICE, MIN_SQUARE, MAX_SQUARE, get_cleaned_db

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REQUIRED_FLATS = {"id", "city", "square", "room"}
REQUIRED_PRICES = {"flat_id", "price"}

def clean_and_save_data(input_folder=DATA_FOLDER, output_file=None, selected_city=None):
    if output_file is None:
        raise ValueError("Не указан путь к выходному файлу output_file.")

    """
    Для каждой базы данных *.sqlite в input_folder:
      1) Читает таблицу flats;
      2) Читает таблицу prices;
      3) Объединяет данные по ключу: flats.id = prices.flat_id (INNER JOIN);
      4) Если в таблице prices присутствует столбец date, он тоже извлекается;
      5) Фильтрует строки по цене и городу;
      6) Сохраняет итоговый DataFrame в файле output_file (в таблице cleaned_flats).
    """
    all_flats = []

    if not os.path.exists(input_folder):
        logging.error("Папка %s не существует", input_folder)
        return

    db_files = [
        os.path.join(input_folder, f) 
        for f in os.listdir(input_folder) 
        if f.endswith(".sqlite")
    ]
    logging.info("Найдено %d файлов для обработки", len(db_files))

    for file in db_files:
        try:
            with sqlite3.connect(file) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='flats';")
                if not cursor.fetchone():
                    logging.warning("Файл %s: отсутствует таблица 'flats'", file)
                    continue

                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prices';")
                if not cursor.fetchone():
                    logging.warning("Файл %s: отсутствует таблица 'prices'", file)
                    continue

                df_flats = pd.read_sql("SELECT * FROM flats;", conn)
                df_prices = pd.read_sql("SELECT * FROM prices;", conn)

                if not REQUIRED_FLATS.issubset(df_flats.columns):
                    logging.warning("Файл %s: в таблице flats отсутствуют столбцы: %s",
                                    file, REQUIRED_FLATS - set(df_flats.columns))
                    continue

                if not REQUIRED_PRICES.issubset(df_prices.columns):
                    logging.warning("Файл %s: в таблице prices отсутствуют столбцы: %s",
                                    file, REQUIRED_PRICES - set(df_prices.columns))
                    continue

                df_flats_needed = df_flats[["id", "city", "square", "room"]]
                if selected_city:
                    df_flats_needed = df_flats_needed[df_flats_needed["city"] == selected_city]
                if df_flats_needed.empty:
                    logging.warning("Файл %s: нет записей для города %s", file, selected_city)
                    continue

                if "date" in df_prices.columns:
                    df_prices_needed = df_prices[["flat_id", "price", "date"]]
                else:
                    df_prices_needed = df_prices[["flat_id", "price"]]

                df_merged = pd.merge(
                    df_flats_needed, 
                    df_prices_needed, 
                    left_on="id", 
                    right_on="flat_id", 
                    how="inner"
                )
                df_merged.drop(columns=["flat_id"], inplace=True, errors="ignore")
                df_merged.drop_duplicates(subset=["id", "city", "square", "room", "price"], inplace=True)

                if df_merged.empty:
                    logging.warning("Файл %s: после объединения нет подходящих записей", file)
                    continue

                all_flats.append(df_merged)

            logging.info("Файл %s успешно объединён (flats + prices)", file)
        except Exception as e:
            logging.error("Ошибка при обработке файла %s: %s", file, e)
            continue

    if not all_flats:
        logging.error("Не удалось обработать ни один файл.")
        return

    # final_data = pd.concat(all_flats, ignore_index=True)

    # # Очистка по цене и другим числовым полям
    # final_data = final_data[(final_data["price"] >= MIN_PRICE) & (final_data["price"] <= MAX_PRICE)]
    # final_data = final_data[(final_data["square"] >= MIN_SQUARE) & (final_data["square"] <= MAX_SQUARE) & (final_data["room"] > 0)]
# После объединения данных и до сохранения итогового DataFrame:
    final_data = pd.concat(all_flats, ignore_index=True)

    # Вычисляем процентильные границы для цены и квадратуры
    price_lower = final_data["price"].quantile(0.03)
    price_upper = final_data["price"].quantile(0.98)
    square_lower = final_data["square"].quantile(0.15)
    square_upper = final_data["square"].quantile(0.98)

    # Применяем фильтрацию по вычисленным границам и исключаем записи с количеством комнат меньше 1
    final_data = final_data[(final_data["price"] >= price_lower) & (final_data["price"] <= price_upper)]
    final_data = final_data[(final_data["square"] >= square_lower) & (final_data["square"] <= square_upper) & (final_data["room"] > 0)]

    columns_to_keep = ["city", "square", "room", "price"]
    if "date" in final_data.columns:
        columns_to_keep.append("date")
    final_data = final_data[columns_to_keep]

    # Поддержка автоматического вывода пути, если забыли указать
    if selected_city and output_file == get_cleaned_db:
        output_file = get_cleaned_db(selected_city)

    try:
        with sqlite3.connect(output_file) as conn:
            final_data.to_sql("cleaned_flats", conn, if_exists="replace", index=False)
        logging.info("Итоговые данные сохранены в %s (таблица cleaned_flats)", output_file)
    except Exception as e:
        logging.error("Ошибка при сохранении итоговых данных: %s", e)

if __name__ == "__main__":
    clean_and_save_data(selected_city="Алматы")
