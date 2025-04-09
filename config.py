# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Пути к папкам
DATA_FOLDER = os.path.join(BASE_DIR, "data")
SRC_FOLDER = os.path.join(BASE_DIR, "src")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
PIPELINES_FOLDER = os.path.join(BASE_DIR, "pipelines")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")

# Создание необходимых директорий
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(PIPELINES_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# Фильтрация по цене и квадратуре (настройки)
MIN_PRICE = 80000
MAX_PRICE = 1_100_000

MIN_SQUARE = 20
MAX_SQUARE = 150


# Получение пути к очищенной БД по городу
def get_cleaned_db(city):
    return os.path.join(PROCESSED_FOLDER, f"cleaned_data_{city}.sqlite")

# Получение пути к pipeline по городу
def get_pipeline_file(city):
    return os.path.join(PIPELINES_FOLDER, f"preprocessing_pipeline_{city}.pkl")

# Получение директории для моделей по городу
def get_models_folder(city):
    folder = os.path.join(MODELS_FOLDER, city)
    os.makedirs(folder, exist_ok=True)
    return folder
