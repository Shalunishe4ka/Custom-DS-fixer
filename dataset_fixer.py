import os
import shutil
import argparse
import warnings
import pytesseract
import multiprocessing as mp
import concurrent.futures
import signal
from PIL import Image, ImageFile
from tqdm import tqdm
import imagehash
import numpy as np

# Позволяет загружать битые изображения
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 📌 Папки по умолчанию (пропишите нужные пути)
DATASET_DIR = ""  # Путь к папке с датасетом
OUTPUT_BAD = ""   # Папка для плохих изображений
TEST_DIR = ""     # Папка с тестовыми изображениями
TRAIN_DIRS = []   # Список папок с тренировочными изображениями, например: ["./train/cat", "./train/dog"]

# 📌 Создаём папку для плохих изображений, если её нет
os.makedirs(OUTPUT_BAD, exist_ok=True)


# 🔍 1. Поиск битых файлов (многопоточный)
def process_file(file_path):
    """Проверяет, битое ли изображение, в т.ч. через предупреждения о 'Truncated File Read'."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            img = Image.open(file_path)
            img.load()
            # Если внутри возникло предупреждение "Truncated File Read", считаем файл битым
            if any("Truncated File Read" in str(warn.message) for warn in w):
                return file_path
        except Exception:
            return file_path  # Вернём путь, если открыть/прочитать не удалось
    return None


def list_truncated_images(directory):
    """Проверяет изображения на ошибки загрузки (битые файлы) параллельно."""
    all_files = [
        os.path.join(root, f)
        for root, _, files in os.walk(directory)
        for f in files
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff'))
    ]

    print(f"🚀 Проверяем {len(all_files)} файлов на битые изображения...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_file, all_files), total=len(all_files)))

    # Оставляем только те, что вернулись не None
    return [f for f in results if f]


# 📌 2. Фильтр изображений (проверка монотонности и количества цветов)
def is_blank_image(image_path):
    """Проверяет, является ли изображение практически однотонным."""
    img = Image.open(image_path).convert("L")
    pixels = np.array(img)
    return np.std(pixels) < 10  # Маленькое отклонение => однотонное изображение


def is_low_color_image(image_path, threshold=50):
    """
    Проверяет, содержит ли изображение мало уникальных цветов
    (характерно для логотипов, заглушек и т.п.).
    """
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img)
    unique_colors = len(np.unique(pixels.reshape(-1, pixels.shape[2]), axis=0))
    return unique_colors < threshold


# 🔄 3. Фильтр изображений с текстом (OCR с таймаутом)
def timeout_handler(signum, frame):
    raise TimeoutError("OCR завис!")


def contains_unavailable_text(image_path):
    """
    Проверяет текст в изображении (OCR), обрезая по таймеру в 5 секунд,
    чтобы не застрять на проблемных файлах.
    """
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # Запускаем таймер на 5 секунд

    try:
        text = pytesseract.image_to_string(Image.open(image_path)).strip().lower()
        signal.alarm(0)  # Сбрасываем таймер, если всё ок
        # В этом списке – ключевые слова, по которым распознаём "ненужные" изображения
        keywords = ["photo unavailable", "9 lives rescue", "adopt me", "shelter", "rescue"]
        return any(word in text for word in keywords)
    except TimeoutError:
        print(f"⚠️ OCR завис на {image_path}, пропускаем...")
        return False
    except Exception as e:
        print(f"⚠️ Ошибка OCR {image_path}: {e}")
        return False


# 📌 4. Поиск и удаление дубликатов
def calculate_hash(image_path):
    """Возвращает хэш (average_hash) для изображения."""
    return imagehash.average_hash(Image.open(image_path))


def images_are_identical(img_path1, img_path2):
    """Сравнивает изображения побайтово (при необходимости)."""
    return np.array_equal(np.array(Image.open(img_path1)), np.array(Image.open(img_path2)))


def find_duplicates(folders):
    """Ищет дубликаты (по хэшу) в заданных папках."""
    all_duplicates = []
    for folder in folders:
        print(f"\n🔍 Ищем дубликаты в {folder}...")
        image_hashes = {}

        for root, _, files in os.walk(folder):
            for file in files:
                # Проверяем расширение
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                    img_path = os.path.join(root, file)
                    try:
                        img_hash = calculate_hash(img_path)
                        image_hashes.setdefault(img_hash, []).append(img_path)
                    except Exception as e:
                        print(f"⚠️ Ошибка при хэшировании {img_path}: {e}")

        # Собираем только те записи, где по одному хэшу > 1 файла
        folder_duplicates = [imgs for imgs in image_hashes.values() if len(imgs) > 1]
        all_duplicates.extend(folder_duplicates)

    return all_duplicates


def remove_duplicates(duplicates):
    """Удаляет все дубликаты, кроме первого файла в каждой группе."""
    for dup_group in duplicates:
        # Первое изображение (dup_group[0]) считаем "основным", остальные удаляем
        for dup in dup_group[1:]:
            try:
                os.remove(dup)
                print(f"🗑️ Удалён дубликат: {dup}")
            except Exception as e:
                print(f"❌ Ошибка при удалении {dup}: {e}")


# 📌 5. Распределение изображений по папкам (пример: 'cat' и 'dog')
def distribute_images(source_dir, target_dir):
    """
    Пример распределения изображений по подпапкам: cat или dog,
    в зависимости от наличия подстроки в имени файла.
    """
    os.makedirs(os.path.join(target_dir, "cat"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "dog"), exist_ok=True)

    for file in tqdm(os.listdir(source_dir), desc="Распределяем изображения"):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            category = "cat" if "cat" in file.lower() else "dog"
            src = os.path.join(source_dir, file)
            dst = os.path.join(target_dir, category, file)
            try:
                shutil.move(src, dst)
            except Exception as e:
                print(f"❌ Ошибка при перемещении {src}: {e}")


# 🚀 Основная функция
def main():
    parser = argparse.ArgumentParser(description="🐶🐱 Датасет-фиксер")
    parser.add_argument("--check", action="store_true",
                        help="Проверить битые файлы, дубликаты и мусорные изображения")
    parser.add_argument("--fix", action="store_true",
                        help="Удалить дубликаты и плохие (подозрительные) изображения")
    parser.add_argument("--distribute", action="store_true",
                        help="Распределить изображения по папкам (cat/dog)")

    args = parser.parse_args()

    # Для удобства: чтобы suspicious был доступен и в блоке fix (если флаги указаны вместе),
    # и не вызвал ошибку, если fix вызван без check.
    suspicious = []

    if args.check:
        print("\n🔍 Запускаем проверку датасета...")

        # 1) Ищем битые файлы
        truncated = list_truncated_images(DATASET_DIR)
        if truncated:
            print(f"❌ Найдено битых файлов: {len(truncated)}")
        else:
            print("✅ Битых файлов нет!")

        # 2) Фильтр изображений (однотонные, мало цветов, неуместный текст)
        for folder in [TEST_DIR] + TRAIN_DIRS:
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff')):
                        img_path = os.path.join(root, file)
                        # Проверяем все три условия
                        if (is_blank_image(img_path)
                                or is_low_color_image(img_path)
                                or contains_unavailable_text(img_path)):
                            suspicious.append(img_path)

        print(f"❌ Найдено подозрительных изображений: {len(suspicious)}")

    if args.fix:
        print("\n🔧 Запускаем исправление...")

        # 3) Поиск и удаление дубликатов
        duplicates = find_duplicates([TEST_DIR] + TRAIN_DIRS)
        remove_duplicates(duplicates)

        # 4) Удаляем все "подозрительные" изображения
        for file in suspicious:
            try:
                os.remove(file)
                print(f"🗑️ Удалено подозрительное изображение: {file}")
            except Exception as e:
                print(f"❌ Ошибка при удалении {file}: {e}")

        print("\n✅ Очистка завершена!")

    if args.distribute:
        print("\n📂 Распределяем изображения по категориям (cat/dog)...")
        distribute_images(TEST_DIR, OUTPUT_BAD)
        print("\n✅ Распределение завершено!")


if __name__ == "__main__":
    main()
