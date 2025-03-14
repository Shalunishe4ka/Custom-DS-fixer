# 🐶🐱 Dataset Fixer — Автоматическая очистка датасета изображений

## 🔥 Описание
`Dataset Fixer` — это удобный Python-скрипт для автоматической очистки и предобработки датасета изображений. Он проверяет и исправляет распространённые проблемы, такие как:

✅ **Битые файлы** — изображения, которые не открываются или имеют ошибки загрузки.

✅ **Подозрительные изображения** — однотонные, содержащие мало цветов (логотипы, заглушки) или с нежелательным текстом (например, "Photo Unavailable").

✅ **Дубликаты** — идентичные изображения (по хэшу), которые можно удалить, чтобы избежать избыточности в данных.

✅ **Классификация** — автоматическое распределение изображений по папкам (`cat` и `dog`).

## 🚀 Установка
Перед началом работы установите необходимые библиотеки:

```bash
pip install pillow numpy tqdm imagehash pytesseract
```

**Дополнительно:** Убедитесь, что у вас установлен `Tesseract OCR` (для распознавания текста на изображениях). Установить его можно так:

**Linux (Ubuntu/Debian):**
```bash
sudo apt install tesseract-ocr
```

**Windows:**
Скачайте и установите [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) и добавьте его в `PATH`.

## ⚙️ Использование

### 📌 Проверка датасета
Чтобы проверить датасет на битые файлы, дубликаты и подозрительные изображения, запустите:

```bash
python dataset_fixer.py --check
```

Если найдены проблемы, они будут выведены в консоль.

### 🔧 Исправление (удаление дубликатов и плохих изображений)
Чтобы автоматически удалить битые файлы, дубликаты и подозрительные изображения, используйте:

```bash
python dataset_fixer.py --fix
```

### 📂 Распределение изображений по папкам
Если в вашем датасете изображения подписаны (`cat_01.jpg`, `dog_42.png` и т.д.), вы можете разложить их по папкам:

```bash
python dataset_fixer.py --distribute
```

## 🛠️ Основные функции

| Функция | Описание |
|---------|----------|
| `list_truncated_images(directory)` | Проверяет папку на наличие битых изображений. |
| `is_blank_image(image_path)` | Определяет, является ли изображение однотонным. |
| `is_low_color_image(image_path, threshold=50)` | Проверяет, содержит ли изображение мало цветов (например, заглушки). |
| `contains_unavailable_text(image_path)` | Использует OCR для поиска нежелательных надписей на изображениях. |
| `find_duplicates(folders)` | Ищет дубликаты изображений в папках. |
| `remove_duplicates(duplicates)` | Удаляет найденные дубликаты. |
| `distribute_images(source_dir, target_dir)` | Распределяет изображения по папкам (`cat`, `dog`). |

## 🏎️ Примеры запуска

### Полная очистка датасета:
```bash
python dataset_fixer.py --check --fix
```

### Очистка и сортировка:
```bash
python dataset_fixer.py --fix --distribute
```

### Проверка, исправление и распределение в одном запуске:
```bash
python dataset_fixer.py --check --fix --distribute
```

## 📌 Заметки
- Скрипт использует `multiprocessing` и `concurrent.futures` для ускоренной обработки изображений.
- OCR (Tesseract) обрабатывает изображения с таймаутом 5 секунд, чтобы избежать зависаний.
- Дубликаты удаляются по хэшу изображений (`imagehash.average_hash`).

## 🏆 Поддержка
Если у вас есть идеи по улучшению скрипта или вы нашли баг, создайте [Issue](https://github.com/your-repo/dataset-fixer/issues) или отправьте Pull Request!

Happy cleaning! 🚀🐾
