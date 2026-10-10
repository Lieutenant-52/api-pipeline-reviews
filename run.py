
import os
import csv
import json
import time
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:3b")

if not API_KEY or not BASE_URL:
    raise ValueError("Укажите API_KEY и BASE_URL в файле .env")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

SYSTEM_PROMPT = """
Ты — ассистент для анализа отзывов покупателей.

Для каждого отзыва определи три показателя:

1. sentiment — тональность отзыва:
   positive, negative или neutral.

2. topic — краткая и конкретная тема отзыва
   исключительно на русском языке.

3. product_aspect — одна категория из списка:
   "товар", "доставка", "упаковка", "цена",
   "обслуживание", "сайт и приложение", "возврат", "другое".

Правила:
- Тему всегда пиши на русском языке.
- Не используй английские слова в topic и product_aspect.
- Не используй слишком общие темы, например "покупки" или "продукт".
- Не добавляй сведения, которых нет в отзыве.
- Выбирай категорию по основной мысли отзыва.
- Если отзыв касается неисправности или характеристик товара,
  выбирай категорию "товар".
- Если отзыв касается скорости доставки, выбирай "доставка".
- Если отзыв касается повреждённой коробки, выбирай "упаковка".
- Если отзыв касается стоимости, выбирай "цена".
- Если отзыв касается сотрудников, выбирай "обслуживание".
- Если отзыв касается сайта или приложения,
  выбирай "сайт и приложение".
- Если отзыв касается возврата товара или денег,
  выбирай "возврат".
- Если ни одна категория не подходит, выбирай "другое".

Пример:
Отзыв: "Телефон начал сильно нагреваться."
Ответ:
{"sentiment": "negative", "topic": "перегрев телефона",
"product_aspect": "товар"}

Верни только один валидный JSON-объект без Markdown
и дополнительных пояснений.

Формат:
{
  "sentiment": "positive|negative|neutral",
  "topic": "краткая тема на русском языке",
  "product_aspect": "категория из списка"
}
"""

VALID_SENTIMENTS = {"positive", "negative", "neutral"}

VALID_ASPECTS = {
    "товар",
    "доставка",
    "упаковка",
    "цена",
    "обслуживание",
    "сайт и приложение",
    "возврат",
    "другое"
}


def analyze_review(review_text):
    """Отправляет отзыв модели и проверяет результат."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": review_text}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=300,
        timeout=60
    )

    raw = response.choices[0].message.content

    if not raw:
        raise ValueError("Модель вернула пустой ответ")

    parsed = json.loads(raw)

    sentiment = parsed.get("sentiment", "neutral")
    if sentiment not in VALID_SENTIMENTS:
        sentiment = "neutral"

    topic = parsed.get("topic", "тема не определена")
    if not isinstance(topic, str) or not topic.strip():
        topic = "тема не определена"

    aspect = parsed.get("product_aspect", "другое")
    if aspect not in VALID_ASPECTS:
        aspect = "другое"

    return {
        "sentiment": sentiment,
        "topic": topic.strip(),
        "product_aspect": aspect
    }


def run_pipeline(input_csv, output_json):
    """Читает отзывы, анализирует их и сохраняет результаты."""

    results = []
    sentiment_counts = Counter()
    aspect_counts = Counter()

    with open(
        input_csv, "r", encoding="utf-8-sig", newline=""
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames or not {
            "id", "text"
        }.issubset(reader.fieldnames):
            raise ValueError(
                "CSV-файл должен содержать столбцы id и text"
            )

        for row in reader:
            review_id = row["id"]
            review_text = row["text"]

            print(f"Обработка отзыва ID {review_id}...")

            try:
                analysis = analyze_review(review_text)

                result = {
                    "id": int(review_id),
                    "text": review_text,
                    **analysis
                }

                sentiment_counts[analysis["sentiment"]] += 1
                aspect_counts[analysis["product_aspect"]] += 1

            except Exception as error:
                print(
                    f"Ошибка при обработке ID {review_id}: "
                    f"{error}"
                )

                result = {
                    "id": int(review_id),
                    "text": review_text,
                    "error": str(error)
                }

            results.append(result)
            time.sleep(0.2)

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2
        )


    summary = {
        "total_reviews": len(results),
        "successful_reviews": sum(
            1 for item in results if "error" not in item
        ),
        "failed_reviews": sum(
            1 for item in results if "error" in item
        ),
        "sentiment_counts": dict(sentiment_counts),
        "product_aspect_counts": dict(aspect_counts)
    }

    with open("summary.json", "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nОбработка завершена.")
    print(f"Результаты сохранены в {output_json}")
    print("Сводная статистика сохранена в summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_pipeline("input.csv", "output.json")
