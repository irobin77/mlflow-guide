import mlflow
import os

from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines

from transformers import pipeline


# os.environ["MISTRAL_API_KEY"] = ""

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test_evaluation")

# Создание объекта модели
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")


# Функция предсказания
def qa_predict_fn(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=20)
    response = outputs[0]["generated_text"]

    return response


# Простой Q&A датасет с вопросами и ожидаемыми ответами
evaluation_dataset = [
    {
        "inputs": {"question": "What is the capital of France?"},
        "expectations": {"expected_response": "Paris"},
    },
    {
        "inputs": {"question": "Who was the first person to build an airplane?"},
        "expectations": {"expected_response": "Wright Brothers"},
    },
    {
        "inputs": {"question": "Who wrote Romeo and Juliet?"},
        "expectations": {"expected_response": "William Shakespeare"},
    },
]


# Определение функций оценки
@scorer
def is_concise(outputs: str) -> bool:
    """Evaluate if the answer is concise (less than 30 words)"""
    return len(outputs.split()) <= 30


scorers = [
    Correctness(model="mistral:/mistral-tiny"),
    Guidelines(name="is_english", guidelines="The answer must be in English", model="mistral:/mistral-tiny"),
    is_concise,
]

# Запуск
results = mlflow.genai.evaluate(
    data=evaluation_dataset,
    predict_fn=qa_predict_fn,
    scorers=scorers,
)
