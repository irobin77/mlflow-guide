import mlflow

from langchain.prompts import PromptTemplate
from langchain_huggingface.llms import HuggingFacePipeline


# Создаем объект модели на базе Hugging Face
hf = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 15},
)

# Включение автологирования
mlflow.langchain.autolog()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("LangChain")

prompt_template = PromptTemplate.from_template(
    """Question: {question}"""
)

chain = prompt_template | hf

print(chain.invoke(
    {
        "question": "Hello! How are you?"
    }
))
