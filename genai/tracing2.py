import mlflow


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("LangChain")

# Define a new GenAI app version, represented as an MLflow LoggedModel
mlflow.set_active_model(name="my-app-v1")

# Log LLM hyperparameters, prompts, and more
mlflow.log_model_params({
    "prompt_template": "My prompt template",
    "llm": "databricks-llama-4-maverick",
    "temperature": 0.2,
})

# Define application code and add MLflow tracing to capture requests and responses.
# (Replace this with your GenAI application or agent code)
@mlflow.trace
def predict(query):
    return f"Response to query: {query}"

# Run your application code. Resulting traces are automatically linked to
# your GenAI app version.
predict("What is MLflow?")
