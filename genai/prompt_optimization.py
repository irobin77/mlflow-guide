import os
from typing import Any
import mlflow
from mlflow.genai.scorers import scorer
from mlflow.genai.optimize import OptimizerConfig, LLMParams


os.environ["MISTRAL_API_KEY"] = ""
mlflow.set_tracking_uri("http://localhost:5000")


# Define a custom scorer function to evaluate prompt performance with the @scorer decorator.
# The scorer function for optimization can take inputs, outputs, and expectations arguments, but not the trace argument.
# Note that the DSPy/MIPROv2 optimizer requires metrics to receive outputs as a dict.
@scorer
def exact_match(expectations: dict[str, Any], outputs: dict[str, Any]) -> bool:
    return expectations["answer"] == outputs["answer"]


# Register the initial prompt
initial_template = """
Answer to this math question: {{question}}.
Return the result in a JSON string in the format of {"answer": "xxx"}.
"""

prompt = mlflow.genai.register_prompt(
    name="math",
    template=initial_template
)

# The data can be a list of dictionaries, a pandas DataFrame, or an mlflow.genai.EvaluationDataset
# It needs to contain inputs and expectations where each row is a dictionary.
train_data = [
    {
        "inputs": {"question": "Given that $y=3$, evaluate $(1+y)^y$."},
        "expectations": {"answer": "64"},
    },
    {
        "inputs": {
            "question": "The midpoint of the line segment between $(x,y)$ and $(-9,1)$ is $(3,-5)$. Find $(x,y)$."
        },
        "expectations": {"answer": "(15,-11)"},
    },
    {
        "inputs": {
            "question": "What is the value of $b$ if $5^b + 5^b + 5^b + 5^b + 5^b = 625^{(b-1)}$? Express your answer as a common fraction."
        },
        "expectations": {"answer": "\\frac{5}{3}"},
    },
    {
        "inputs": {"question": "Evaluate the expression $a^3\\cdot a^2$ if $a= 5$."},
        "expectations": {"answer": "3125"},
    },
    {
        "inputs": {"question": "Evaluate $\\lceil 8.8 \\rceil+\\lceil -8.8 \\rceil$."},
        "expectations": {"answer": "17"},
    },
]

eval_data = [
    {
        "inputs": {
            "question": "The sum of 27 consecutive positive integers is $3^7$. What is their median?"
        },
        "expectations": {"answer": "81"},
    },
    {
        "inputs": {"question": "What is the value of $x$ if $x^2 - 10x + 25 = 0$?"},
        "expectations": {"answer": "5"},
    },
    {
        "inputs": {
            "question": "If $a\\ast b = 2a+5b-ab$, what is the value of $3\\ast10$?"
        },
        "expectations": {"answer": "26"},
    },
    {
        "inputs": {
            "question": "Given that $-4$ is a solution to $x^2 + bx -36 = 0$, what is the value of $b$?"
        },
        "expectations": {"answer": "-5"},
    },
]

# Optimize the prompt
result = mlflow.genai.optimize_prompt(
    target_llm_params=LLMParams(model_name="mistral:/mistral-tiny"),
    prompt=prompt,
    train_data=train_data,
    eval_data=eval_data,
    scorers=[exact_match],
    optimizer_config=OptimizerConfig(
        num_instruction_candidates=8,
        max_few_show_examples=2,
    ),
)

# The optimized prompt is automatically registered as a new version
print(result.prompt.uri)
