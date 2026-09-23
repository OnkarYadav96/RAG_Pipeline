"""LLM-as-judge evaluators for LangSmith (Groq + wrappers)."""

import json
import os
from typing import Callable

from openai import OpenAI
from langsmith import wrappers

JUDGE_MODEL = "openai/gpt-oss-20b"

_groq_client = wrappers.wrap_openai(
    OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
)


def groq_judge(system_prompt: str, user_prompt: str, schema_hint: str) -> dict:
    response = _groq_client.chat.completions.create(
        model=JUDGE_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": system_prompt + "\nRespond with JSON only. Schema: " + schema_hint,
            },
            {"role": "user", "content": user_prompt},
        ],
    )
    return json.loads(response.choices[0].message.content)


def helpfulness(inputs: dict, outputs: dict) -> dict:
    result = groq_judge(
        system_prompt="You are an expert evaluator. Score helpfulness from 0 to 1.",
        user_prompt=f"Question:\n{inputs.get('question', '')}\n\nAnswer:\n{outputs.get('answer', '')}",
        schema_hint='{"score": number, "comment": string}',
    )
    return {"key": "helpfulness", "score": float(result["score"]), "comment": result.get("comment", "")}


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    result = groq_judge(
        system_prompt=(
            "Compare actual answer to expected answer. "
            "Return score 1 if semantically correct else 0."
        ),
        user_prompt=(
            f"Question: {inputs.get('question')}\n"
            f"Expected: {reference_outputs.get('answer')}\n"
            f"Actual: {outputs.get('answer')}"
        ),
        schema_hint='{"score": 0 or 1, "comment": string}',
    )
    return {"key": "correctness", "score": float(result["score"]), "comment": result.get("comment", "")}


def rag_faithfulness(inputs: dict, outputs: dict) -> dict:
    result = groq_judge(
        system_prompt=(
            "Check if the answer is supported by the context only. "
            "Score 0–1 (1 = fully grounded, 0 = hallucinated)."
        ),
        user_prompt=(
            f"Question: {inputs.get('question')}\n"
            f"Context: {outputs.get('context', '')}\n"
            f"Answer: {outputs.get('answer')}"
        ),
        schema_hint='{"score": number, "comment": string}',
    )
    return {"key": "faithfulness", "score": float(result["score"]), "comment": result.get("comment", "")}


def relevance(inputs: dict, outputs: dict) -> dict:
    result = groq_judge(
        system_prompt=(
            "Score how relevant the answer is to the question. "
            "1 = fully on-topic, 0 = unrelated."
        ),
        user_prompt=(
            f"Question: {inputs.get('question')}\n"
            f"Answer: {outputs.get('answer')}"
        ),
        schema_hint='{"score": number, "comment": string}',
    )
    return {"key": "relevance", "score": float(result["score"]), "comment": result.get("comment", "")}


def groundedness(inputs: dict, outputs: dict) -> dict:
    result = groq_judge(
        system_prompt=(
            "Score groundedness: is every claim in the answer supported by the context? "
            "1 = fully grounded, 0 = unsupported or hallucinated. "
            "If context is empty, score based on whether the answer avoids invented specifics."
        ),
        user_prompt=(
            f"Question: {inputs.get('question')}\n"
            f"Context: {outputs.get('context', '')}\n"
            f"Answer: {outputs.get('answer')}"
        ),
        schema_hint='{"score": number, "comment": string}',
    )
    return {"key": "groundedness", "score": float(result["score"]), "comment": result.get("comment", "")}


ALL_EVALUATORS: list[Callable] = [
    helpfulness,
    correctness,
    relevance,
    groundedness,
]
