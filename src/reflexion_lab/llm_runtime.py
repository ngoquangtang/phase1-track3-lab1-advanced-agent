import os
import json
import time
from typing import List
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

from .schemas import QAExample, JudgeResult, ReflectionEntry
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM

load_dotenv()

# Cấu hình client - Mặc định dùng OpenAI hoặc Ollama nếu có biến môi trường
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "your-api-key-here"),
    base_url=os.getenv("OPENAI_BASE_URL", None)
)

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def call_llm(system_prompt: str, user_prompt: str, model: str = "gpt-3.5-turbo", json_mode: bool = False) -> tuple[str, int, int]:
    start_time = time.time()
    
    response_format = {"type": "json_object"} if json_mode else None
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format=response_format
    )
    
    latency_ms = int((time.time() - start_time) * 1000)
    content = response.choices[0].message.content
    tokens = response.usage.total_tokens
    
    return content, tokens, latency_ms

def actor_answer(example: QAExample, attempt_id: int, agent_type: str, reflection_memory: List[str]) -> tuple[str, int, int]:
    context_str = "\n".join([f"Source: {c.title}\nContent: {c.text}" for c in example.context])
    
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {example.question}\n"
    
    if agent_type == "reflexion" and reflection_memory:
        history = "\n".join([f"- Attempt {i+1} reflection: {r}" for i, r in enumerate(reflection_memory)])
        user_prompt += f"\nPrevious failed attempts history:\n{history}\nPlease use the strategies above to avoid repeating mistakes."

    content, tokens, latency = call_llm(ACTOR_SYSTEM, user_prompt)
    return content, tokens, latency

def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, int, int]:
    user_prompt = f"Gold Answer: {example.gold_answer}\nAssistant Answer: {answer}\nContext:\n{str(example.context)}"
    
    content, tokens, latency = call_llm(EVALUATOR_SYSTEM, user_prompt, json_mode=True)
    
    try:
        data = json.loads(content)
        result = JudgeResult(
            score=data.get("score", 0),
            reason=data.get("reason", "No reason provided"),
            missing_evidence=data.get("missing_evidence", []),
            spurious_claims=data.get("spurious_claims", [])
        )
    except Exception as e:
        result = JudgeResult(score=0, reason=f"Failed to parse judge output: {str(e)}")
        
    return result, tokens, latency

def reflector(example: QAExample, attempt_id: int, judge: JudgeResult) -> tuple[ReflectionEntry, int, int]:
    user_prompt = f"Question: {example.question}\nGold Answer: {example.gold_answer}\nJudge Feedback: {judge.reason}\nMissing facts: {judge.missing_evidence}"
    
    content, tokens, latency = call_llm(REFLECTOR_SYSTEM, user_prompt, json_mode=True)
    
    try:
        data = json.loads(content)
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=data.get("failure_reason", ""),
            lesson=data.get("lesson", ""),
            next_strategy=data.get("next_strategy", "")
        )
    except Exception as e:
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason="Reflection failed",
            lesson="Error in reflector",
            next_strategy="Try to be more careful."
        )
        
    return entry, tokens, latency
