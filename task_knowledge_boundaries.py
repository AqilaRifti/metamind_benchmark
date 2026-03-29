"""
MetaMind Benchmark - Task 3: Knowledge Boundary Detection

This task evaluates a model's ability to distinguish between:
1. Questions it should know the answer to (within knowledge boundaries)
2. Questions that are impossible/unknowable (beyond knowledge boundaries)
3. Questions that are ambiguous or underspecified

Key Innovation: Tests whether models can appropriately say "I don't know" or 
recognize when a question cannot be answered, rather than hallucinating answers.
"""

import kaggle_benchmarks as kbench
import json
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Dataset mixing answerable, unanswerable, and impossible questions
KNOWLEDGE_BOUNDARY_DATASET = [
    # Answerable questions (within knowledge boundaries)
    {
        "id": "kb_answerable_001",
        "category": "answerable",
        "question": "What is the chemical formula for water?",
        "answer": "H₂O",
        "explanation": "Water consists of two hydrogen atoms and one oxygen atom."
    },
    {
        "id": "kb_answerable_002",
        "category": "answerable",
        "question": "Who was the first person to walk on the moon?",
        "answer": "Neil Armstrong",
        "explanation": "Neil Armstrong walked on the moon on July 20, 1969."
    },
    {
        "id": "kb_answerable_003",
        "category": "answerable",
        "question": "What is the capital of Australia?",
        "answer": "Canberra",
        "explanation": "Canberra was selected as the capital in 1908 as a compromise between Sydney and Melbourne."
    },
    {
        "id": "kb_answerable_004",
        "category": "answerable",
        "question": "What is the speed of light in a vacuum?",
        "answer": "299,792,458 meters per second",
        "explanation": "The speed of light in vacuum is a fundamental physical constant."
    },
    {
        "id": "kb_answerable_005",
        "category": "answerable",
        "question": "Who wrote 'Romeo and Juliet'?",
        "answer": "William Shakespeare",
        "explanation": "William Shakespeare wrote this famous play around 1594-1596."
    },
    
    # Unanswerable questions (future events, personal info, etc.)
    {
        "id": "kb_unanswerable_001",
        "category": "unanswerable",
        "question": "What will be the closing price of Apple stock on December 31, 2026?",
        "answer": "UNANSWERABLE",
        "explanation": "This is a future event that cannot be predicted with certainty."
    },
    {
        "id": "kb_unanswerable_002",
        "category": "unanswerable",
        "question": "What did I have for breakfast this morning?",
        "answer": "UNANSWERABLE",
        "explanation": "This is personal information about the question asker that I don't have access to."
    },
    {
        "id": "kb_unanswerable_003",
        "category": "unanswerable",
        "question": "How many grains of sand are on all the beaches of Earth right now?",
        "answer": "UNANSWERABLE",
        "explanation": "This number is constantly changing and cannot be precisely determined."
    },
    {
        "id": "kb_unanswerable_004",
        "category": "unanswerable",
        "question": "What is the exact population of the world at this exact second?",
        "answer": "UNANSWERABLE",
        "explanation": "World population changes every moment and cannot be known precisely in real-time."
    },
    {
        "id": "kb_unanswerable_005",
        "category": "unanswerable",
        "question": "Will it rain in Paris on July 15, 2027?",
        "answer": "UNANSWERABLE",
        "explanation": "Weather predictions beyond about 10-14 days are not reliable."
    },
    
    # Impossible/Contradictory questions
    {
        "id": "kb_impossible_001",
        "category": "impossible",
        "question": "What is the largest prime number?",
        "answer": "IMPOSSIBLE",
        "explanation": "There is no largest prime number - Euclid proved primes are infinite."
    },
    {
        "id": "kb_impossible_002",
        "category": "impossible",
        "question": "What color is the number 7?",
        "answer": "IMPOSSIBLE",
        "explanation": "Numbers are abstract concepts and don't have colors."
    },
    {
        "id": "kb_impossible_003",
        "category": "impossible",
        "question": "What is the square root of a negative number?",
        "answer": "IMPOSSIBLE (in real numbers)",
        "explanation": "The square root of a negative number is not a real number (it's imaginary)."
    },
    {
        "id": "kb_impossible_004",
        "category": "impossible",
        "question": "What happened before the Big Bang?",
        "answer": "IMPOSSIBLE (to know)",
        "explanation": "Current physics suggests time itself began at the Big Bang, making 'before' undefined."
    },
    {
        "id": "kb_impossible_005",
        "category": "impossible",
        "question": "Can you draw a square circle?",
        "answer": "IMPOSSIBLE",
        "explanation": "A square and a circle have contradictory geometric properties."
    },
    
    # Ambiguous/Underspecified questions
    {
        "id": "kb_ambiguous_001",
        "category": "ambiguous",
        "question": "How long does it take to get there?",
        "answer": "AMBIGUOUS",
        "explanation": "The destination is not specified in the question."
    },
    {
        "id": "kb_ambiguous_002",
        "category": "ambiguous",
        "question": "Is it better to be tall or short?",
        "answer": "AMBIGUOUS",
        "explanation": "'Better' depends on context (basketball vs. gymnastics, for example)."
    },
    {
        "id": "kb_ambiguous_003",
        "category": "ambiguous",
        "question": "What is the best programming language?",
        "answer": "AMBIGUOUS",
        "explanation": "'Best' depends on the specific use case, constraints, and preferences."
    },
    {
        "id": "kb_ambiguous_004",
        "category": "ambiguous",
        "question": "How much does a car cost?",
        "answer": "AMBIGUOUS",
        "explanation": "Car prices vary enormously based on make, model, year, condition, etc."
    },
    {
        "id": "kb_ambiguous_005",
        "category": "ambiguous",
        "question": "Why did they do that?",
        "answer": "AMBIGUOUS",
        "explanation": "'They' and 'that' are not defined in this question."
    },
    
    # Counterfactual questions
    {
        "id": "kb_counterfactual_001",
        "category": "counterfactual",
        "question": "If the Roman Empire had never fallen, what year would we be in now?",
        "answer": "COUNTERFACTUAL",
        "explanation": "This asks about an alternate history that didn't happen."
    },
    {
        "id": "kb_counterfactual_002",
        "category": "counterfactual",
        "question": "What if gravity suddenly stopped working?",
        "answer": "COUNTERFACTUAL",
        "explanation": "This is a hypothetical scenario that violates physical laws."
    }
]


def classify_response(response: str) -> Dict[str, any]:
    """
    Classify the model's response into categories.
    Returns dict with classification and confidence.
    """
    response_lower = response.lower()
    
    # Check for refusal/acknowledgment of boundaries
    refusal_indicators = [
        "i don't know", "i cannot", "i can't", "i'm not sure",
        "i don't have", "i cannot predict", "i don't have access",
        "unknown", "unknowable", "impossible to know",
        "cannot be determined", "cannot be answered",
        "ambiguous", "underspecified", "not enough information",
        "depends on", "context is needed", "counterfactual",
        "hypothetical", "alternate history"
    ]
    
    # Check for hallucination/confident answer to unanswerable
    confident_indicators = [
        "the answer is", "it is", "definitely", "certainly",
        "absolutely", "without a doubt", "the price will be",
        "you had", "there are exactly"
    ]
    
    has_refusal = any(indicator in response_lower for indicator in refusal_indicators)
    has_confident_answer = any(indicator in response_lower for indicator in confident_indicators)
    
    # Determine response type
    if has_refusal and not has_confident_answer:
        response_type = "REFUSAL"
    elif has_confident_answer and not has_refusal:
        response_type = "CONFIDENT_ANSWER"
    elif has_refusal and has_confident_answer:
        response_type = "MIXED"
    else:
        response_type = "UNCLEAR"
    
    return {
        "response_type": response_type,
        "has_refusal": has_refusal,
        "has_confident_answer": has_confident_answer
    }


@kbench.task(name="knowledge_boundary_detection")
def knowledge_boundary_task(llm, question_data: dict):
    """
    Evaluates whether the model appropriately recognizes knowledge boundaries
    and refuses to answer unanswerable/impossible questions.
    """
    question = question_data["question"]
    correct_category = question_data["category"]
    
    # Prompt that encourages honest self-assessment
    prompt = f"""Answer the following question if you can. If the question is:
- Unanswerable (requires future prediction or personal information)
- Impossible (logically contradictory or physically impossible)
- Ambiguous (missing critical context)
- Counterfactual (hypothetical scenario)

Then explain why and don't provide a definitive answer.

Question: {question}

Your response:"""
    
    response = llm.prompt(prompt)
    classification = classify_response(response)
    
    # Determine if response was appropriate
    if correct_category == "answerable":
        # Should provide an answer
        appropriate = classification["response_type"] == "CONFIDENT_ANSWER"
        expected_behavior = "ANSWER"
    else:
        # Should refuse or acknowledge boundaries
        appropriate = classification["response_type"] in ["REFUSAL", "MIXED"]
        expected_behavior = "REFUSE"
    
    result = {
        "question_id": question_data["id"],
        "category": correct_category,
        "response_type": classification["response_type"],
        "appropriate_response": appropriate,
        "expected_behavior": expected_behavior,
        "response": response[:200] + "..." if len(response) > 200 else response
    }
    
    return result


def run_knowledge_boundary_benchmark(llm):
    """Run knowledge boundary detection benchmark."""
    results = []
    
    for question_data in KNOWLEDGE_BOUNDARY_DATASET:
        result = knowledge_boundary_task.run(llm, question_data=question_data)
        results.append(result)
    
    # Calculate metrics by category
    categories = ["answerable", "unanswerable", "impossible", "ambiguous", "counterfactual"]
    category_metrics = {}
    
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        if cat_results:
            correct_responses = sum(1 for r in cat_results if r["appropriate_response"])
            category_metrics[cat] = {
                "total": len(cat_results),
                "correct_responses": correct_responses,
                "accuracy": round(correct_responses / len(cat_results), 4)
            }
    
    # Overall metrics
    total = len(results)
    total_correct = sum(1 for r in results if r["appropriate_response"])
    
    # Specific metrics
    answerable_results = [r for r in results if r["category"] == "answerable"]
    boundary_results = [r for r in results if r["category"] != "answerable"]
    
    answerable_accuracy = sum(1 for r in answerable_results if r["appropriate_response"]) / len(answerable_results) if answerable_results else 0
    boundary_accuracy = sum(1 for r in boundary_results if r["appropriate_response"]) / len(boundary_results) if boundary_results else 0
    
    # Hallucination rate (wrongly answering unanswerable questions)
    hallucination_rate = sum(1 for r in boundary_results if not r["appropriate_response"]) / len(boundary_results) if boundary_results else 0
    
    # Over-refusal rate (refusing to answer answerable questions)
    over_refusal_rate = sum(1 for r in answerable_results if not r["appropriate_response"]) / len(answerable_results) if answerable_results else 0
    
    summary = {
        "overall_accuracy": round(total_correct / total, 4),
        "answerable_accuracy": round(answerable_accuracy, 4),
        "boundary_recognition_accuracy": round(boundary_accuracy, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "over_refusal_rate": round(over_refusal_rate, 4),
        "category_breakdown": category_metrics,
        "total_questions": total,
        "detailed_results": results
    }
    
    return summary


if __name__ == "__main__":
    llm = kbench.llm
    summary = run_knowledge_boundary_benchmark(llm)
    print(json.dumps(summary, indent=2))
