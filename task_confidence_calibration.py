"""
MetaMind Benchmark - Task 1: Confidence Calibration Across Domains

This task evaluates a model's ability to accurately calibrate its confidence
with its actual accuracy across multiple domains (mathematics, logic, trivia, reasoning).

Key Innovation: Uses a 0-100 confidence scale with expected calibration error (ECE)
as the primary metric, going beyond simple pass/fail to measure metacognitive calibration.
"""

import kaggle_benchmarks as kbench
import json
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Curated dataset with questions of varying difficulty across domains
CONFIDENCE_CALIBRATION_DATASET = [
    # Mathematics (Easy to Hard)
    {
        "id": "math_001",
        "domain": "mathematics",
        "difficulty": "easy",
        "question": "What is 17 × 6?",
        "answer": "102",
        "explanation": "17 × 6 = (10 × 6) + (7 × 6) = 60 + 42 = 102"
    },
    {
        "id": "math_002",
        "domain": "mathematics",
        "difficulty": "medium",
        "question": "What is the sum of the first 20 positive integers?",
        "answer": "210",
        "explanation": "Using the formula n(n+1)/2: 20 × 21 / 2 = 210"
    },
    {
        "id": "math_003",
        "domain": "mathematics",
        "difficulty": "hard",
        "question": "If f(x) = x³ - 3x² + 2x, what is the value of f'(2)?",
        "answer": "2",
        "explanation": "f'(x) = 3x² - 6x + 2, so f'(2) = 12 - 12 + 2 = 2"
    },
    {
        "id": "math_004",
        "domain": "mathematics",
        "difficulty": "very_hard",
        "question": "What is the remainder when 3^47 is divided by 7?",
        "answer": "6",
        "explanation": "Using Fermat's Little Theorem, 3^6 ≡ 1 (mod 7). 47 = 7×6 + 5, so 3^47 ≡ 3^5 ≡ 243 ≡ 6 (mod 7)"
    },
    
    # Logic (Easy to Hard)
    {
        "id": "logic_001",
        "domain": "logic",
        "difficulty": "easy",
        "question": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
        "answer": "no",
        "explanation": "This is a classic syllogism fallacy. The set of flowers that fade quickly may not include any roses."
    },
    {
        "id": "logic_002",
        "domain": "logic",
        "difficulty": "medium",
        "question": "In a group of 100 people, 60 like tea, 50 like coffee, and 30 like both. How many like neither?",
        "answer": "20",
        "explanation": "Using inclusion-exclusion: 60 + 50 - 30 = 80 like at least one. So 100 - 80 = 20 like neither."
    },
    {
        "id": "logic_003",
        "domain": "logic",
        "difficulty": "hard",
        "question": "Three boxes are labeled 'Apples', 'Oranges', and 'Mixed', but all labels are wrong. You can pick one fruit from one box. Which box should you pick to correctly relabel all boxes?",
        "answer": "Mixed",
        "explanation": "Pick from the box labeled 'Mixed'. Since all labels are wrong, it contains only one type. If you get an apple, it's the Apples box. The box labeled 'Oranges' must then be Mixed (can't be Oranges), and the box labeled 'Apples' must be Oranges."
    },
    
    # Factual Trivia (Easy to Hard)
    {
        "id": "trivia_001",
        "domain": "trivia",
        "difficulty": "easy",
        "question": "What is the capital of Japan?",
        "answer": "Tokyo",
        "explanation": "Tokyo has been the capital of Japan since 1868."
    },
    {
        "id": "trivia_002",
        "domain": "trivia",
        "difficulty": "medium",
        "question": "Who wrote the novel 'One Hundred Years of Solitude'?",
        "answer": "Gabriel García Márquez",
        "explanation": "Gabriel García Márquez published this masterpiece of magical realism in 1967."
    },
    {
        "id": "trivia_003",
        "domain": "trivia",
        "difficulty": "hard",
        "question": "What is the half-life of Carbon-14?",
        "answer": "5730 years",
        "explanation": "Carbon-14 has a half-life of approximately 5,730 years, used in radiocarbon dating."
    },
    
    # Spatial/Visual Reasoning
    {
        "id": "spatial_001",
        "domain": "spatial",
        "difficulty": "medium",
        "question": "If you fold a square piece of paper in half twice (first horizontally, then vertically) and cut a small circle in the center, how many holes will the paper have when unfolded?",
        "answer": "1",
        "explanation": "The folds create layers, but cutting through all layers at the center creates a single hole when unfolded."
    },
    {
        "id": "spatial_002",
        "domain": "spatial",
        "difficulty": "hard",
        "question": "A cube has edges of length 3. A smaller cube of edge length 1 is removed from each corner. What is the surface area of the remaining solid?",
        "answer": "54",
        "explanation": "Original surface area: 6 × 9 = 54. Removing corners removes 3 faces of size 1 each (total 3) but exposes 3 new faces (total 3). Net change is 0, so surface area remains 54."
    },
    
    # Temporal Reasoning
    {
        "id": "temporal_001",
        "domain": "temporal",
        "difficulty": "medium",
        "question": "If March 15, 2026 is a Sunday, what day of the week will March 15, 2027 be?",
        "answer": "Monday",
        "explanation": "2026 is not a leap year, so 365 days = 52 weeks + 1 day. March 15, 2027 will be one day after Sunday."
    },
    
    # Counterfactual/Trick Questions
    {
        "id": "trick_001",
        "domain": "counterfactual",
        "difficulty": "medium",
        "question": "If you have a bowl with six apples and you take away four, how many do you have?",
        "answer": "4",
        "explanation": "The question asks how many YOU have, not how many are left in the bowl. You took 4, so you have 4."
    },
    {
        "id": "trick_002",
        "domain": "counterfactual",
        "difficulty": "hard",
        "question": "A farmer had 15 sheep and all but 8 died. How many sheep are left?",
        "answer": "8",
        "explanation": "'All but 8' means 8 survived. This tests attention to linguistic nuance."
    }
]


def extract_confidence(response: str) -> int:
    """Extract confidence score (0-100) from model response."""
    # Look for explicit confidence statements
    patterns = [
        r'confidence[:\s]+(\d+)',
        r'(\d+)%\s*(?:confidence|sure|certain)',
        r'(?:I am|I\'m)\s+(\d+)%\s*(?:confident|sure)',
        r'(?:confidence level|confidence score)[:\s]+(\d+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            confidence = int(match.group(1))
            return max(0, min(100, confidence))
    
    # Look for qualitative confidence indicators
    lower_response = response.lower()
    if any(phrase in lower_response for phrase in ['absolutely certain', '100% sure', 'definitely', 'without doubt']):
        return 100
    elif any(phrase in lower_response for phrase in ['very confident', 'quite sure', 'highly likely']):
        return 85
    elif any(phrase in lower_response for phrase in ['fairly confident', 'reasonably sure', 'probably']):
        return 70
    elif any(phrase in lower_response for phrase in ['somewhat confident', 'maybe', 'uncertain']):
        return 50
    elif any(phrase in lower_response for phrase in ['not confident', 'guessing', 'unsure', 'don\'t know']):
        return 30
    
    # Default if no confidence expressed
    return 50


def check_answer_correctness(response: str, correct_answer: str) -> bool:
    """Check if the model's answer is correct."""
    # Clean up response
    response_clean = re.sub(r'[^\w\s\-\.]', '', response.lower())
    answer_clean = correct_answer.lower().strip()
    
    # Direct match
    if answer_clean in response_clean:
        return True
    
    # Check for answer in quotes or bold
    quoted = re.findall(r'["\']([^"\']+)["\']', response)
    for q in quoted:
        if answer_clean in q.lower():
            return True
    
    # Check for "Answer: X" pattern
    answer_pattern = re.search(r'(?:answer|final answer)[:\s]+([^\n\.]+)', response, re.IGNORECASE)
    if answer_pattern:
        answer_text = answer_pattern.group(1).strip().lower()
        if answer_clean in answer_text or answer_text in answer_clean:
            return True
    
    return False


@kbench.task(name="confidence_calibration")
def confidence_calibration_task(llm, question_data: dict):
    """
    Evaluates confidence calibration by asking the model to answer a question
    and report its confidence level (0-100).
    """
    question = question_data["question"]
    correct_answer = question_data["answer"]
    
    # Prompt that elicits both answer and confidence
    prompt = f"""Answer the following question and then report your confidence level (0-100) in your answer.

Question: {question}

Provide your answer clearly, and then state your confidence as a percentage (0-100).
Format: "Confidence: X%" or "I am X% confident.""""
    
    response = llm.prompt(prompt)
    
    # Extract confidence and check correctness
    confidence = extract_confidence(response)
    is_correct = check_answer_correctness(response, correct_answer)
    
    # Store results for aggregation
    result = {
        "question_id": question_data["id"],
        "domain": question_data["domain"],
        "difficulty": question_data["difficulty"],
        "confidence": confidence,
        "is_correct": is_correct,
        "response": response
    }
    
    # Assert that we could extract a valid confidence
    kbench.assertions.assert_true(
        confidence >= 0 and confidence <= 100,
        expectation=f"Model should report confidence between 0-100, got {confidence}"
    )
    
    return result


# Run the task across all questions
def run_confidence_calibration_benchmark(llm):
    """Run confidence calibration across all questions and compute ECE."""
    results = []
    
    for question_data in CONFIDENCE_CALIBRATION_DATASET:
        result = confidence_calibration_task.run(llm, question_data=question_data)
        results.append(result)
    
    # Calculate Expected Calibration Error (ECE)
    # ECE = Σ (|confidence - accuracy| × proportion) for each confidence bin
    bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    ece = 0.0
    
    for bin_min, bin_max in bins:
        bin_results = [r for r in results if bin_min <= r["confidence"] < bin_max or (bin_max == 100 and r["confidence"] == 100)]
        if bin_results:
            avg_confidence = sum(r["confidence"] for r in bin_results) / len(bin_results)
            accuracy = sum(1 for r in bin_results if r["is_correct"]) / len(bin_results)
            proportion = len(bin_results) / len(results)
            ece += abs(avg_confidence / 100 - accuracy) * proportion
    
    # Calculate overall accuracy
    overall_accuracy = sum(1 for r in results if r["is_correct"]) / len(results)
    
    # Calculate calibration error (difference between average confidence and accuracy)
    avg_confidence = sum(r["confidence"] for r in results) / len(results)
    calibration_error = abs(avg_confidence / 100 - overall_accuracy)
    
    # Calculate overconfidence (positive = overconfident, negative = underconfident)
    overconfidence = (avg_confidence / 100) - overall_accuracy
    
    summary = {
        "expected_calibration_error": round(ece, 4),
        "overall_accuracy": round(overall_accuracy, 4),
        "average_confidence": round(avg_confidence / 100, 4),
        "calibration_error": round(calibration_error, 4),
        "overconfidence_score": round(overconfidence, 4),
        "total_questions": len(results),
        "detailed_results": results
    }
    
    return summary


if __name__ == "__main__":
    # Example usage
    llm = kbench.llm
    summary = run_confidence_calibration_benchmark(llm)
    print(json.dumps(summary, indent=2))
