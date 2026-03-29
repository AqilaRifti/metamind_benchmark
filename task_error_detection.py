"""
MetaMind Benchmark - Task 2: Error Detection and Self-Correction

This task evaluates a model's ability to:
1. Provide an initial answer to a problem
2. Critically evaluate its own answer for errors
3. Correct any identified errors

Key Innovation: Tests retrospective metacognition - the ability to monitor and
correct one's own reasoning, which is crucial for reliable AI systems.
"""

import kaggle_benchmarks as kbench
import json
import re
from typing import Dict, List, Tuple, Optional

# Dataset designed to elicit common reasoning errors
ERROR_DETECTION_DATASET = [
    {
        "id": "error_001",
        "type": "arithmetic",
        "difficulty": "easy",
        "question": "Calculate: 15 + 27 × 3",
        "common_error": "126",
        "correct_answer": "96",
        "error_type": "order_of_operations",
        "explanation": "Following order of operations: 27 × 3 = 81, then 15 + 81 = 96. Common error: (15 + 27) × 3 = 126."
    },
    {
        "id": "error_002",
        "type": "algebra",
        "difficulty": "medium",
        "question": "Solve for x: 2(x - 3) = 10",
        "common_error": "8",
        "correct_answer": "8",
        "error_type": "distribution",
        "explanation": "2x - 6 = 10, so 2x = 16, x = 8."
    },
    {
        "id": "error_003",
        "type": "logic",
        "difficulty": "medium",
        "question": "A bat and a ball cost $11 in total. The bat costs $10 more than the ball. How much does the ball cost?",
        "common_error": "$1",
        "correct_answer": "$0.50",
        "error_type": "intuitive_bias",
        "explanation": "If ball = $0.50, bat = $10.50, total = $11. Common intuitive error: ball = $1, bat = $10 (only $9 more)."
    },
    {
        "id": "error_004",
        "type": "probability",
        "difficulty": "hard",
        "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
        "common_error": "100 minutes",
        "correct_answer": "5 minutes",
        "error_type": "proportionality_bias",
        "explanation": "Each machine makes 1 widget in 5 minutes. 100 machines make 100 widgets in the same 5 minutes."
    },
    {
        "id": "error_005",
        "type": "counting",
        "difficulty": "medium",
        "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
        "common_error": "24 days",
        "correct_answer": "47 days",
        "error_type": "halving_bias",
        "explanation": "Since the patch doubles daily, it was half the size one day before covering the whole lake: 48 - 1 = 47 days."
    },
    {
        "id": "error_006",
        "type": "word_problem",
        "difficulty": "hard",
        "question": "John's father has 5 children: Winter, Spring, Summer, Autumn, and...?",
        "common_error": "Fall",
        "correct_answer": "John",
        "error_type": "pattern_completion",
        "explanation": "The question states 'John's father', so John is the fifth child."
    },
    {
        "id": "error_007",
        "type": "spatial",
        "difficulty": "medium",
        "question": "You are running a race and you pass the person in 2nd place. What place are you in now?",
        "common_error": "1st place",
        "correct_answer": "2nd place",
        "error_type": "first_place_bias",
        "explanation": "You passed the person in 2nd place, so you take their position - you are now in 2nd place."
    },
    {
        "id": "error_008",
        "type": "algebra",
        "difficulty": "hard",
        "question": "If 3 cats can catch 3 mice in 3 minutes, how many cats are needed to catch 100 mice in 100 minutes?",
        "common_error": "100 cats",
        "correct_answer": "3 cats",
        "error_type": "linear_scaling",
        "explanation": "3 cats catch 1 mouse per minute (3 mice in 3 min). In 100 minutes, 3 cats catch 100 mice."
    },
    {
        "id": "error_009",
        "type": "set_theory",
        "difficulty": "medium",
        "question": "A set contains all integers from 1 to 100. How many numbers contain the digit '7'?",
        "common_error": "10",
        "correct_answer": "19",
        "error_type": "boundary_counting",
        "explanation": "7, 17, 27, 37, 47, 57, 67, 70-79 (10 numbers), 87, 97 = 19 numbers. Common error: forgetting 70-79 contains 10 numbers, not 1."
    },
    {
        "id": "error_010",
        "type": "rate",
        "difficulty": "hard",
        "question": "A car travels from A to B at 30 mph and returns from B to A at 60 mph. What is the average speed for the entire trip?",
        "common_error": "45 mph",
        "correct_answer": "40 mph",
        "error_type": "arithmetic_mean_bias",
        "explanation": "Average speed = total distance / total time. For distance d: 2d / (d/30 + d/60) = 2 / (1/30 + 1/60) = 2 / (3/60) = 40 mph."
    }
]


def extract_answer(response: str) -> Optional[str]:
    """Extract the answer from model response."""
    # Look for explicit answer patterns
    patterns = [
        r'(?:final answer|answer)[:\s]+([^\n\.]+)',
        r'(?:therefore|so|thus)[:\s]+([^\n\.]+)',
        r'\$?([\d\.]+)\s*(?:dollars?|mph)?\s*$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    # Return last line if it looks like an answer
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        # Check if last line is short and looks like an answer
        if len(last_line) < 50 and (re.search(r'\d', last_line) or '$' in last_line):
            return last_line
    
    return None


def check_answer_match(response_answer: str, correct_answer: str) -> bool:
    """Check if extracted answer matches correct answer."""
    if not response_answer:
        return False
    
    response_clean = re.sub(r'[^\w\$\.]', '', response_answer.lower())
    correct_clean = re.sub(r'[^\w\$\.]', '', correct_answer.lower())
    
    return response_clean == correct_clean or correct_clean in response_clean


@kbench.task(name="error_detection_self_correction")
def error_detection_task(llm, problem_data: dict):
    """
    Three-phase task:
    1. Initial answer
    2. Self-evaluation for errors
    3. Corrected answer (if errors found)
    """
    question = problem_data["question"]
    correct_answer = problem_data["correct_answer"]
    error_type = problem_data["error_type"]
    
    # Phase 1: Initial Answer
    initial_prompt = f"""Solve this problem. Show your work and provide a final answer.

Problem: {question}

Provide your answer clearly."""
    
    initial_response = llm.prompt(initial_prompt)
    initial_answer = extract_answer(initial_response)
    initial_correct = check_answer_match(initial_answer, correct_answer)
    
    # Phase 2: Self-Evaluation
    evaluation_prompt = f"""You just answered: {question}

Your answer was: {initial_answer}

Now, critically evaluate your answer. Look for potential errors in:
- Arithmetic calculations
- Logical reasoning
- Assumptions made
- Alternative interpretations

Do you think your answer is correct? If not, what error might you have made?
Respond with either:
"My answer is CORRECT" or "My answer is INCORRECT because...""""
    
    evaluation_response = llm.prompt(evaluation_prompt)
    evaluation_lower = evaluation_response.lower()
    
    # Determine if model detected an error
    detected_error = "incorrect" in evaluation_lower or "error" in evaluation_lower
    
    # Phase 3: Final Answer
    if detected_error:
        correction_prompt = f"""Based on your error analysis, provide your corrected final answer to:

{question}

Your corrected answer:"""
        final_response = llm.prompt(correction_prompt)
        final_answer = extract_answer(final_response)
    else:
        final_answer = initial_answer
        final_response = initial_response
    
    final_correct = check_answer_match(final_answer, correct_answer)
    
    # Analyze metacognitive performance
    result = {
        "question_id": problem_data["id"],
        "question_type": problem_data["type"],
        "difficulty": problem_data["difficulty"],
        "error_type": error_type,
        "initial_answer": initial_answer,
        "initial_correct": initial_correct,
        "detected_error": detected_error,
        "final_answer": final_answer,
        "final_correct": final_correct,
        "metacognitive_performance": classify_metacognitive_performance(
            initial_correct, detected_error, final_correct
        )
    }
    
    return result


def classify_metacognitive_performance(initial_correct: bool, detected_error: bool, final_correct: bool) -> str:
    """
    Classify the model's metacognitive performance:
    - OPTIMAL: Correct initially, maintained confidence (no false correction)
    - SELF_CORRECTED: Incorrect initially, detected error, corrected successfully
    - OVERCONFIDENT: Incorrect initially, didn't detect error
    - UNDERCONFIDENT: Correct initially, falsely detected error and changed to wrong answer
    - FAILED_CORRECTION: Incorrect initially, detected error but still wrong
    """
    if initial_correct and final_correct and not detected_error:
        return "OPTIMAL"
    elif initial_correct and not final_correct:
        return "UNDERCONFIDENT"
    elif not initial_correct and detected_error and final_correct:
        return "SELF_CORRECTED"
    elif not initial_correct and not detected_error:
        return "OVERCONFIDENT"
    elif not initial_correct and detected_error and not final_correct:
        return "FAILED_CORRECTION"
    else:
        return "UNKNOWN"


def run_error_detection_benchmark(llm):
    """Run error detection benchmark across all problems."""
    results = []
    
    for problem_data in ERROR_DETECTION_DATASET:
        result = error_detection_task.run(llm, problem_data=problem_data)
        results.append(result)
    
    # Calculate metacognitive metrics
    total = len(results)
    optimal = sum(1 for r in results if r["metacognitive_performance"] == "OPTIMAL")
    self_corrected = sum(1 for r in results if r["metacognitive_performance"] == "SELF_CORRECTED")
    overconfident = sum(1 for r in results if r["metacognitive_performance"] == "OVERCONFIDENT")
    underconfident = sum(1 for r in results if r["metacognitive_performance"] == "UNDERCONFIDENT")
    failed_correction = sum(1 for r in results if r["metacognitive_performance"] == "FAILED_CORRECTION")
    
    initial_accuracy = sum(1 for r in results if r["initial_correct"]) / total
    final_accuracy = sum(1 for r in results if r["final_correct"]) / total
    accuracy_improvement = final_accuracy - initial_accuracy
    
    # Error detection rate (of incorrect initial answers, how many were detected)
    incorrect_initial = [r for r in results if not r["initial_correct"]]
    error_detection_rate = sum(1 for r in incorrect_initial if r["detected_error"]) / len(incorrect_initial) if incorrect_initial else 0
    
    # Correction success rate (of detected errors, how many were corrected)
    detected_errors = [r for r in results if r["detected_error"]]
    correction_success_rate = sum(1 for r in detected_errors if r["final_correct"]) / len(detected_errors) if detected_errors else 0
    
    summary = {
        "initial_accuracy": round(initial_accuracy, 4),
        "final_accuracy": round(final_accuracy, 4),
        "accuracy_improvement": round(accuracy_improvement, 4),
        "error_detection_rate": round(error_detection_rate, 4),
        "correction_success_rate": round(correction_success_rate, 4),
        "metacognitive_profile": {
            "optimal": optimal,
            "self_corrected": self_corrected,
            "overconfident": overconfident,
            "underconfident": underconfident,
            "failed_correction": failed_correction
        },
        "total_problems": total,
        "detailed_results": results
    }
    
    return summary


if __name__ == "__main__":
    llm = kbench.llm
    summary = run_error_detection_benchmark(llm)
    print(json.dumps(summary, indent=2))
