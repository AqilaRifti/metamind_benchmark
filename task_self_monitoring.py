"""
MetaMind Benchmark - Task 4: Strategic Self-Monitoring in Multi-Step Reasoning

This task evaluates a model's ability to monitor its own reasoning process
during complex multi-step problems. It requires the model to:
1. Break down complex problems into steps
2. Self-monitor each step for potential errors
3. Request clarification when uncertain
4. Adjust strategy based on intermediate results

Key Innovation: Tests prospective metacognition - monitoring reasoning in progress
rather than just retrospective evaluation.
"""

import kaggle_benchmarks as kbench
import json
import re
from typing import Dict, List, Tuple

# Complex multi-step problems requiring self-monitoring
SELF_MONITORING_DATASET = [
    {
        "id": "monitor_001",
        "type": "multi_step_math",
        "difficulty": "medium",
        "question": "A store is having a 30% off sale. You buy a shirt originally priced at $40 and a pair of pants originally priced at $60. After the discount, you need to pay 8% sales tax. What is your final total?",
        "steps": [
            "Calculate original total: $40 + $60 = $100",
            "Apply 30% discount: $100 × 0.70 = $70",
            "Apply 8% tax: $70 × 1.08 = $75.60"
        ],
        "correct_answer": "$75.60",
        "common_errors": [
            "Apply discount to each item separately then forget tax",
            "Apply tax before discount",
            "Calculate discount as 30% of final price instead of original"
        ]
    },
    {
        "id": "monitor_002",
        "type": "constraint_satisfaction",
        "difficulty": "hard",
        "question": "Five people (Alice, Bob, Carol, Dave, Eve) need to sit in 5 chairs in a row. Alice must sit at one end. Bob and Carol refuse to sit next to each other. Dave must sit next to Eve. How many valid arrangements are possible?",
        "steps": [
            "Alice has 2 choices (either end)",
            "Dave and Eve form a block (2 arrangements: DE or ED)",
            "Place DE/ED block in remaining 4 positions",
            "Place Bob and Carol in remaining spots ensuring they're not adjacent",
            "Count valid arrangements"
        ],
        "correct_answer": "16",
        "common_errors": [
            "Forget Alice can be at either end",
            "Treat Dave-Eve as single unit without considering internal arrangement",
            "Double count or miss some valid arrangements"
        ]
    },
    {
        "id": "monitor_003",
        "type": "logical_deduction",
        "difficulty": "hard",
        "question": "Four friends (A, B, C, D) each have a different favorite color: red, blue, green, yellow. Clues: A doesn't like red or blue. B likes yellow. C doesn't like red. What is D's favorite color?",
        "steps": [
            "B = yellow (given)",
            "A can only be green (not red, not blue, yellow taken)",
            "C can be blue or green, but green taken, so C = blue",
            "D must have the remaining color: red"
        ],
        "correct_answer": "red",
        "common_errors": [
            "Assume colors must be in alphabetical order",
            "Miss that A is forced to green",
            "Incorrectly eliminate valid options"
        ]
    },
    {
        "id": "monitor_004",
        "type": "sequential_reasoning",
        "difficulty": "medium",
        "question": "A bacteria colony doubles every hour. At 6 PM, there are 1,024 bacteria. At what time were there exactly 128 bacteria?",
        "steps": [
            "Work backwards: 1024 → 512 → 256 → 128",
            "That's 3 hours earlier",
            "6 PM minus 3 hours = 3 PM"
        ],
        "correct_answer": "3 PM",
        "common_errors": [
            "Divide 1024 by 128 = 8, then go back 8 hours",
            "Calculate forward from some arbitrary time",
            "Misunderstand 'doubles every hour'"
        ]
    },
    {
        "id": "monitor_005",
        "type": "probability",
        "difficulty": "hard",
        "question": "You roll three fair six-sided dice. What is the probability that the sum is exactly 10?",
        "steps": [
            "Total outcomes: 6³ = 216",
            "Count combinations that sum to 10: (1,3,6), (1,4,5), (2,2,6), (2,3,5), (2,4,4), (3,3,4)",
            "Account for permutations: 6+6+3+6+3+3 = 27",
            "Probability = 27/216 = 1/8"
        ],
        "correct_answer": "1/8",
        "common_errors": [
            "Miss some combinations",
            "Forget to account for different permutations",
            "Count (2,2,6) as having 6 permutations instead of 3"
        ]
    },
    {
        "id": "monitor_006",
        "type": "pattern_recognition",
        "difficulty": "medium",
        "question": "Find the next number in the sequence: 1, 4, 10, 19, 31, 46, ?",
        "steps": [
            "Find differences: 3, 6, 9, 12, 15",
            "Pattern: differences increase by 3",
            "Next difference: 15 + 3 = 18",
            "Next number: 46 + 18 = 64"
        ],
        "correct_answer": "64",
        "common_errors": [
            "Look for multiplicative pattern",
            "Miss that differences themselves form a pattern",
            "Incorrectly calculate the next difference"
        ]
    },
    {
        "id": "monitor_007",
        "type": "word_problem",
        "difficulty": "hard",
        "question": "A train leaves Station A at 8:00 AM traveling at 60 mph toward Station B, 300 miles away. Another train leaves Station B at 9:00 AM traveling at 80 mph toward Station A. At what time do they meet?",
        "steps": [
            "First train travels alone for 1 hour: 60 miles covered",
            "Remaining distance: 240 miles",
            "Combined speed: 60 + 80 = 140 mph",
            "Time to meet: 240/140 = 12/7 hours ≈ 1.714 hours ≈ 1 hour 43 minutes",
            "Meeting time: 9:00 AM + 1:43 = 10:43 AM"
        ],
        "correct_answer": "10:43 AM",
        "common_errors": [
            "Ignore the 1-hour head start",
            "Use 300 miles instead of 240 for remaining distance",
            "Calculate time from 8:00 AM instead of 9:00 AM"
        ]
    },
    {
        "id": "monitor_008",
        "type": "set_operation",
        "difficulty": "medium",
        "question": "In a survey of 100 people: 60 like coffee, 45 like tea, and 30 like both. How many people like neither coffee nor tea?",
        "steps": [
            "Use inclusion-exclusion: |C ∪ T| = |C| + |T| - |C ∩ T|",
            "|C ∪ T| = 60 + 45 - 30 = 75",
            "Neither = 100 - 75 = 25"
        ],
        "correct_answer": "25",
        "common_errors": [
            "Simply add 60 + 45 = 105, then 100 - 105 = -5",
            "Double count the intersection",
            "Forget to subtract those who like both"
        ]
    }
]


def extract_final_answer(response: str) -> str:
    """Extract the final answer from a multi-step response."""
    # Look for explicit final answer
    patterns = [
        r'(?:final answer|answer)[:\s]+([^\n\.]+)',
        r'(?:therefore|thus|so)[:\s]+(?:the answer is )?([^\n\.]+)',
        r'(?:^|\n)(?:\d+\.\s*)?\$?([\d\./:]+)\s*$',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    # Return last substantial line
    lines = [l.strip() for l in response.split('\n') if l.strip()]
    for line in reversed(lines):
        if len(line) < 100 and (re.search(r'\d', line) or any(word in line.lower() for word in ['am', 'pm', 'red', 'blue', 'green', 'yellow'])):
            return line
    
    return ""


def check_step_monitoring(response: str) -> Dict:
    """
    Analyze whether the model shows evidence of step-by-step self-monitoring.
    Looks for explicit monitoring language.
    """
    response_lower = response.lower()
    
    monitoring_indicators = {
        "explicit_check": ["let me check", "let me verify", "checking", "verifying"],
        "uncertainty_expression": ["i think", "this seems", "this might", "possibly"],
        "error_acknowledgment": ["wait", "actually", "i made an error", "correction"],
        "step_validation": ["this step is correct", "this checks out", "this makes sense"],
        "alternative_consideration": ["alternatively", "another way", "or we could"]
    }
    
    monitoring_score = 0
    detected_indicators = {}
    
    for category, phrases in monitoring_indicators.items():
        detected = [p for p in phrases if p in response_lower]
        if detected:
            monitoring_score += 1
            detected_indicators[category] = detected
    
    return {
        "monitoring_score": monitoring_score,
        "max_possible": len(monitoring_indicators),
        "detected_indicators": detected_indicators,
        "shows_explicit_monitoring": monitoring_score >= 2
    }


def count_reasoning_steps(response: str) -> int:
    """Count the number of distinct reasoning steps in the response."""
    # Look for numbered steps
    numbered_steps = re.findall(r'(?:^|\n)(?:step\s*)?(\d+)[:\.\)]', response, re.IGNORECASE)
    if numbered_steps:
        return len(set(numbered_steps))
    
    # Look for bullet points or line breaks that indicate steps
    bullet_steps = re.findall(r'(?:^|\n)(?:[-•*]|\(\d+\)|\d+\.)\s+', response)
    if bullet_steps:
        return len(bullet_steps)
    
    # Count sentences that look like reasoning steps
    sentences = re.split(r'[.!?]+', response)
    reasoning_sentences = [s for s in sentences if len(s.strip()) > 20 and any(word in s.lower() for word in ['so', 'therefore', 'thus', 'then', 'next', 'first', 'second'])]
    
    return len(reasoning_sentences)


@kbench.task(name="strategic_self_monitoring")
def self_monitoring_task(llm, problem_data: dict):
    """
    Evaluates self-monitoring during multi-step reasoning.
    Prompts the model to show its work and self-monitor.
    """
    question = problem_data["question"]
    correct_answer = problem_data["correct_answer"]
    
    # Prompt that encourages step-by-step reasoning with self-monitoring
    prompt = f"""Solve this problem step by step. As you work through it:
1. Show each step clearly
2. Check your work as you go
3. If you're uncertain about a step, note it
4. Verify your final answer makes sense

Problem: {question}

Your step-by-step solution:"""
    
    response = llm.prompt(prompt)
    
    # Extract and check answer
    final_answer = extract_final_answer(response)
    is_correct = check_answer_match(final_answer, correct_answer)
    
    # Analyze self-monitoring
    monitoring_analysis = check_step_monitoring(response)
    
    # Count steps
    num_steps = count_reasoning_steps(response)
    expected_steps = len(problem_data["steps"])
    
    result = {
        "question_id": problem_data["id"],
        "question_type": problem_data["type"],
        "difficulty": problem_data["difficulty"],
        "final_answer": final_answer,
        "is_correct": is_correct,
        "num_steps_taken": num_steps,
        "expected_steps": expected_steps,
        "monitoring_score": monitoring_analysis["monitoring_score"],
        "max_monitoring_score": monitoring_analysis["max_possible"],
        "shows_explicit_monitoring": monitoring_analysis["shows_explicit_monitoring"],
        "detected_monitoring_indicators": list(monitoring_analysis["detected_indicators"].keys()),
        "step_adequacy": "ADEQUATE" if num_steps >= expected_steps - 1 else "INSUFFICIENT"
    }
    
    return result


def check_answer_match(response_answer: str, correct_answer: str) -> bool:
    """Check if extracted answer matches correct answer."""
    if not response_answer:
        return False
    
    response_clean = re.sub(r'[^\w\$\./:]', '', response_answer.lower())
    correct_clean = re.sub(r'[^\w\$\./:]', '', correct_answer.lower())
    
    return response_clean == correct_clean or correct_clean in response_clean or response_clean in correct_clean


def run_self_monitoring_benchmark(llm):
    """Run self-monitoring benchmark across all problems."""
    results = []
    
    for problem_data in SELF_MONITORING_DATASET:
        result = self_monitoring_task.run(llm, problem_data=problem_data)
        results.append(result)
    
    # Calculate metrics
    total = len(results)
    correct = sum(1 for r in results if r["is_correct"])
    show_monitoring = sum(1 for r in results if r["shows_explicit_monitoring"])
    adequate_steps = sum(1 for r in results if r["step_adequacy"] == "ADEQUATE")
    
    # Correlation between monitoring and correctness
    monitoring_correct = sum(1 for r in results if r["shows_explicit_monitoring"] and r["is_correct"])
    no_monitoring_correct = sum(1 for r in results if not r["shows_explicit_monitoring"] and r["is_correct"])
    
    monitoring_total = sum(1 for r in results if r["shows_explicit_monitoring"])
    no_monitoring_total = total - monitoring_total
    
    monitoring_accuracy = monitoring_correct / monitoring_total if monitoring_total > 0 else 0
    no_monitoring_accuracy = no_monitoring_correct / no_monitoring_total if no_monitoring_total > 0 else 0
    
    # Average monitoring score
    avg_monitoring_score = sum(r["monitoring_score"] for r in results) / total
    max_possible = results[0]["max_monitoring_score"] if results else 5
    
    summary = {
        "overall_accuracy": round(correct / total, 4),
        "explicit_monitoring_rate": round(show_monitoring / total, 4),
        "step_adequacy_rate": round(adequate_steps / total, 4),
        "average_monitoring_score": round(avg_monitoring_score, 4),
        "max_monitoring_score": max_possible,
        "monitoring_effectiveness": {
            "with_monitoring_accuracy": round(monitoring_accuracy, 4),
            "without_monitoring_accuracy": round(no_monitoring_accuracy, 4),
            "improvement": round(monitoring_accuracy - no_monitoring_accuracy, 4)
        },
        "total_problems": total,
        "detailed_results": results
    }
    
    return summary


if __name__ == "__main__":
    llm = kbench.llm
    summary = run_self_monitoring_benchmark(llm)
    print(json.dumps(summary, indent=2))
