"""
MetaMind: Comprehensive Metacognitive Evaluation Benchmark

A multi-task benchmark for evaluating metacognitive abilities in AI systems,
including confidence calibration, error detection, knowledge boundary awareness,
and strategic self-monitoring.

Track: Metacognition
Author: Aqila Hamizan Rifti
"""

import kaggle_benchmarks as kbench
from task_confidence_calibration import run_confidence_calibration_benchmark
from task_error_detection import run_error_detection_benchmark
from task_knowledge_boundaries import run_knowledge_boundary_benchmark
from task_self_monitoring import run_self_monitoring_benchmark
import json


@kbench.benchmark(
    name="MetaMind: Comprehensive Metacognitive Evaluation",
    description="""
    MetaMind evaluates four key dimensions of metacognition in AI systems:
    
    1. Confidence Calibration: Measures alignment between stated confidence and actual accuracy
    2. Error Detection & Self-Correction: Tests ability to recognize and fix own mistakes
    3. Knowledge Boundary Detection: Evaluates appropriate refusal on unanswerable questions
    4. Strategic Self-Monitoring: Assesses step-by-step reasoning monitoring
    
    This benchmark provides a detailed cognitive profile of a model's metacognitive capabilities,
    revealing how well it "knows what it knows" - a critical capability for reliable AI systems.
    """,
    version="1.0.0"
)
def metamind_benchmark(llm):
    """
    Main benchmark function that runs all metacognition tasks and aggregates results.
    """
    print("=" * 60)
    print("MetaMind: Comprehensive Metacognitive Evaluation")
    print("=" * 60)
    
    # Task 1: Confidence Calibration
    print("\n[1/4] Running Confidence Calibration Assessment...")
    calibration_results = run_confidence_calibration_benchmark(llm)
    
    # Task 2: Error Detection
    print("\n[2/4] Running Error Detection & Self-Correction Assessment...")
    error_detection_results = run_error_detection_benchmark(llm)
    
    # Task 3: Knowledge Boundaries
    print("\n[3/4] Running Knowledge Boundary Detection Assessment...")
    boundary_results = run_knowledge_boundary_benchmark(llm)
    
    # Task 4: Self-Monitoring
    print("\n[4/4] Running Strategic Self-Monitoring Assessment...")
    monitoring_results = run_self_monitoring_benchmark(llm)
    
    # Aggregate all results
    aggregated = aggregate_metamind_results(
        calibration_results,
        error_detection_results,
        boundary_results,
        monitoring_results
    )
    
    print("\n" + "=" * 60)
    print("MetaMind Evaluation Complete")
    print("=" * 60)
    
    return aggregated


def aggregate_metamind_results(calibration, error_detection, boundaries, monitoring):
    """
    Aggregate results from all tasks into a comprehensive metacognitive profile.
    """
    # Calculate overall metacognition score (weighted average)
    weights = {
        "confidence_calibration": 0.25,
        "error_detection": 0.30,
        "knowledge_boundaries": 0.25,
        "self_monitoring": 0.20
    }
    
    # Normalize scores to 0-1 scale
    calibration_score = 1 - calibration["expected_calibration_error"]  # Lower ECE is better
    error_detection_score = error_detection["error_detection_rate"] * error_detection["correction_success_rate"]
    boundary_score = boundaries["overall_accuracy"]
    monitoring_score = monitoring["overall_accuracy"] * (monitoring["explicit_monitoring_rate"] + 0.5) / 1.5
    
    overall_score = (
        weights["confidence_calibration"] * calibration_score +
        weights["error_detection"] * error_detection_score +
        weights["knowledge_boundaries"] * boundary_score +
        weights["self_monitoring"] * monitoring_score
    )
    
    # Create cognitive profile
    profile = {
        "overall_metacognition_score": round(overall_score, 4),
        "component_scores": {
            "confidence_calibration": round(calibration_score, 4),
            "error_detection_self_correction": round(error_detection_score, 4),
            "knowledge_boundary_awareness": round(boundary_score, 4),
            "strategic_self_monitoring": round(monitoring_score, 4)
        },
        "detailed_metrics": {
            "confidence_calibration": {
                "expected_calibration_error": calibration["expected_calibration_error"],
                "overall_accuracy": calibration["overall_accuracy"],
                "average_confidence": calibration["average_confidence"],
                "overconfidence_score": calibration["overconfidence_score"]
            },
            "error_detection": {
                "initial_accuracy": error_detection["initial_accuracy"],
                "final_accuracy": error_detection["final_accuracy"],
                "accuracy_improvement": error_detection["accuracy_improvement"],
                "error_detection_rate": error_detection["error_detection_rate"],
                "correction_success_rate": error_detection["correction_success_rate"],
                "metacognitive_profile": error_detection["metacognitive_profile"]
            },
            "knowledge_boundaries": {
                "overall_accuracy": boundaries["overall_accuracy"],
                "answerable_accuracy": boundaries["answerable_accuracy"],
                "boundary_recognition_accuracy": boundaries["boundary_recognition_accuracy"],
                "hallucination_rate": boundaries["hallucination_rate"],
                "over_refusal_rate": boundaries["over_refusal_rate"]
            },
            "self_monitoring": {
                "overall_accuracy": monitoring["overall_accuracy"],
                "explicit_monitoring_rate": monitoring["explicit_monitoring_rate"],
                "step_adequacy_rate": monitoring["step_adequacy_rate"],
                "average_monitoring_score": monitoring["average_monitoring_score"],
                "monitoring_effectiveness": monitoring["monitoring_effectiveness"]
            }
        },
        "interpretation": generate_interpretation(
            overall_score, calibration, error_detection, boundaries, monitoring
        )
    }
    
    return profile


def generate_interpretation(overall_score, calibration, error_detection, boundaries, monitoring):
    """Generate human-readable interpretation of results."""
    interpretations = []
    
    # Overall assessment
    if overall_score >= 0.8:
        interpretations.append("STRONG METACOGNITION: The model demonstrates robust self-awareness across all dimensions.")
    elif overall_score >= 0.6:
        interpretations.append("MODERATE METACOGNITION: The model shows reasonable self-awareness with room for improvement.")
    elif overall_score >= 0.4:
        interpretations.append("LIMITED METACOGNITION: The model struggles with self-monitoring and calibration.")
    else:
        interpretations.append("WEAK METACOGNITION: The model shows significant gaps in self-awareness.")
    
    # Specific insights
    if calibration["overconfidence_score"] > 0.2:
        interpretations.append("OVERCONFIDENCE: The model tends to be overconfident relative to its actual accuracy.")
    elif calibration["overconfidence_score"] < -0.1:
        interpretations.append("UNDERCONFIDENCE: The model tends to be underconfident relative to its actual accuracy.")
    else:
        interpretations.append("GOOD CALIBRATION: The model's confidence generally matches its accuracy.")
    
    if error_detection["error_detection_rate"] < 0.5:
        interpretations.append("BLIND SPOTS: The model often fails to recognize its own errors.")
    
    if boundaries["hallucination_rate"] > 0.3:
        interpretations.append("HALLUCINATION RISK: The model frequently answers unanswerable questions confidently.")
    
    if monitoring["explicit_monitoring_rate"] < 0.3:
        interpretations.append("LIMITED SELF-MONITORING: The model rarely shows explicit reasoning checks.")
    
    return " ".join(interpretations)


if __name__ == "__main__":
    # Run the benchmark
    llm = kbench.llm
    results = metamind_benchmark(llm)
    
    # Print formatted results
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))
