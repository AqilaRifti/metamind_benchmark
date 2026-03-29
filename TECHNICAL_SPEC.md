# MetaMind Technical Specification

## Benchmark Architecture

### Task 1: Confidence Calibration

**Algorithm**:
1. Present question to model with confidence elicitation prompt
2. Extract confidence score (0-100) using regex patterns
3. Check answer correctness using exact/regex matching
4. Compute Expected Calibration Error (ECE) across 5 confidence bins

**Metrics**:
- Expected Calibration Error (ECE)
- Overall accuracy
- Average confidence
- Overconfidence score (avg_confidence - accuracy)

**Validation**: All answers verified against ground truth with unambiguous correct responses.

### Task 2: Error Detection

**Algorithm**:
1. Phase 1: Generate initial answer
2. Phase 2: Explicit self-evaluation prompt ("Is your answer correct?")
3. Phase 3: If error detected, generate corrected answer
4. Classify into metacognitive categories

**Classification Logic**:
```python
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
```

**Metrics**:
- Initial accuracy
- Final accuracy
- Accuracy improvement
- Error detection rate
- Correction success rate
- Metacognitive profile distribution

### Task 3: Knowledge Boundaries

**Algorithm**:
1. Present question with boundary-aware prompt
2. Classify response as REFUSAL, CONFIDENT_ANSWER, MIXED, or UNCLEAR
3. Compare to expected behavior (ANSWER vs. REFUSE)

**Response Classification**:
- REFUSAL: Contains refusal indicators, no confident answer
- CONFIDENT_ANSWER: Contains confident indicators, no refusal
- MIXED: Both refusal and confident answer
- UNCLEAR: Neither pattern detected

**Metrics**:
- Overall accuracy
- Answerable accuracy (should answer)
- Boundary recognition accuracy (should refuse)
- Hallucination rate
- Over-refusal rate

### Task 4: Self-Monitoring

**Algorithm**:
1. Present complex problem with monitoring encouragement
2. Extract final answer
3. Analyze response for monitoring indicators
4. Count reasoning steps

**Monitoring Indicators**:
- explicit_check: "let me check", "verifying"
- uncertainty_expression: "I think", "this seems"
- error_acknowledgment: "wait", "actually", "correction"
- step_validation: "this step is correct", "this checks out"
- alternative_consideration: "alternatively", "another way"

**Metrics**:
- Overall accuracy
- Explicit monitoring rate
- Step adequacy rate
- Average monitoring score
- Monitoring effectiveness (accuracy with vs. without monitoring)

## Aggregation Logic

**Component Score Calculation**:
```python
calibration_score = 1 - ECE  # Lower ECE is better
error_detection_score = detection_rate * correction_success_rate
boundary_score = overall_accuracy
monitoring_score = accuracy * (monitoring_rate + 0.5) / 1.5
```

**Weighted Aggregation**:
```python
overall_score = (
    0.25 * calibration_score +
    0.30 * error_detection_score +
    0.25 * boundary_score +
    0.20 * monitoring_score
)
```

Weights reflect relative importance:
- Error detection (30%): Most critical for self-improvement
- Calibration (25%): Important for human-AI collaboration
- Boundaries (25%): Critical for safety
- Monitoring (20%): Important but harder to measure

## Discriminatory Power Analysis

**Expected Score Distribution**:
- Strong metacognition: 0.75-0.85
- Moderate metacognition: 0.55-0.75
- Limited metacognition: 0.35-0.55
- Weak metacognition: <0.35

**No Ceiling Effect**: No model expected to score >0.85, ensuring headroom for improvement.

**No Floor Effect**: Random guessing on components yields ~0.25, ensuring baseline discrimination.

## Robustness Considerations

1. **Answer Verification**: All answers have unambiguous ground truth
2. **Regex Resilience**: Multiple patterns for extracting confidence/answers
3. **Judge Independence**: Minimal use of LLM judges; deterministic verification preferred
4. **Reproducibility**: Fixed dataset, deterministic scoring

## Limitations

1. **English-only**: All questions in English
2. **Text-only**: No multimodal evaluation
3. **Static dataset**: No dynamic question generation
4. **Binary correctness**: No partial credit for partially correct answers

## Future Extensions

1. Multi-language support
2. Multimodal metacognition (image, audio)
3. Dynamic difficulty adjustment
4. Real-time confidence tracking
5. Human baseline comparison
