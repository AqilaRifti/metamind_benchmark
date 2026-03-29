# MetaMind: Comprehensive Metacognitive Evaluation Benchmark

[![Kaggle Benchmark](https://img.shields.io/badge/Kaggle-Benchmark-blue)](https://www.kaggle.com)
[![Track: Metacognition](https://img.shields.io/badge/Track-Metacognition-purple)](https://www.kaggle.com/competitions/kaggle-measuring-agi)
[![License: CC0](https://img.shields.io/badge/License-CC0-green)](https://creativecommons.org/publicdomain/zero/1.0/)

**MetaMind** is a comprehensive benchmark for evaluating metacognitive abilities in AI systems—measuring how well models "know what they know."

## 🎯 Overview

Current AI benchmarks measure *what* models know. MetaMind measures *how well they know it.* This distinction is critical for:
- **Safe deployment**: Knowing when to defer to humans
- **Error detection**: Recognizing and correcting mistakes
- **Hallucination prevention**: Refusing to answer unanswerable questions
- **Self-improvement**: Monitoring and adjusting reasoning

## 📊 The Four Dimensions of Metacognition

### 1. Confidence Calibration
Measures alignment between stated confidence (0-100%) and actual accuracy.

**Key Metric**: Expected Calibration Error (ECE)

### 2. Error Detection & Self-Correction
Tests whether models recognize and fix their own mistakes through a three-phase protocol.

**Key Metrics**: Error detection rate, correction success rate

### 3. Knowledge Boundary Detection
Evaluates appropriate refusal on unanswerable, impossible, and ambiguous questions.

**Key Metrics**: Hallucination rate, over-refusal rate

### 4. Strategic Self-Monitoring
Assesses step-by-step reasoning monitoring during complex problems.

**Key Metrics**: Monitoring-accuracy correlation, explicit monitoring rate

## 🚀 Quick Start

```python
import kaggle_benchmarks as kbench
from benchmark import metamind_benchmark

# Run the complete benchmark
llm = kbench.llm
results = metamind_benchmark(llm)

print(f"Overall Metacognition Score: {results['overall_metacognition_score']:.4f}")
```

## 📁 Repository Structure

```
metamind_benchmark/
├── benchmark.py                      # Main benchmark aggregation
├── task_confidence_calibration.py    # Task 1: Confidence calibration
├── task_error_detection.py           # Task 2: Error detection
├── task_knowledge_boundaries.py      # Task 3: Knowledge boundaries
├── task_self_monitoring.py           # Task 4: Self-monitoring
├── metamind_demo.ipynb              # Demo notebook with analysis
├── WRITEUP.md                        # Kaggle submission writeup
├── cover_image.png                   # Cover image for media gallery
├── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

## 📈 Sample Output

```json
{
  "overall_metacognition_score": 0.6543,
  "component_scores": {
    "confidence_calibration": 0.8234,
    "error_detection_self_correction": 0.5678,
    "knowledge_boundary_awareness": 0.7123,
    "strategic_self_monitoring": 0.5890
  },
  "interpretation": "MODERATE METACOGNITION: The model shows reasonable self-awareness with room for improvement."
}
```

## 🔬 Dataset Statistics

| Task | Questions | Categories | Difficulty Levels |
|------|-----------|------------|-------------------|
| Confidence Calibration | 18 | 6 domains | 4 levels |
| Error Detection | 10 | 6 error types | 3 levels |
| Knowledge Boundaries | 22 | 5 categories | Mixed |
| Self-Monitoring | 8 | 6 problem types | 2 levels |

**Total**: 58 unique evaluation items

## 💡 Key Innovations

1. **Multi-Dimensional Assessment**: Four distinct metacognitive faculties evaluated independently
2. **Calibration Metrics**: Beyond accuracy—measures confidence-accuracy alignment
3. **Error Taxonomy**: Classifies failure modes (overconfident, underconfident, blind to errors)
4. **Cognitive Profile**: Reveals strengths/weaknesses invisible to single-score benchmarks

## 🎓 Citation

If you use MetaMind in your research, please cite:

```bibtex
@misc{metamind2026,
  title={MetaMind: Comprehensive Metacognitive Evaluation Benchmark},
  author={Hamizan Rifti, Aqila},
  year={2026},
  howpublished={Kaggle Benchmarks},
  note={Measuring Progress Toward AGI - Cognitive Abilities Hackathon}
}
```

## 📚 References

1. Fleming, S. M., & Lau, H. C. (2014). How to measure metacognition. *Frontiers in Human Neuroscience*, 8, 443.
2. Plomecka, M., Yan, Y., Kang, N., et al. (2026). Measuring Progress Toward AGI: A Cognitive Taxonomy. *Google DeepMind*.
3. Steyvers, M., et al. (2025). Beyond Accuracy: How AI Metacognitive Sensitivity improves AI-assisted Decision Making.

## 📄 License

This benchmark is released under CC0 (Public Domain) as required by the competition rules.

## 🤝 Acknowledgments

Created for the Kaggle "Measuring Progress Toward AGI - Cognitive Abilities" hackathon sponsored by Google DeepMind.
