# Fairness-FR
## Demographic Fairness Evaluation of Face Recognition Systems

Fairness-FR is an end-to-end research and evaluation framework for studying the
performance and demographic fairness of face recognition systems.

The project does not evaluate a face recognition model only by asking
"How accurate is it?"

Instead, it evaluates two important dimensions together:

1. **Recognition performance** — how reliably the system accepts genuine users
   and rejects impostors.
2. **Demographic fairness** — whether the recognition behaviour is consistent
   across demographic groups.

The framework implements a complete evaluation pipeline beginning with dataset
preparation and pair generation and continuing through embedding extraction,
similarity scoring, biometric performance evaluation, demographic fairness
analysis, and cross-model comparison.

An interactive Streamlit dashboard is also provided for exploring the generated
results.

---

# 1. Motivation

Face recognition systems are increasingly used in security, authentication,
access control, surveillance, and other applications where incorrect decisions
can have significant consequences.

A model can achieve high overall accuracy while still behaving differently for
different demographic groups.

For example, suppose two models have similar overall accuracy:

- Model A has relatively consistent error rates across demographic groups.
- Model B has excellent performance for some groups but considerably higher
  false rejection or false acceptance rates for another group.

Looking only at overall accuracy would hide this difference.

Therefore, Fairness-FR evaluates both:

```text
Overall Recognition Performance
              +
Biometric Error Rates
              +
Demographic Group Behaviour
              +
Cross-Model Comparison
