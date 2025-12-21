# Deep Learning Group Project — Assignment Specification

**Group size:** 3 students per group  
**Due date:** *By the end of 17 January 2026*

---

## 1. Overview

You are provided with a **custom dataset of 3,000 samples**.  
Your task is to design, train, justify, and evaluate a **deep learning model** using **Keras / TensorFlow**.

This is a **group project**.  
You will be assessed on correctness, reasoning, experimentation quality, clarity, and reproducibility.

---

## 2. Files You Will Receive

### Dataset
- `dataset_dev_3000.npz`

This file contains two NumPy arrays:

| Key | Shape | Type | Description |
|----|------|------|------------|
| `X` | `(3000, 32, 32)` | `float32` | input data |
| `y` | `(3000, 3)` | `float32` | target values |

The ordering of samples is consistent:  
**`X[i]` corresponds to `y[i]` for all `i`.**

There is **no predefined train / validation split**.  
You must create your own split(s).

---

## 3. Target Format and Ranges

Each row of `y` corresponds to **three independent targets**:

| Component | Meaning | Range | Type |
|---------|--------|------|------|
| `y[:,0]` | Target A | integers in `{0,1,2,3,4,5,6,7,8,9}` | classification |
| `y[:,1]` | Target B | integers in `{0,1,…,31}` | classification |
| `y[:,2]` | Target C | real value in `[0,1]` | regression |

The three targets are **independent**.  
You should **not assume any ordering or hierarchy** among them.

You design a **single model predicting all components simultaneously**.

---

## 4. Task Objective

Your goal is to build a **deep learning model** that predicts all three target components from the input data.

You are encouraged to experiment with:
- different architectures (CNNs, shared vs. multi-head designs),
- preprocessing or normalization,
- loss functions and loss weighting,
- regularization and optimization strategies.

---

## 5. What You Must Submit

Each group must submit:

- `submission_groupId.ipynb`
- `model_groupId.h5`

Your notebook **must support BOTH options**:

| Option | Description |
|------|-------------|
| **A. Load Model** | Loads the saved model and evaluates it |
| **B. Train Model** | Trains the model from scratch |

Both options must result in the same callable:

```python
def predict_fn(X32x32: np.ndarray) -> np.ndarray:
    ...
```

- Input: `(N, 32, 32)`
- Output: `(N, 3)` with the same format as `y`

---

## 6. Notebook Requirements

Your notebook must clearly include the following sections:

- **Introduction (markdown)** — problem understanding and goals  
- **Dataset inspection** — shapes, value ranges, observations  
- **Train/validation split strategy** — justification  
- **Model architecture reasoning** — why this design  
- **Theory & techniques** — loss functions, activations, optimization  
- **Experiments** — variations, learning curves, comparisons  
- **Option A — Load model**
- **Option B — Train model**
- **Evaluation & discussion**
- **Reflection** — limitations and possible improvements  

Markdown explanations are **mandatory**, not optional.

---

## 7. Allowed Libraries and Environment

- Framework: **Keras / TensorFlow only**
- Allowed: `numpy`, `matplotlib`, `pandas`, `scikit-learn`, and similar standard scientific Python packages
- Internet access is allowed
- Lightweight `pip install` inside the notebook is acceptable if clearly stated
- Do **not** download external datasets during execution

---

## 8. Academic Honesty Statement

Include the following at the top of your notebook:

> *I declare that this submission is my own work, and that I did not use any pretrained model or code that I did not explicitly cite.*

---

## 9. Grading Criteria

| Criterion | Weight |
|--------|-------|
| Code quality & reproducibility | 10% |
| Depth of reasoning | 10% |
| Correctness & model performance | 50% |
| Documentation & experimental evidence | 30% |

There is **no strict performance threshold**, but poor results without justification will affect the mark.

---

## 10. Submission Notes

- Submit exactly the required files
- Ensure your notebook runs end-to-end
- Clearly label all figures and experiments
