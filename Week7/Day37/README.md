# Week 07 · Day 37 — TF-IDF from Scratch

**IIT Gandhinagar · NLP Foundations · Cohort 1**

---

## What this notebook covers

| Task | Description |
|------|-------------|
| Q1(a) | Full TF-IDF matrix for 10,000 reviews using sparse representation — no sklearn |
| Q1(b) | Top-5 review retrieval for query: *"wireless earbuds battery life poor"* using cosine similarity |
| Q1(c) | Comparison against sklearn's `TfidfVectorizer` — average L2 difference reported |
| Q1(d) | Word with highest average TF-IDF score in the Electronics category with explanation |
| Q2(a) | Manual step-by-step computation of TF, IDF, TF-IDF for `fabric` in Doc_42 |
| Q2(b) | IDF contrast: why `IDF('the') ≈ 0` while `IDF('embroidery')` is high |
| Q2(c) | 3-sentence rebuttal to "just use word frequency, TF-IDF is overcomplicated" |
| Bonus | BM25 weighting (k1=1.5, b=0.75) — results compared against TF-IDF cosine ranking |

---

## Files in this folder

| File | Description |
|------|-------------|
| `Week07_Monday_TF_IDF.ipynb` | Main notebook with all tasks implemented |
| `Week07_Monday_Report.docx` | Written report summarising findings |
| `top5_query_results.png` | Top-5 retrieved reviews for the electronics query |
| `top_electronics_tfidf.png` | Bar chart of highest TF-IDF words in Electronics category |
| `l2_difference_distribution.png` | Distribution of L2 differences vs sklearn |
| `tfidf_vs_bm25_comparison.png` | Side-by-side ranking comparison: TF-IDF vs BM25 |

---

## Installation

```bash
pip install numpy pandas scipy scikit-learn jupyter
```

---

## How to run

1. Place `shopsense_reviews.csv` in the same folder as the notebook
2. Launch Jupyter:
   ```bash
   jupyter notebook Week07_Monday_TF_IDF.ipynb
   ```
3. Run all cells top to bottom (`Kernel → Restart & Run All`)

---

## Dataset

ShopSense E-Commerce Reviews — 10,000 rows across Electronics, Clothing, Food, Home, Beauty, Books categories.
> Dataset is not committed to this repo. Add it locally before running.
