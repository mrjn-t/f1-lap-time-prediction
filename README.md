# 🏎️ F1 Lap Time Prediction (FastF1)

This mini project explores whether simple race context features can be used to predict Formula 1 lap times with basic machine learning models.

The goal is not perfect accuracy, but to practice building a complete ML workflow and understand how model evaluation behaves on real engineering-style data.

---

## Project overview

**Task:** predict lap time (in seconds) using a small, interpretable feature set:

- Driver  
- Tyre compound  
- Tyre age (TyreLife)  
- Lap number  
- TyreLife² (to capture non-linear degradation)

**Workflow:** data loading → preprocessing → feature engineering → baseline → model training → validation → error analysis

I kept the pipeline simple and readable rather than using complex models.

---

## Data

Data is collected using the **FastF1** Python package.

**Races used (2023):**
- Bahrain Grand Prix
- Saudi Arabian Grand Prix
- Australian Grand Prix

After filtering pit laps and missing values, the dataset contains ~2500 laps across 3 race events.

---

## Validation approach

I compared two evaluation methods:

### 1) Random split
Laps are split randomly into train and test sets.  
This often produces better metrics, but is less realistic because laps from the same race can appear in both sets.

### 2) Group-based split (by race) — main evaluation
The model is trained on two races and tested on a completely unseen race.  
This is closer to real-world generalization across events/tracks.

---

## Models

- **Baseline:** mean predictor (DummyRegressor)
- **Ridge Regression:** linear model (simple and stable)
- **Random Forest:** non-linear model (can overfit with few races)

---

## Results (group-based split)

| Model | MAE (s) |
|------|---------:|
| Baseline (mean) | ~14.0 |
| Ridge Regression | ~12.8 |
| Random Forest | ~13.0 |

Both Ridge and Random Forest improve over the baseline, showing that tyre and race context features contain useful predictive signal.

Interestingly, Ridge performed slightly better than Random Forest on the unseen race, suggesting that simpler models may generalize better when the dataset is small.

---

## Error analysis

To better understand model behavior, I analyzed:

- prediction error by **driver**
- prediction error by **tyre compound**
- **true vs predicted** scatter plot
- feature importance (Random Forest)

This helps identify where the model struggles and which inputs matter most.

---

## Limitations

Lap time depends on many factors not included in this small feature set, such as:

- traffic and overtakes
- fuel load and strategy
- safety car periods
- race incidents
- track-specific conditions

With only three races, results can vary depending on which event is used as the test set.

The main insight is not the absolute error value, but the importance of realistic validation and careful interpretation.

---

## What I learned

This project helped me practice:

- building a complete ML pipeline from raw data to evaluation
- using proper validation to avoid overly optimistic results
- comparing linear vs non-linear models
- interpreting model performance beyond metrics (error analysis + feature importance)

---

## Tech stack

- Python
- FastF1
- Pandas
- Scikit-learn
- Matplotlib

---

## Installation

pip install -r requirements.txt

---

## Author

Personal machine learning project created as part of my preparation for MSc studies in Data Science / AI.
