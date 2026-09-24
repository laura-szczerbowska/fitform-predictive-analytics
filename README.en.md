<div align="right">
  <a href="./README.md">Polski</a> | <strong>English</strong>
</div>

# FitForm: Time-Series Weight Dynamics & Predictive Analytics Engine

Predictive and analytical module for the FitForm platform, converting daily activity and nutrition logs into reliable body weight change forecasts. The project addresses the challenges of irregular weigh-ins and daily water weight fluctuations, providing the application's recommendation engine with actionable decision metrics grounded in a 10-model regression benchmark and feature importance analysis.
> *Project developed as part of an academic team initiative under the Project-Based Learning (PjBL) framework.*

<br>

## 1. Business Problem

In fitness applications, standard body weight forecasting based solely on static caloric deficits (e.g., Wishnofsky's rule) fails when faced with real-world users:
* **Short-term water fluctuations:** Body weight can fluctuate by 1-2 kg from day to day due to diet composition or post-workout recovery, failing to reflect true adipose tissue loss or gain.
* **Irregular logging behavior:** Users weigh themselves intermittently (every 2-4 days), while dietary intakes and step counts are logged daily.
* **Input errors and noise:** Extreme, erroneous manual entries distort classical learning algorithms.

**Solution:**  
This project implements an end-to-end predictive pipeline (`train_pipeline.py`) capable of handling irregular intervals, smoothing noise via rolling statistics, and forecasting the underlying weight trajectory. The generated forecast feeds directly into the application engine, which visualizes weight trajectory predictions over a selected timeframe, adjusts the user's physique target (**cutting / bulking / maintenance**), and calibrates the recommended training routine.

<br>

## 2. Pipeline Workflow

```text
PostgreSQL Data Extraction
       ↓
Data Cleaning & Anomaly Filtering (5th-95th Percentiles)
       ↓
Feature Engineering (Rolling Windows, Per-kg Ratios)
       ↓
Target Normalization Across Time Intervals (kg/day)
       ↓
Data Partitioning & GroupKFold Validation (Leakage-Free)
       ↓
Training & Benchmarking of 10 Regression Models + Ensemble
       ↓
Scenario Simulation & Stability Testing
       ↓
Model Explainability Analysis (PFI, SHAP, Timeline Fit)
       ↓
Production Model Serialization (.pkl) for Backend API
```
<br>


## 3. Data & Dataset Preparation

Data used for analysis and model training originates from a relational PostgreSQL database (populated via ETL pipelines) and comprises daily entries of macronutrients, caloric balance, step counts, and structured workout sessions.


#### The dataset combines two primary sources:

* **Empirical data (5 real users):** A 3-month continuous history of daily entries from live individuals, capturing natural habits, irregular logging, measurement errors, and authentic biological weight trajectories.

* **Synthetic data (Faker):** A structured cohort generated under a custom schema modeling diverse behavioral profiles - from textbook adherence (high consistency, stable deficit/surplus) to edge cases (irregular weigh-ins, episodic caloric spikes, extreme activity levels, illness).
  

<br>


### 1) Data Cleaning
* **Imputation of missing weigh-ins:** Irregular weigh-ins were handled per user using `ffill` and `bfill` methods, ensuring baseline continuity for subsequent calculations.
* **Removal of incomplete entries:** The final logged observation for each profile was excluded from the training set, as it lacked a subsequent measurement required to calculate the daily weight delta.
* **Error and outlier filtering:** Zero-value entries that would cause division errors (e.g., missing energy expenditure) were converted to nulls instead of throwing program errors. Additionally, extreme daily weight drops and spikes below the 5th and above the 95th percentiles were removed to filter out obvious manual data entry mistakes.


<br>


### 2) Feature Engineering
* **Core energy balance:** The primary differential feature defining daily surplus or deficit:
  

`bilans_kcal` = `spozyte_kcal` − `spalone_kcal`


* **Interval normalization:** The target variable (y) represents the normalized daily rate of weight change, accounting for variable time elapsed between consecutive weigh-ins:


$$\text{daily weight delta [kg/day]} = \frac{\Delta \text{weight [kg]}}{\text{days between weigh-ins}}$$


* **Fluctuation smoothing:** 7-day rolling averages were calculated for caloric balance (`bilans_kcal_ma7`), allowing the model to capture the general direction of change rather than transient single-day noise.
* **Weight-normalized metrics:** Beyond raw calories and grams of protein, body-weight-scaled indicators were added (`kcal_na_kg`, `bialko_na_kg`) to enable comparison across individuals with different body weights.
* **Behavioral weekend flag:** A weekend indicator (Saturday-Sunday) was introduced to isolate cyclic dietary deviations and activity shifts during rest days.

<br>




### 3) Input Feature Space (11 Features)
The models received 11 variables spanning three core operational domains:
* **Energy & balance:** daily caloric balance, 7-day rolling average caloric balance, calories consumed, calories burned, calories per kg of body weight.
* **Nutrition:** protein intake (g) and protein per kg of body weight.
* **Activity & lifestyle:** resistance training (0/1), cardio duration (min), step count, and weekend flag (0/1).


<br>


### 4) Validation Methodology (Zero Data Leakage)
Random splitting (`train_test_split`) on user time-series data can lead to data leakage, as the model memorizes the specific characteristics of individual users. A two-tier safeguard was applied:
1. **Profile Split (80/20) and Empirical Data:** 80% of user profiles were allocated to the training set, while 20% formed an entirely isolated test set. Real user profiles (`user_id 1-5`) were ensured to be present in both splits, preventing the model from overfitting solely to synthetic patterns.
2. **GroupKFold Cross-Validation:** 3-fold cross-validation grouped strictly by `user_id`. Models were evaluated exclusively on complete user profiles they had not seen during training.

<br>


## 4. Evaluation & Model Benchmarks

### Comparison of 10 Regression Models
Ten diverse machine learning algorithms were benchmarked - ranging from linear models, through nearest-neighbor (KNN) and support vector machines (SVM), to boosting methods and a dedicated Ensemble model (`VotingRegressor`).


<br>


| Model | MAE mean CV [kg] | RMSE mean CV [kg] | R² mean CV | Test MAE [kg] | Test RMSE [kg] | Test R² |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** | **0.0026** | **0.0049** | **0.9768** | **0.0022** | **0.0033** | **0.9899** |
| **Ensemble (Voting)** | **0.0026** | **0.0046** | **0.9796** | **0.0021** | **0.0033** | **0.9900** |
| **Random Forest** | 0.0024 | 0.0044 | 0.9818 | 0.0021 | 0.0033 | 0.9898 |
| **XGBoost** | 0.0026 | 0.0050 | 0.9756 | 0.0021 | 0.0033 | 0.9900 |
| **Decision Tree** | 0.0026 | 0.0053 | 0.9721 | 0.0020 | 0.0033 | 0.9896 |
| **CatBoost** | 0.0027 | 0.0046 | 0.9804 | 0.0023 | 0.0034 | 0.9892 |
| **KNN** | 0.0035 | 0.0062 | 0.9617 | 0.0030 | 0.0050 | 0.9769 |
| **Linear Regression** | 0.0112 | 0.0166 | 0.7458 | 0.0114 | 0.0156 | 0.7697 |
| **Ridge Regression** | 0.0112 | 0.0166 | 0.7459 | 0.0114 | 0.0156 | 0.7699 |
| **SVM (RBF)** | 0.0242 | 0.0283 | 0.2790 | 0.0227 | 0.0270 | 0.3103 |
| **Elastic Net** | 0.0273 | 0.0341 | -0.0450 | 0.0252 | 0.0326 | -0.0041 |

*[Full metric export: static/models_comparison.csv](static/models_comparison.csv)*


<br>


> **Methodological Note on Metrics ($R^2 ≈0.99$, $\text{MAE} ≈ 0.002\text{ kg}$):**  
> The exceptionally high performance of the models stems from the dominant share of synthetic data (95%), generated using the Faker library based on energy balance rules. Empirical data accounted for 5% of the total volume, with careful representation ensured in the isolated test set. On a fully empirical dataset, these metrics will naturally be lower. In its current form, the model has effectively learned the baseline physiological relationships, serving as a stable engine for scenario simulations.

<br>

### Key Table Insights

* **Superiority of Gradient Boosting Models:** **LightGBM** and the **Ensemble** accurately capture non-linear interactions between workout intensity, step count, and caloric deficit ($R^2 ≈ 0.99$).
* **Limitations of Linear Models:** Standard Linear Regression and Ridge achieved $R^2 ≈ 0.77$. While reflecting the overall trend, they struggle with daily fluctuations and non-linear metabolic thresholds.
* **Underperformance of SVM and Elastic Net:** These models smoothed the data excessively ($R^2 <= 0.31$). Consequently, they failed to respond to day-to-day dietary and activity variations, making their predictions ineffective in the simulator.
* **Deployment Decision (Why LightGBM?):**  
  Although the performance differences were marginal, LightGBM was selected for its **simplicity and operational efficiency**: unlike the Ensemble, it does not require running three distinct models simultaneously, and it is lighter and more maintainable than XGBoost without compromising precision.

<br>

## Scenario Simulation

A step-by-step simulation function (`symulacja_wagi`) was implemented to test model stability over a 30-day forecast horizon under a defined scenario (e.g., initial weight of 85 kg, caloric deficit, 14,000 steps, resistance training):
* **Error Accumulation Verification:** Evaluated whether day-by-day iterative forecasting introduces unrealistic metabolic drifts.
* **Business Value:** This mechanism serves as the foundation for the simulation module in the FitForm app, enabling users to preview projected physique outcomes before starting a training plan.

<br>


## 5. Model Explainability & Feature Impact Visualization


#### PFI (Permutation Feature Importance)

Direct degradation in model performance (increase in MAE error) after randomly permuting the values of a given feature:


![PFI LightGBM](static/pfi_lightgbm.png)


> **Plot Translation Guide:**
> * **X-axis (`Wzrost błędu dobowej zmiany wagi (MAE [kg])`):** Increase in daily weight delta error (MAE [kg])
> * **Y-axis Feature Translations:**
>   * *Średni bilans kcal (7 dni)*: 7-day average caloric balance
>   * *Dobowy bilans kaloryczny*: Daily caloric balance
>   * *Liczba kroków*: Step count
>   * *Spalone kalorie (kcal)* / *Spożyte kalorie (kcal)*: Calories burned (kcal) / Calories consumed (kcal)
>   * *Białko / Kalorie / kg masy ciała*: Protein / Calories per kg of body weight
>   * *Dzień weekendowy* / *Trening siłowy* / *Czas cardio (min)*: Weekend day / Resistance training / Cardio duration (min)
>   * *Spożycie białka (g)*: Protein intake (g)


>**Conclusion:**
>The current caloric balance and the 7-day rolling average balance drive over 80% of prediction stability.


<br>


#### SHAP (Shapley Additive Explanations)

* **Global Feature Impact (Summary Plot):**  
  Shapley value analysis allows identifying the hierarchy of the model's decision factors:

<p align="center">
  <img src="static/shap_summary.png" alt="Feature Impact on Daily Weight Change" width="700">
</p>

> **Plot Translation Guide:**
> * **Title (`Wpływ zmiennych na dobową zmianę wagi`):** Feature impact on daily weight change.
> * **X-axis (`SHAP value (impact on model output)`):** Impact on model output (daily weight change in kg/day).
> * **Color Bar (`Feature value`):** Red = High feature value, Blue = Low feature value.


> **Key Analytical Takeaways (Summary Plot):**
> * **Dominance of Caloric Balance:** The 7-day rolling average balance and current daily balance exert the strongest influence on weight loss or gain. Caloric surplus (red points on the right) significantly increases the weight forecast.
> * **Role of Physical Activity:** Step count and cardio duration consistently stimulate weight loss (blue values and shift to the left), mitigating daily energy surpluses.

<br>

* **Local Daily Decomposition (Waterfall Plot):**  
  Deconstruction of an individual prediction for a selected day - explaining which daily behaviors drove the weight decrease or increase relative to the baseline value:

<p align="center">
  <img src="static/shap_waterfall.png" alt="SHAP Waterfall" width="700">
</p>

> **Plot Translation Guide:**
> * **Title (`Lokalna interpretacja pojedynczej predykcji`):** Local interpretation of a single prediction.
> * **Baseline:** $E[f(X)] = -0.007$ kg/day (expected average model prediction).
> * **Final Output:** $f(x) = -0.067$ kg/day (predicted weight change for this day).


> **Single Prediction Interpretation (Waterfall Plot):**
> * **Key Impact of Deficit:** Confirming the global findings, on this analyzed day, the negative 7-day rolling balance and current daily deficit were responsible for lowering the forecast by nearly 0.05 kg/day.
> * **Product Value for FitForm:** This decomposition allows the application to deliver clear daily dashboard feedback: "Your projected weight loss is predominantly driven by your sustained 7-day deficit, rather than today's workout alone."

<br>

#### Timeline Fit

Verification of daily model predictions against ground truth time-series data for the profile displaying the highest change dynamics:
<br>

![Model Predictions vs Ground Truth on Timeline](static/models_timeline_comparison.png)

> **Plot Translation Guide:**
> * **Title (`Analiza porównawcza modeli`):** Comparative model analysis.
> * **Axes:** X-axis = `Dni` (Days, 0-40), Y-axis = `Masa ciała (kg)` (Body weight in kg).
> * **Legend:** `WARTOŚĆ RZECZYWISTA` = Ground truth actual value (black line with points).


> **Plot Commentary:** The chart illustrates a 40-day iterative forward simulation. While tree-based models (LightGBM, XGBoost) accurately tracked the target metabolic trend without drift, heavily regularized linear models (Elastic Net) exhibited underfitting, suppressing daily change dynamics.


<br>


## 6. Tech Stack

* **Core Language & Libraries:** Python, Pandas, NumPy
* **Database & Backend:** PostgreSQL / Supabase, SQLAlchemy
* **Machine Learning:** LightGBM, XGBoost, CatBoost, Scikit-learn
* **Interpretability & Visualization:** SHAP, Matplotlib

  
<br>


## 7. Project Architecture

```text
fitform-predictive-analytics/
├── static/
│   ├── models_comparison.csv             # Exported model evaluation metrics
│   ├── models_timeline_comparison.png    # Time-series fit comparison
│   ├── pfi_lightgbm.png                  # Permutation Feature Importance plot
│   ├── shap_summary.png                  # Global SHAP Summary plot
│   └── shap_waterfall.png                # Single-prediction SHAP decomposition
├── .env.example                          # Database environment variables template
├── .gitignore                            # Temporary and sensitive file exclusions
├── README.md                             # Project documentation (English)
├── README.pl.md                          # Project documentation (Polish)
├── requirements.txt                      # Pipeline dependencies
└── train_pipeline.py                     # End-to-end analytical and training pipeline
```
<br>

## 8. Jak uruchomić projekt

Sklonuj repozytorium:

```bash
git clone https://github.com/laura-szczerbowska/fitform-predictive-analytics.git
```
Install the required packages: 
```bash
pip install -r requirements.txt
```
Set up the environment variables: 
```bash
cp .env.example .env
```
Run the Python pipeline: 
```bash
python train_pipeline.py
```

<br>

## 9. Future Roadmap

Potential module extensions include:

* Integrating external fitness tracker APIs (Apple Health, Google Fit) for automated heart rate and energy expenditure ingestion
* Benchmarking sequence-based architectures (LSTM / GRU) across extended multi-week forecast horizons
* Developing an interactive analytics dashboard for monitoring trajectory projections

<br>

## 10. Key Takeaways

Leveraging **11 daily behavioral indicators**, the **LightGBM** model reliably forecasts underlying body weight trajectories (**$R^2 ≈ 0.99$**, **$MAE ≈ 0.002$ kg/day**), effectively filtering daily stochastic noise caused by fluid shifts and irregular weigh-ins.

Model predictions empower the FitForm decision engine by:

* Automatically tailoring physique milestones (cutting, bulking, maintenance)
* Dynamically calibrating training and caloric recommendations based on empirical user trends
* Facilitating early detection of metabolic adaptation plateaus and declining adherence
* Delivering transparent explanations (XAI) detailing primary drivers behind individual weight variations

This repository showcases a complete Data Science / Analytics lifecycle: from data extraction in PostgreSQL, through time-series preprocessing and domain feature engineering, to leak-free 10-model benchmarking under `GroupKFold` validation and business-level explainability (SHAP, PFI).
