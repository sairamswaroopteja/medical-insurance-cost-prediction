# Medical Insurance Cost Prediction

## Problem Statement
Predict individual medical insurance charges using demographic and lifestyle features.  
Dataset: 1,338 records (1 duplicate removed), 6 features → 1 target.

## Project Structure
```
insurance_project/
|
├── src/
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── models/   ← saved All models used + best_model 
|── plots/                 ← visualisations
├── insurance.csv          ← raw data
├── run_pipeline.py        ← end-to-end pipeline
├── requirements.txt       
|── results.csv          
```
## Installation

1. Create Virtual Environment
python -m venv venv
2. Activate Environment

Windows:

venv\Scripts\activate

Mac/Linux:

source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

How to Run
 Step 1: Train Models & Generate Results
    python run_pipeline.py

    This will:

    Preprocess data
    Perform feature engineering
    Train multiple models
    Tune hyperparameters
    Evaluate models
    Save all models in models/
    Save best model as best_model.pkl
    Generate results.csv and plots

 Step 2: Make Predictions
    Run:

    python src/predict.py

    Then enter inputs like:

    Enter age: 30
    Enter sex (male/female): male
    Enter BMI: 27.5
    Enter number of children: 1
    Smoker? (yes/no): no
    Enter region: northwest

Output:
  Predicted Insurance Cost: $XXXX

## Dataset
| Feature    | Type        | Description                        |
|------------|-------------|------------------------------------|
| age        | int         | Age of the primary beneficiary      |
| sex        | categorical | Male / Female                       |
| bmi        | float       | Body Mass Index                     |
| children   | int         | Number of dependents                |
| smoker     | categorical | Yes / No                            |
| region     | categorical | US region (NE, NW, SE, SW)          |
| **charges**| float       | **Target — medical cost billed**    |

## Steps Followed
1. Data loading & cleaning (remove duplicate)
2. EDA — distributions, bivariate & multivariate analysis, correlation heatmap
3. Preprocessing — label/one-hot encoding, log(charges) target transformation
4. Feature engineering — age², bmi_category, has_children, smoker×bmi interaction
5. Train/test split (80/20, random_state=42)
6. Model training & hyperparameter tuning
7. Evaluation — RMSE, MAE, R²
8. SHAP analysis for interpretability
9. Risk segmentation (Low / Medium / High)

## Models & Results

| Model               | RMSE ($) | MAE ($) | R²    |
|---------------------|----------|----------|--------|
| **Random Forest**   | **4,263**| **1,996**| **0.901** |
| AdaBoost            | 4,487    | 2,460    | 0.891 |
| KNN                 | 4,596    | 2,131    | 0.885 |
| Ridge Regression    | 7,991    | 4,155    | 0.653 |
| Linear Regression   | 8,036    | 4,198    | 0.649 |
| Lasso Regression    | 8,320    | 4,171    | 0.623 |

**Best model: Random Forest Regressor (R² = 0.901)**

## Key Insights
> **"Smoking is the most influential factor in determining medical insurance cost,  
> and its interaction with BMI significantly increases predicted expenses."**

- Smokers pay on average **3–4× more** than non-smokers.
- The `smoker_bmi` interaction feature is the top engineered predictor.
- Age has a nonlinear effect — captured well by `age²`.
- Tree-based models (RF, XGBoost) far outperform linear models, indicating  
  non-linear relationships in the data.



## Requirements
```
pandas numpy matplotlib seaborn scikit-learn shap
```
