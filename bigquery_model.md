
#  Model Development in BigQuery SQL (XGBoost)

The maternal health risk prediction model was developed using **BigQuery ML** with **XGBoost (Boosted Tree Classifier)** for accurate classification of maternal health risk levels (Low, Medium, High).

---

##  Step 1: Feature Engineering – Add Derived Columns
I created synthetic features for better model performance and interpretability:

CREATE OR REPLACE TABLE `smart-mum-project.smartmum.smart_mum_data_enhanced` AS
SELECT
  *,
  -- Access to Hospital based on region and ANC visits
  CASE 
    WHEN Region = 'Rural' AND ANC_Visits = 0 THEN 'Low'
    WHEN Region = 'Rural' AND ANC_Visits <= 2 THEN 'Medium'
    ELSE 'High'
  END AS Hospital_Access,

  -- Delivery History based on Gravidity
  CASE
    WHEN Gravidity = 0 THEN 'FirstTime'
    WHEN Gravidity BETWEEN 1 AND 2 THEN 'Moderate'
    ELSE 'Multiple'
  END AS Delivery_History
FROM
  `smart-mum-project.smartmum.smart_mum_data`;
```

---

##  Step 2: Train the XGBoost Model in BigQuery ML

CREATE OR REPLACE MODEL `smart-mum-project.smartmum.smart_mum_data_model_xgb`
OPTIONS (
  model_type = 'BOOSTED_TREE_CLASSIFIER',
  input_label_cols = ['RiskLevel'],
  auto_class_weights = TRUE,
  max_iterations = 50,
  enable_global_explain = TRUE
) AS
SELECT
  Age, SystolicBP, DiastolicBP, BS, BodyTemp,
  HeartRate, ANC_Visits, BMI, Trimester, Gravidity,
  Region, Mother_Education, Hospital_Access, Delivery_History,
  RiskLevel
FROM
  `smart-mum-project.smartmum.smart_mum_data_enhanced`;
```

---

##  Step 3: Evaluate the Model


SELECT *
FROM ML.EVALUATE(MODEL `smart-mum-project.smartmum.smart_mum_data_model_xgb`);
```

Key metrics like `accuracy`, `precision`, `recall`, and `log_loss` were assessed for model performance.

---

##  Step 4: Generate Predictions

SELECT
  *, predicted_RiskLevel, predicted_RiskLevel_probs
FROM
  ML.PREDICT(MODEL `smart-mum-project.smartmum.smart_mum_data_model_xgb`,
    (SELECT * FROM `smart-mum-project.smartmum.smart_mum_data_enhanced`)
  );
```

---

##  Step 5: Extract Probabilities for Each Risk Class


SELECT
  p.Age, p.SystolicBP, p.DiastolicBP, p.BS, p.BodyTemp,
  p.HeartRate, p.ANC_Visits, p.BMI, p.Trimester, p.Gravidity,
  p.Region, p.Mother_Education, p.Hospital_Access, p.Delivery_History,
  p.predicted_RiskLevel,

  (SELECT prob FROM UNNEST(p.predicted_RiskLevel_probs) WHERE label = 'Low') AS prob_Low,
  (SELECT prob FROM UNNEST(p.predicted_RiskLevel_probs) WHERE label = 'Medium') AS prob_Medium,
  (SELECT prob FROM UNNEST(p.predicted_RiskLevel_probs) WHERE label = 'High') AS prob_High

FROM
  ML.PREDICT(MODEL `smart-mum-project.smartmum.smart_mum_data_model_xgb`,
    (SELECT * FROM `smart-mum-project.smartmum.smart_mum_data_enhanced`)
  ) p;
```
