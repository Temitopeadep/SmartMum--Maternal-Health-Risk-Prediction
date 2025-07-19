# SmartMum: Maternal Health Risk Prediction Through AI (XGBoost Model)
![Model Accuracy](https://img.shields.io/badge/Accuracy-89.6%25-brightgreen)
![AUC Score](https://img.shields.io/badge/AUC-0.95-blue)

## Overview
This project was developed for the 3MTT knowledge showcase (July Edition), it's an AI-powered maternal health risk prediction tool built using BigQuery ML (XGBoost) and integrated with Power BI and Power Apps. It enables early identification of pregnancy-related complications by analyzing clinical and demographic data.

## Project Goal
To reduce maternal health risks through early prediction and actionable insights for healthcare providers and expectant mothers.

## Project Components
### Executive Summary
1. Model: XGBoost via BigQuery ML
2. Accuracy: 89.6%
3. 	Recall: 87.3%
4. Precision: 88.1%
5. AUC Score: 0.95
6. Dataset Size: 1,000 Records
###  Key Features
- Uses real health indicators to predict Low / Medium / High risk.
- Power BI dashboard with demographics, clinical variables, and trends.
- Power Apps form for public access without technical knowledge.

### Data Source
Dataset: [Maternal Health Risk | Kaggle](...)
- 1,000 anonymized records  
- Core columns: Age, BP, BS, BodyTemp, HeartRate, BMI, ANC Visits, Trimester, Region, Education  
- Enhanced columns: Delivery History, Hospital Access

  ### Dashboard
![Screenshot](https://github.com/Temitopeadep/SmartMum--Maternal-Health-Risk-Prediction/blob/main/dashboard1.png)
![Screenshot](https://github.com/Temitopeadep/SmartMum--Maternal-Health-Risk-Prediction/blob/main/dashboard2.png)

### PowerApps Interface
![Screenshot](https://github.com/Temitopeadep/SmartMum--Maternal-Health-Risk-Prediction/blob/main/powerapp%20interface.png)

### Findings
1. Model Performance: The XGBoost model achieved strong metrics — Accuracy: 89.6%, Recall: 87.3%, Precision: 88.1%, and AUC: 0.95, confirming its reliability in predicting maternal health risks.
2. Risk Distribution: Most cases fall into the Medium Risk category, suggesting a window for preventive care. High-risk cases, though fewer, remain clinically significant.
3. ANC Visits: Low or zero antenatal care visits strongly correlate with higher maternal health risk levels. Regular ANC visits are protective.
4. Clinical Indicators: Elevated BMI, Heart Rate, Blood Pressure, and Body Temperature are key predictors of high-risk classifications.
5. Demographics & Access: Both urban and rural regions exhibit similar risk trends. However, women with limited access to facilities or education face compounded risks.

### Recommendation
* Promote early and frequent ANC visits to shift risk levels from high to medium/low.
* Integrate AI predictions into frontline healthcare tools for early intervention.
* Train health workers to identify and act on clinical indicators such as elevated BP, BMI, and heart rate.
* Expand outreach in low-access areas, especially for under-educated and rural populations.
* Scale the SmartMum app across regions to support real-time maternal risk assessment.

  ###  Tools Used
- Google BigQuery ML – for training the XGBoost classification model to predict maternal risk levels.
- Microsoft Power BI – for data visualization and dashboard development.
- Microsoft Power Apps – to build a no-code, user-friendly maternal risk prediction form.
- GitHub – for hosting project code and documentation
- ChatGPT (by OpenAI) – for technical assistance

### Links
- [BigQuery](https://console.cloud.google.com/bigquery?sq=954458391922:ad90499052664c02a6328e99d623c19f)
- [Power App](https://apps.powerapps.com/play/e/default-18f15a3f-ea52-4261-b725-0b0f0d1bab03/a/3f4e200a-5ef6-4243-a44a-16aac7ab0a22?tenantId=18f15a3f-ea52-4261-b725-0b0f0d1bab03&hint=19c11dff-6196-48df-8786-c149e04f3516&source=sharebutton&sourcetime=1752784196040)

  ##  Author
**Temitope Adepoju**  
Driven by a mission to make maternal health safer through data.


