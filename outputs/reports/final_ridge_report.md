# Data Quality Report

## Final Model Report – Ridge Regression
### Best Parameters
- **alpha**: 1.0
- **k_best**: 12
### Test Set Metrics
- **RMSE**: 0.2006550270306913
- **MAE**: 0.1603578961733596
- **R2**: 0.9529640058163702
### Features
- **Numeric**: ['Age', 'StudyTimeWeekly', 'Absences']
- **Categorical**: ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
- **Selected After Fit**: ['Age', 'StudyTimeWeekly', 'Absences', 'Gender_0', 'Gender_1', 'Ethnicity_0', 'Ethnicity_1', 'Ethnicity_2', 'Ethnicity_3', 'ParentalEducation_0', 'ParentalEducation_1', 'ParentalEducation_2', 'ParentalEducation_3', 'ParentalEducation_4', 'Tutoring_0', 'Tutoring_1', 'ParentalSupport_0', 'ParentalSupport_1', 'ParentalSupport_2', 'ParentalSupport_3', 'ParentalSupport_4', 'Extracurricular_0', 'Extracurricular_1', 'Sports_0', 'Sports_1', 'Music_0', 'Music_1', 'Volunteering_0', 'Volunteering_1']
- **Notes**: Ridge tuned with 5-fold CV. Generalizes well (R²≈0.953). Linear models outperform non-linear ones in this dataset.
