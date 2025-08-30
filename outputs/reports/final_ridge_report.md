# Data Quality Report

## Final Model Report – Ridge Regression
### Best Parameters
- **alpha**: 1.0
- **k_best**: 12
### Test Set Metrics
- **RMSE**: 0.2184313782232134
- **MAE**: 0.17404454376962314
- **R2**: 0.9442608565242703
### Features
- **Numeric**: ['Age', 'StudyTimeWeekly', 'Absences', 'FamilyCapitalScore', 'EngagementIndex']
- **Categorical**: ['Gender', 'Ethnicity', 'ParentalEducation', 'Tutoring', 'ParentalSupport', 'Extracurricular', 'Sports', 'Music', 'Volunteering']
- **Selected After Fit**: ['StudyTimeWeekly', 'Absences', 'FamilyCapitalScore', 'Tutoring_0', 'Tutoring_1', 'ParentalSupport_0', 'ParentalSupport_3', 'ParentalSupport_4', 'Extracurricular_0', 'Extracurricular_1', 'Music_0', 'Music_1']
- **Notes**: Ridge tuned with 5-fold CV. Generalizes well (R²≈0.944). Linear models outperform non-linear ones in this dataset.
