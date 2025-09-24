# Beyond Grades – Unveiling the Factors Behind Students’ Success

Impact Certificate: Applied Machine Learning  
Challenge: Regression  
Tomorrow University - Calibration Phase (Sep 2025)
Industry Focus: Social  
Student: Enrico Vaccari 

## 1. Introduction
How can I use data to better understand and improve educational outcomes?  
In this project I apply **regression analysis** to predict student performance scores based on academic, familial and socio-demographic factors. By identifying which features have the strongest influence on performance, I aim to provide actionable insights for schools, teachers and policymakers who seek to foster **equity and sustainability in education** (aligning with SDGs 4 and 10, namely Quality Education and Reduced Inequalities).


## 2. Problem Statement & Hypotheses
- **What to predict**: The goal of this study is to **predict students’ performance scores** using a dataset of open demographic, familial and academic data.
- **Why it matters**: Early detection of at-risk students enables timely support and helps reduce inequality.  
- **How it would support positive change**: This project aligns with Tomorrow University’s mission of using data science for social good, emphasizing responsible use of technology in service of sustainability and educational fairness.

 **Hypotheses**
- Students with **higher parental education levels** tend to achieve better performance scores.  
- Stronger **study habits and consistent attendance** are positively correlated with higher outcomes.  
- **Gender** has limited predictive power once socio-economic and academic variables are considered.  


## 3. Target Variable
- **Performance Score (`GPA`)** → continuous numeric variable.  
- **Regression suitability**: The score is quantitative and continuous, making it an appropriate target for regression models (supervised approach).


## 4. Features
Available features include gender, parental education, study time, attendance, family income, extracurricular activity and GPA/GradeClass.  

- **Likely strong predictors**: absences, parental education, attendance, study habits, family income, access to resources.  
- **Possible exclusions**: features with low variance, excessive missingness or multicollinearity.  
- **Additional note**: GPA is available as a continuous variable while GradeClass is a categorical binned version (Fail–Excellent).  

Below is a list of features included in the dataset, along with short descriptions for each:

- **`Age`**: Student's age in years (typically between 15 and 18)
- **`Gender`**: Binary variable indicating student's gender
- **`Ethnicity`**: Categorical representation of the student’s ethnic background
- **`ParentalEducation`**: Education level of the student's most educated parent (reverse-coded: 0 = highest, 4 = no formal education)
- **`ParentalSupport`**: Indicates whether the student receives emotional/academic support from parents
- **`Tutoring`**: Indicates whether the student receives additional tutoring support
- **`StudyTimeWeekly`**: Number of hours the student studies per week (numeric)
- **`Absences`**: Total number of school days missed
- **`Sports`**: Indicates whether the student participates in sports
- **`Music`**: Indicates whether the student plays a musical instrument
- **`Extracurricular`**: Indicates participation in extracurricular activities (e.g., clubs, competitions)
- **`Volunteering`**: Indicates whether the student engages in volunteering activities
- **`GradeClass`**: categorical binned version of **`GPA`**
- **`GPA`** (target): Final grade point average on a scale from 0 to 4.0

## 5. Dataset Info
- **Dataset name**: *Student Performance Data* (Kaggle, curated by Muhammad Azam)  
- **Why chosen**: Recent, clean, structured and highly relevant to educational and social sustainability research  
- **Format**: CSV, UTF-8 encoded, structured tabular format  
- **Size**: Thousands of student records  
- **License**: Creative Commons Attribution 4.0 (CC BY 4.0)  
- **Geographic coverage**: Multan, Pakistan  
- **Temporal coverage**: March 11 – June 9, 2025 (last update: June 14, 2025)  
- **Source:** [Kaggle Dataset – Muhammad Azam](https://www.kaggle.com/datasets/muhammadazam121/student-performance-data)  
- **Target variable:** `Performance_Score` (continuous)  
- **Additional variables:** GPA (continuous 0–4 scale), GradeClass (categorical bins from Fail to Excellent), gender, parental education, study time, attendance and more  


## 6. Data Quality Assessment

| Dimension             | Assessment  | Notes |
|-----------------------|-------------|-------|
| **Accuracy**          | High        | Cleaned, duplicates removed, consistent labeling |
| **Completeness**      | Good        | Core predictors included but lacks nutrition, sleep or mental health |
| **Consistency**       | High        | Standardized column names and formats |
| **Timeliness**        | Very recent | Coverage in 2025, up to June |
| **Relevance**         | Strong      | Directly aligned with education research goals |
| **Representativeness**| Moderate    | Balanced by gender but limited to Multan region |


## 7. Limitations
- **Regional scope**: Data is specific to Multan, limiting generalizability.  
- **Missing influencing factors**: Nutrition, sleep quality, and mental health are absent.  
- **Imperfect measures**: GPA and GradeClass are affected by subjectivity and institutional grading policies.  


## 8. Stakeholders, Beneficiaries, and Impact
**Stakeholders**:  
- School administrators (resource planning, curriculum design)  
- Teachers (personalized support, early interventions)  
- Parents (understanding influence of home environment)  
- Policymakers & NGOs (evidence for equity-driven policies)  

**Beneficiaries**:  
- Students (tailored interventions, reduced risk of exclusion)  
- Teachers (better understanding of performance drivers)  
- Communities (improved educational outcomes, reduced inequality)  

**Impact**:  
- **Early detection** of at-risk students.  
- **Equity in education** by identifying what truly matters (and what doesn’t).  
- **Policy insights** grounded in real-world data.  


## 9. Methodology Overview
The project will follow a full machine learning pipeline:  
1. **EDA**: Explore distributions, correlations, and key patterns.  
2. **Data Cleaning & Preprocessing**: Handle missing values, encode categorical features, scale numeric features.  
3. **Feature Selection**: Use correlation and domain knowledge to refine predictors.  
4. **Modeling**: Apply regression models (Linear Regression, Ridge, Lasso, etc.).  
5. **Evaluation**: Assess with metrics (R², RMSE, MAE) and interpret coefficients.  
6. **Ethical Reflection**: Evaluate fairness, limitations, and transparency of predictions.  
7. **Communication**: Visualize results and prepare stakeholder-ready insights.  


## 10. Next Steps
- Conduct correlation analysis and feature engineering.  
- Implement baseline regression models.  
- Evaluate performance and optimize models.  
- Document findings, ethical implications, and stakeholder impact.  


## 11. References
- Kaggle dataset: [Student Performance Data](https://www.kaggle.com/datasets/muhammadazam121/student-performance-data)  
- OECD (2023). *PISA Results on Equity in Education.*  
- UNESCO (2024). *Global Education Monitoring Report.*  


## 12. Acknowledgements
Dataset curated by **Muhammad Azam**, Kaggle. Licensed under **CC BY 4.0**.  
Prepared as part of an academic project on data-driven education and sustainability.  


## 13. Full Impact Report (including Stakeholder Summary)
For a detailed write-up, see the [Notion Report](https://www.notion.so/Beyond-Grades-Impact-Report-26392f104f7a8076a150d04b9822a438?source=copy_link)
  