# Student Performance Prediction

## Overview
This project aims to predict students' final exam scores based on their study hours and previous scores using Linear Regression. The dataset consists of 20 students with three key variables:
- Study Hours
- Previous Scores
- Final Exam Score (target variable)

## Dataset
The dataset is stored in `student_performance.csv` and contains the following columns:
- `StudentID`: Unique identifier for each student
- `StudyHours`: Number of hours a student studied
- `PreviousScores`: Scores obtained in previous assessments
- `FinalExamScore`: Actual scores obtained in the final exam

## Dependencies
Ensure you have the following Python libraries installed:
```sh
pip install pandas numpy matplotlib scikit-learn
```

## Workflow
1. **Data Preparation**:
   - A dataset is created and saved as a CSV file.
   - The dataset is loaded using Pandas and previewed.

2. **Feature Selection**:
   - Independent variables: `StudyHours` and `PreviousScores`
   - Dependent variable: `FinalExamScore`

3. **Train-Test Split**:
   - 80% of the data is used for training, and 20% for testing.

4. **Model Training & Evaluation**:
   - A Linear Regression model is trained on the training dataset.
   - Model performance is evaluated using:
     - Mean Absolute Error (MAE)
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)

5. **Visualization**:
   - Scatter plot of actual vs. predicted scores.
   - Bar graph comparing study hours and final exam scores.
   - Line graph comparing previous scores and final exam scores.

## Execution
Run the Python script to generate predictions and visualize the results:
```sh
python student_performance.py
```

## Results
The model outputs:
- Coefficients for `StudyHours` and `PreviousScores`
- Intercept of the regression line
- Performance metrics (MAE, MSE, RMSE)

Additionally, visualizations help in understanding the correlation between study habits, previous performance, and final exam results.

## Future Improvements
- Increase dataset size for better accuracy.
- Consider additional features like attendance and engagement levels.
- Experiment with advanced regression models.

## Author
This project was implemented as part of a learning experience in Machine Learning and Data Analysis.

