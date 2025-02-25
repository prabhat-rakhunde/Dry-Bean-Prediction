# Dry Bean Prediction

## Project Overview
The **Dry Bean Prediction** project focuses on classifying different types of dry beans based on their physical characteristics using machine learning models. The project involves **exploratory data analysis (EDA)** and **model building** to improve classification accuracy.

## Dataset Description
The dataset consists of several features related to the shape and size of dry beans, including:
- **Area**: The number of pixels within the bean's boundary.
- **Perimeter**: The total length of the bean's boundary.
- **Major Axis Length**: The longest line passing through the bean.
- **Minor Axis Length**: The shortest line passing through the bean.
- **Eccentricity**: A measure of bean shape elongation.
- **Convex Area**: The number of pixels of the smallest convex shape that encloses the bean.
- **Equivalent Diameter**: The diameter of a circle with the same area as the bean.
- **Solidity**: The ratio of bean area to its convex area.
- **Aspect Ratio**: Ratio of the major axis length to minor axis length.
- **Class Label**: The type of dry bean, which includes **SEKER, BARBUNYA, BOMBAY, CALI, HOROZ, SIRA, DERMASON**.

## Analysis Conducted
1. **Data Preprocessing**:
   - Handling missing values and outliers.
   - Feature scaling using standardization techniques.
   - Encoding categorical variables (if necessary).
2. **Exploratory Data Analysis (EDA)**:
   - Distribution analysis of numerical features.
   - Correlation heatmaps to identify feature importance.
   - Box plots and scatter plots for visual interpretation.
3. **Machine Learning Model Development**:
   - **Random Forest Classifier**: Best-performing model with the highest accuracy.
   - **Logistic Regression**: Baseline classification model.
   - **K-Nearest Neighbors (KNN)**: Distance-based classification.
   - **Support Vector Machine (SVM)**: Effective in high-dimensional spaces.
   - **Model Evaluation**: Comparing accuracy, precision, recall, and F1-score.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- **Visualization**: Histograms, correlation heatmaps, scatter plots
- **Machine Learning Models**: Random Forest, Logistic Regression, KNN, SVM

## Key Insights
- **Random Forest Classifier** achieved the best accuracy among tested models.
- Features such as **Area, Perimeter, and Major Axis Length** were highly correlated with bean classification.
- Proper feature selection and scaling improved overall model performance.

## Future Enhancements
- Expanding the dataset with additional bean varieties for better generalization.
- Applying **deep learning models** such as Convolutional Neural Networks (CNNs) for improved classification.
- Developing an **interactive web application** for real-time bean classification.

## How to Use the Project
1. Load the dataset into **Jupyter Notebook**.
2. Execute data preprocessing scripts to clean and prepare data.
3. Train machine learning models and evaluate their performance.
4. Modify feature selection and hyperparameters to optimize accuracy.

## Author
Prabhat Rakhunde
