# Interpretable Machine Learning: A Practical Exploration

This repository contains a series of Jupyter notebooks that explore and implement various techniques for Interpretable Machine Learning (XAI), based on the concepts from the book *Interpretable Machine Learning with Python*. The goal is to move beyond simply building predictive models to understanding *why* they make the decisions they do.

Each notebook tackles a different dataset and focuses on a specific set of interpretation methods, progressing from simple, intrinsically interpretable models to complex, model-agnostic techniques for "black-box" models.

## What is Interpretable Machine Learning (XAI)?

As machine learning models become more complex and integrated into high-stakes decision-making (like in finance, healthcare, and autonomous systems), it's no longer enough for them to be accurate. We need to understand how they work.

**Interpretable Machine Learning (XAI)** is a field dedicated to methods and models that make the predictions and behavior of machine learning systems understandable to humans. It helps us answer questions like:
- Why did the model make this specific prediction?
- Which features are most important for the model's decisions?
- How does the model's output change if we change a specific input?
- Can I trust this model to be fair, robust, and reliable?

## Notebooks Overview

### 01 - Understanding a Simple Weight Prediction Model
This notebook serves as a gentle introduction to model interpretability using the most straightforward example: **Simple Linear Regression**.

- **Goal:** Predict a person's weight based on their height.
- **Model:** A "white-box" `sklearn.linear_model.LinearRegression` model.
- **Work Done & Interpretation Techniques:**
    - Loaded height and weight data directly from a webpage using `pandas.read_html`.
    - Trained a linear regression model to find the best-fit line.
    - **Generated the regression equation (`y = mx + b`)**, providing a clear, mathematical explanation of the relationship.
    - Evaluated the model using **Mean Absolute Error (MAE)** to understand the average prediction error in practical terms (e.g., "the model is off by ~8 pounds on average").
    - Visualized the model's predictions, the actual data points, and the error boundaries using `matplotlib`.
    - Calculated **Pearson's Correlation Coefficient** and the **p-value** to statistically validate the strength and significance of the linear relationship.

### 02 - Interpreting a Logistic Regression Model for CVD Risk
This notebook steps up the complexity by using **Logistic Regression** to classify cardiovascular disease (CVD) risk. It introduces more advanced concepts for interpreting linear models in a classification context.

- **Goal:** Predict the presence or absence of cardiovascular disease.
- **Model:** A `statsmodels.Logit` model, chosen for its detailed statistical summary.
- **Work Done & Interpretation Techniques:**
    - Performed data cleaning and preparation, including handling outliers and anomalous data points.
    - Interpreted model coefficients as **log-odds** and converted them to more intuitive **odds ratios**.
    - Calculated a more robust measure of **feature importance** by combining model coefficients with the standard deviation of each feature, accounting for different scales.
    - Dived into **local interpretation** by generating **2D decision boundary plots** to visualize how pairs of features (e.g., Age vs. Blood Pressure) interact to influence the final prediction.
    - Demonstrated **feature engineering** by creating a Body Mass Index (BMI) feature from height and weight to address non-linearity and improve model interpretability.

### 03 - A Deep Dive into Traditional Interpretable Models
This notebook provides a comparative study of various traditional "white-box" or intrinsically interpretable models, applying them to the task of predicting airline flight delays.

- **Goal:** Predict the duration of carrier-caused flight delays.
- **Models:** A comprehensive suite including Linear Regression, Ridge Regression, Decision Trees, RuleFit, k-NN, and more.
- **Work Done & Interpretation Techniques:**
    - **Linear & Ridge Regression:** Used `statsmodels` to extract detailed statistical outputs like p-values and confidence intervals. Visualized how Ridge regularization shrinks coefficients to reduce model complexity.
    - **Decision Trees:** Explored the core logic of a decision tree by:
        - Visualizing the tree structure to see the exact split conditions.
        - Exporting the tree's logic as a set of human-readable **IF-THEN rules**.
        - Calculating Gini Importance to rank features.
    - **RuleFit:** Implemented this hybrid model that combines the strengths of linear models and decision trees to automatically generate and rank simple linear effects and complex feature interactions.
    - **Critical Analysis:** Investigated the practical challenge of **data leakage**, questioning whether certain features (like `WEATHER_DELAY`) would realistically be available at prediction time.

### 04 - Global Model-Agnostic Interpretation Methods
This notebook shifts focus to interpreting "black-box" models—those whose internal workings are too complex to be easily understood (e.g., gradient boosted trees, neural networks). It applies powerful model-agnostic techniques to a used car price prediction problem.

- **Goal:** Predict the price of a used car using high-performance tree-based models.
- **Models:** `CatBoost` and `RandomForest`.
- **Work Done & Interpretation Techniques:**
    - **Model-Specific Importance:** First, compared the inconsistent built-in feature importance methods of CatBoost and Random Forest to motivate the need for a unified approach.
    - **Permutation Importance:** Measured feature importance by shuffling a feature's values and observing the drop in model performance.
    - **SHAP (SHapley Additive exPlanations):** The main focus of the notebook. This game-theory-based method was used to create rich, detailed explanations:
        - **Global Explanations:**
            - **SHAP Bar Plot:** For a clear, global ranking of feature importance.
            - **Beeswarm Plot:** To visualize not only the importance but also the distribution and direction of a feature's impact on predictions.
            - **Clustering Bar Plot:** To automatically group redundant or interacting features.
    - **Partial Dependence Plots (PDP) & Accumulated Local Effects (ALE):**
        - Visualized the average marginal effect of a feature on predictions.
        - Highlighted the critical weakness of PDPs (the feature independence assumption) and demonstrated why **ALE plots are a more robust alternative**.
        - Used **2D ALE plots** to visualize and understand complex **feature interactions** (e.g., how the effect of `year` on price changes with a car's `odometer` reading).

## Technologies Used
- **Core Libraries:** Python 3, Pandas, NumPy, Scikit-learn
- **Modeling:** Statsmodels, CatBoost
- **Visualization:** Matplotlib, Seaborn
- **Interpretability Libraries:** SHAP, PDPbox, PyALE

## How to Use This Repository
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/majid-200/XAI.git
    ```
2.  **Install dependencies:** It is recommended to create a virtual environment. The required libraries are listed at the top of each notebook. You can install them using pip:
    ```bash
    pip install pandas scikit-learn matplotlib statsmodels catboost shap pdpbox PyALE
    ```
3.  **Run the notebooks:** Launch Jupyter Lab or Jupyter Notebook and open the `.ipynb` files to explore the code and analysis.
    ```bash
    jupyter lab
    ```

## Acknowledgments
This work is a practical implementation of the concepts taught in the book [**Interpretable Machine Learning with Python**](https://github.com/PacktPublishing/Interpretable-Machine-Learning-with-Python) by Serg Masís. All datasets used are sourced as described within the book and notebooks.
