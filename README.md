[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=15538997&assignment_repo_type=AssignmentRepo)
# Project Overview: Advanced Machine Learning Assignment
## Project Title: Predicting Customer Churn and Segmentation for a Telecom Company

This project involves predicting customer churn (supervised learning) and performing customer segmentation (unsupervised learning) using a dataset from a telecom company. The dataset contains information about customer demographics, account information, services subscribed, and usage metrics.

- Dataset Name: Telco Customer Churn
- Description: This dataset includes customer data from a telecommunications company, which provides services to different customers. It contains various features such as customer demographics, services subscribed, and tenure, making it suitable for both classification tasks (predicting churn) and clustering tasks (customer segmentation).


# Project Breakdown
## 1. Problem Definition
Objective:
- Churn Prediction: Predict whether a customer will churn (leave the company) based on their usage data, account information, and demographic details.
- Customer Segmentation: Group customers into different segments based on their behavior and characteristics to tailor marketing and service strategies.
  - Output: A classification model to predict customer churn and a clustering model for customer segmentation.

## 2. Data Understanding
Tasks:
- Dataset Exploration: Understand the features, types of variables, and target the variable (Churn).
- Data Description: Review the provided data to understand the meaning of each feature and how it relates to customer churn. 
  - Outcome: A clear understanding of the dataset, including feature distributions and potential correlations.

## 3. Data Preparation
Tasks:
- Data Cleaning: Handle missing values, remove duplicates, and address any data quality issues.
- Feature Engineering: Create new features that could help improve the prediction, such as customer lifetime value or usage trends.
- Categorical Encoding: Convert categorical features (e.g., gender, contract type) into numerical representations.
- Feature Scaling: Standardize numerical features to ensure they are on a comparable scale, particularly for clustering.
- Imbalanced Data Handling: Since churn is often an imbalanced problem, use techniques like SMOTE (Synthetic Minority Over-sampling Technique) or class weighting to address this.
   - Outcome: A well-prepared dataset ready for model training, with features engineered to improve model performance. The prepared dataset should be saved as a file as clean_data.csv. 


## 4. Exploratory Data Analysis (EDA)
Tasks:
- Univariate Analysis: Analyze each feature individually to understand its distribution and how it may affect churn.
- Bivariate and Multivariate Analysis: Explore relationships between features, particularly focusing on correlations with the target variable (Churn).
- Visualization: Use visual tools like histograms, box plots, scatter plots, heatmaps, and pair plots to uncover patterns.
- Correlation Analysis: Identify multicollinearity issues and decide on feature selection or dimensionality reduction.
  - Outcome: Insights into key features that influence churn and potential feature interactions. Here it is expected you give explanations of what you have found out. 


## 5. Model Development
### Supervised Learning:
- Classification Models: Train various classification models such as Logistic Regression, Decision Trees, Random Forest, Gradient Boosting Machines, XGBoost, and Neural Networks to predict churn. Note that in this, we have not covered XGBoost and Neural networks, so you can do them as a research or if you are not sure, you can leave them out. I reccommend that you give it a trial as it will help you in your lifelong learning tech industry and data science. 
- Handling Imbalance: Use techniques like class weighting, oversampling, or undersampling during training. *Also this is not covered, but you can research so that you learn how to produce good model*
### Unsupervised Learning:
- Clustering Models: Apply clustering techniques like K-Means, Hierarchical Clustering, and DBSCAN to segment customers.
- Dimensionality Reduction: If necessary, use techniques like PCA or t-SNE before clustering to reduce the dimensionality of the data.
    - Outcome: A set of trained models for churn prediction and customer segmentation.


## 6. Model Evaluation
Tasks:
- Performance Metrics for Classification: Evaluate models using metrics like Accuracy, Precision, Recall, F1-Score, AUC-ROC, and Confusion Matrix.
- Cross-Validation: Use k-fold cross-validation to assess the robustness and generalization of the classification models.
- Hyperparameter Tuning: Use techniques like GridSearchCV or RandomizedSearchCV to optimize model parameters.
- Clustering Evaluation: Evaluate clustering models using metrics such as silhouette score, Davies-Bouldin Index, and inertia (for K-Means).
  - Outcome: Identification of the best models for both churn prediction and customer segmentation based on their performance metrics.


## 7. Model Selection and Finalization
Tasks:
- Best Model Selection: Choose the best-performing classification model for churn prediction and the best clustering model for segmentation.
- Model Interpretation: Use techniques like SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) to interpret model predictions.
- Final Model Training: Retrain the selected models on the full dataset if applicable. (This applies to the bits where you did not use the whole dataset at first)
- Model Export: Save the final models for deployment or further analysis.
  - Outcome: Finalized models ready for deployment, with clear interpretations of their predictions.


## 8. Model Deployment (Optional)
Tasks:
- Model Serialization: Save the trained models using joblib or pickle.
- API Development: Develop an API using Flask or FastAPI to serve the model predictions.
- Deployment: Deploy the API on a cloud platform like Heroku or AWS for real-time predictions.
   - Outcome: A deployed model ready for integration with business processes or customer-facing applications.


## 9. Documentation and Reporting
Tasks:
- Documentation: Document the entire process, including the rationale behind each decision, challenges faced, and solutions implemented.
- Reporting: Create a comprehensive report summarizing the findings, model performance, customer segments, and potential business strategies.
  - Outcome: A well-documented project that can be shared with stakeholders, demonstrating a deep understanding of machine learning in a business context. Please save it as a pdf file and include in your repo. 


### References: 
- https://muhammaddawoodaslam.medium.com/model-evaluation-and-metrics-in-data-science-1204c2004555#:~:text=Cross%2Dvalidation%20is%20a%20technique,validation%20and%20stratified%20cross%2Dvalidation.
- https://medium.com/@brandon93.w/cross-validation-in-data-science-c87974f8f7d
- Model interpretability: https://towardsdatascience.com/three-interpretability-methods-to-consider-when-developing-your-machine-learning-model-5bf368b47fac
- SHAP and LIME Explained: https://medium.com/analytics-vidhya/shap-shapley-additive-explanations-and-lime-local-interpretable-model-agnostic-explanations-8c0aa33e91f
- Exporting Machine Learning models guide: https://saturncloud.io/blog/exporting-machine-learning-models-a-comprehensive-guide-for-data-scientists/#:~:text=Model%20exporting%20is%20the%20process,make%20predictions%20on%20new%20data.
- File types used to export ML Models: https://saturncloud.io/blog/exporting-machine-learning-models-a-comprehensive-guide-for-data-scientists/#:~:text=Model%20exporting%20is%20the%20process,make%20predictions%20on%20new%20data.
- 
