## Problem Statement

Customer churn is a critical challenge for the banking industry, as losing existing customers directly impacts revenue and increases acquisition costs. Identifying customers who are likely to leave in advance enables organizations to take proactive retention measures.

The objective of this project is to develop a machine learning model capable of predicting customer churn based on demographic, financial, and behavioral attributes. The model aims to accurately identify high-risk customers, enabling targeted interventions to improve customer retention.

Given the imbalanced nature of the dataset, special emphasis is placed on optimizing recall to minimize false negatives, ensuring that potential churn customers are not overlooked. The project also focuses on interpreting model predictions to uncover key drivers of churn, thereby supporting data-driven business decision-making.

This end-to-end solution integrates data preprocessing, model development, performance optimization, and business insights to deliver a practical and actionable churn prediction system.



## Exploratory Data Analysis

The dataset was first examined to understand its structure, quality, and underlying patterns influencing customer churn. The dataset consists of 10,000 customer records with no missing values, indicating a clean and reliable data source for analysis.

An initial assessment of the target variable revealed a slight class imbalance, with approximately 79.6% of customers retained and 20.4% having churned. While not extreme, this imbalance highlights the need for careful evaluation metrics beyond accuracy during model development.

Several key patterns were identified during the analysis:

- **Geography** emerged as a significant factor, with customers in Germany exhibiting nearly double the churn rate compared to those in France and Spain.
- **Customer activity** showed a strong relationship with churn, where inactive customers were significantly more likely to leave than active ones.
- **Number of products** played a critical role, with customers holding two products showing the lowest churn rates, while those with only one product or more than two products demonstrated higher churn tendencies.
- **Age** revealed a non-linear relationship with churn, with middle-aged customers (approximately 45–60 years) exhibiting the highest churn rates.
- **Account balance** indicated that customers with higher balances tend to churn more, suggesting potential dissatisfaction among high-value customers.
- **Credit card ownership** showed minimal impact on churn, indicating it is not a strong predictive feature.

It is important to note that certain observations, particularly for customers with three or more products and those with very high balances, are based on smaller sample sizes and should be interpreted with caution.

Overall, the analysis highlights that customer engagement and product usage are the most influential factors affecting churn, providing a strong foundation for building predictive models and designing targeted retention strategies.



## Data Preprocessing and Preparation

Following the exploratory data analysis, the dataset was prepared for machine learning modeling through a series of preprocessing steps. Non-informative features such as unique identifiers and irrelevant textual data were removed to ensure that only meaningful variables were retained.

The dataset was then divided into input features and the target variable representing customer churn. Categorical variables, including geography and gender, were transformed into numerical representations using one-hot encoding. To avoid redundancy and multicollinearity, one category from each encoded variable was dropped and treated as the baseline.

Subsequently, the dataset was split into training and testing sets using an 80:20 ratio. Stratified sampling was applied to maintain the original class distribution of churn across both sets, ensuring reliable model evaluation.

To further enhance model performance, feature scaling was performed using standardization. The scaling parameters were learned exclusively from the training data and then applied to the test data to prevent data leakage.

At the end of this phase, the dataset was fully prepared for modeling, with clean, scaled, and properly structured inputs suitable for training machine learning algorithms.


## Model Development and Evaluation

To model customer churn, a Logistic Regression classifier was implemented as a baseline approach. Logistic Regression was selected due to its interpretability and effectiveness in binary classification problems.

The model was trained using the preprocessed and scaled training dataset. To address class imbalance in the target variable, class weights were set to "balanced", ensuring that the minority class (churn) received appropriate importance during training. Additionally, the maximum number of iterations was increased to ensure proper convergence of the model.

Once trained, the model was used to generate predictions on the test dataset. Two types of outputs were obtained: class predictions and probability scores. The class predictions provided binary outputs (0 = non-churn, 1 = churn) based on a default threshold of 0.5. In addition, probability scores were extracted using the predict_proba() function, which returns the likelihood of each class. The probability corresponding to the churn class was specifically used for further analysis.

Model performance was evaluated using multiple metrics to capture different aspects of prediction quality. Accuracy was used to measure overall correctness, while precision quantified the proportion of correctly predicted churn cases among all predicted churn instances. Recall measured the model’s ability to correctly identify actual churn customers, which is particularly important in churn prediction tasks. The F1 score provided a balance between precision and recall. Finally, the ROC-AUC score was used to evaluate the model’s ability to distinguish between churn and non-churn classes across different classification thresholds.

This evaluation framework provided a comprehensive understanding of model performance and established a baseline for further improvements through threshold tuning and advanced modeling techniques.

## Threshold Tuning and Model Optimization

To enhance model performance and align predictions with business objectives, threshold tuning was performed on the Logistic Regression model. Instead of relying on the default classification threshold of 0.5, multiple thresholds (0.3, 0.4, 0.5, and 0.6) were evaluated to analyze their impact on performance metrics.

The results demonstrated a clear trade-off between precision and recall. Lower thresholds significantly increased recall, enabling the model to capture a larger proportion of churn customers. For instance, at a threshold of 0.3, the model achieved a recall of approximately 0.92, successfully identifying most churn cases. However, this came at the cost of a substantial increase in false positives, leading to reduced precision.

As the threshold increased, precision improved while recall declined. At a threshold of 0.6, the model achieved the highest F1 score, indicating a better statistical balance between precision and recall. However, this configuration resulted in a higher number of false negatives, meaning more churn customers were missed.

To further align evaluation with business priorities, the F2 score was considered, placing greater emphasis on recall. The highest F2 score was observed at a threshold of 0.4, indicating an optimal balance when prioritizing the identification of churn customers.

The confusion matrix analysis reinforced these findings by highlighting the trade-off between false positives and false negatives. Lower thresholds minimized missed churn cases (false negatives) but increased false alarms (false positives), while higher thresholds exhibited the opposite behavior.

Based on these observations, the optimal threshold depends on the business objective. If minimizing customer churn is the primary goal, a threshold of 0.4 is recommended due to its higher recall and F2 score. Alternatively, if a balanced approach is preferred, a threshold of 0.6 may be selected for its higher F1 score.

This analysis demonstrates the importance of threshold tuning in adapting model performance to real-world requirements, moving beyond default settings to achieve more meaningful and actionable outcomes.


## Random Forest Model Optimization and Evaluation

To improve model performance, threshold tuning was applied to the Random Forest classifier. Multiple probability thresholds (0.3, 0.4, 0.5, and 0.6) were evaluated to analyze their impact on classification performance.

The results indicate a clear trade-off between precision and recall. Lower thresholds increased recall, allowing the model to identify a larger proportion of churn customers, while higher thresholds improved precision but led to an increase in missed churn cases.

At a threshold of 0.3, the model achieved the highest F2 score, indicating an optimal balance with emphasis on recall. This configuration resulted in improved identification of churn customers while maintaining a reasonable number of false positives. The confusion matrix analysis further confirmed that this threshold reduced false negatives compared to higher thresholds.

As the threshold increased, precision improved; however, recall decreased significantly. At thresholds of 0.5 and 0.6, the model became more conservative, minimizing false positives but failing to identify a substantial portion of churn customers.

Overall, the Random Forest model demonstrated strong performance in terms of ROC-AUC and precision, indicating good classification capability. However, without threshold tuning, the model exhibited low recall, making it less suitable for churn prediction in its default configuration.

When compared to Logistic Regression, Random Forest achieved a better balance between precision and recall, while Logistic Regression demonstrated superior recall. This highlights the importance of selecting models and thresholds based on business objectives.

In scenarios where maximizing churn detection is critical, Logistic Regression with optimized threshold is preferred. However, for a more balanced approach, Random Forest with a lower threshold provides a viable alternative.

This analysis highlights the importance of aligning model performance with domain-specific objectives, particularly in imbalanced classification problems such as customer churn prediction.


## Feature Importance Analysis

Feature importance analysis was conducted using the Random Forest model to identify the key drivers of customer churn. The results indicate that age is the most significant factor influencing churn, followed by financial attributes such as credit score, estimated salary, and account balance.

Customer engagement variables, including the number of products and tenure, also play a crucial role in determining churn behavior. These findings suggest that both demographic and behavioral factors contribute significantly to customer retention.

In contrast, variables such as gender and credit card ownership were found to have minimal impact on churn prediction. Additionally, geographical factors showed limited but notable influence, particularly for customers located in Germany.

This analysis provides valuable insights into the underlying factors driving churn and enables the development of targeted business strategies to improve customer retention.



## Model Comparison and Final Selection

A comparative analysis was conducted between Logistic Regression and Random Forest models after applying threshold tuning to both approaches.

The Logistic Regression model, optimized at a threshold of 0.4, achieved a significantly higher recall, indicating its strong ability to identify churn customers. In contrast, the Random Forest model, optimized at a threshold of 0.3, demonstrated better balance between precision and recall, reflected in a higher F1 and F2 score.

However, in the context of customer churn prediction, minimizing false negatives is more critical, as failing to identify a churn customer directly impacts business revenue. Therefore, despite Random Forest showing better overall balance, Logistic Regression was selected as the final model due to its superior recall performance.

This decision highlights the importance of aligning model selection with business objectives rather than relying solely on statistical metrics.



## Deployment Considerations

The developed model can be deployed as part of a decision-support system to assist in customer retention strategies. It can be integrated into banking systems to generate real-time or batch predictions of customer churn.

Predictions can be used to flag high-risk customers, enabling proactive engagement through targeted marketing campaigns or personalized offers. The model can also be retrained periodically with updated data to maintain performance over time.

This deployment approach ensures that the model provides continuous value by supporting data-driven decision-making in real-world scenarios.



## Business Recommendations

Based on the model insights, customer churn is primarily influenced by demographic factors, financial status, and engagement levels.

To reduce churn, organizations should focus on identifying high-risk customers and implementing targeted retention strategies. Increasing customer engagement through product bundling, personalized communication, and activity-based incentives can significantly improve retention.

Special attention should be given to high-value customers with large account balances, as they exhibit higher churn tendencies. Additionally, segment-specific strategies should be developed to address regional and demographic differences in customer behavior.

By leveraging these insights, organizations can proactively reduce churn and enhance customer lifetime value.



## Conclusion

This project developed an end-to-end machine learning solution for customer churn prediction, covering data analysis, preprocessing, model development, optimization, and interpretation.

The results demonstrate that model performance must be evaluated in alignment with business objectives, particularly in imbalanced classification problems. While Random Forest provided a balanced performance, Logistic Regression was selected as the final model due to its superior ability to identify churn customers.

Feature importance analysis further provided valuable insights into the key drivers of churn, enabling actionable business strategies.

Overall, this project highlights the importance of combining statistical modeling with domain understanding to deliver practical and impactful solutions.