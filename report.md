<!-- #region -->
# Assignment 2

---

## Abstract 

This report explores the application of predictive modeling techniques for personalized recipe recommendations using a dataset sourced from Food.com. The dataset comprises two key components: recipe metadata and user-recipe interaction data. By leveraging user preferences and recipe attributes, the primary objective is to develop a machine learning model that predicts user ratings for recipes. This task is motivated by the broader goal of designing a recommender system tailored to user preferences in the culinary domain. Through exploratory data analysis (EDA), we uncover patterns, trends, and anomalies in the data, which inform feature engineering and model selection. The findings and methodologies presented in this report serve as a foundation for building a robust recommendation engine, showcasing the potential for data-driven personalization in the context of online recipe platforms.

--- 

## Data

The dataset used in this study is sourced from [Food.com](https://www.food.com/) and comprises two key components: **recipe metadata** and **user-recipe interactions**. The `recipe` dataset contains detailed information about individual recipes, including attributes such as preparation time, number of steps, and list of ingredients, while the interaction dataset captures user ratings and reviews for recipes. Together, these datasets form the foundation for building a recommender system that predicts user ratings for recipes.

### Dataset 1 : `recipes` 

#### Size and Composition

The recipe dataset consists of **231,637** entries and 12 columns, representing the metadata of various recipes. Each row corresponds to a unique recipe, with features such as `name`, `id`, `minutes`, `n_steps`, `n_ingredients`, and others.

#### Key Statistics:

|       |     id |          minutes |   contributor_id |      n_steps |   n_ingredients |
|:------|-------:|-----------------:|-----------------:|-------------:|----------------:|
| count | 231637 | 231637           | 231637           | 231637       |    231637       |
| mean  | 222015 |   9398.55        |      5.53489e+06 |      9.7655  |         9.05115 |
| std   | 141207 |      4.46196e+06 |      9.97914e+07 |      5.99513 |         3.7348  |
| min   |     38 |      0           |     27           |      0       |         1       |
| 25%   |  99944 |     20           |  56905           |      6       |         6       |
| 50%   | 207249 |     40           | 173614           |      9       |         9       |
| 75%   | 333816 |     65           | 398275           |     12       |        11       |
| max   | 537716 |      2.14748e+09 |      2.00229e+09 |    145       |        43       |


- Recipes require an average of **9.8 steps** and use approximately **9 ingredients on average**.
- The preparation time (minutes) is highly variable, **ranging from 0 minutes to 2.14 billion minutes**, the latter being an apparent outlier.
- Missing data is minimal, with only **4,979 missing descriptions** and **one missing recipe name**.

![Distribution of Numerical Features](images/recipesNumerical.png)


#### Interesting Findings:

- The distribution of minutes is skewed, with most recipes **requiring under 100 minutes**, but extreme outliers significantly inflate the range. 
- A small number of recipes share very common ingredient combinations, such as ['eggs', 'water'], ['flour', 'baking powder', 'salt'], and ['butter', 'sugar', 'flour'].
- Recipes show varying levels of complexity, with the number of steps (`n_steps`) and number of ingredients (`n_ingredients`) distributed broadly, indicating diversity in the types of recipes available.

### Dataset 2 : `interactions`

#### Size and Composition
The interaction dataset contains **1,132,367** entries across 5 columns: `user_id`, `recipe_id`, `date`, `rating`, and `review`. Each row represents an interaction between a user and a recipe, where the rating indicates user satisfaction.

#### Key Statistics:

|       |          user_id |        recipe_id |      rating |
|:------|-----------------:|-----------------:|------------:|
| count |      1.13237e+06 |      1.13237e+06 | 1.13237e+06 |
| mean  |      1.38429e+08 | 160897           | 4.41102     |
| std   |      5.01427e+08 | 130399           | 1.26475     |
| min   |   1533           |     38           | 0           |
| 25%   | 135470           |  54257           | 4           |
| 50%   | 330937           | 120547           | 5           |
| 75%   | 804550           | 243852           | 5           |
| max   |      2.00237e+09 | 537716           | 5           |

- The dataset includes **226,570 unique users** and **231,637 unique recipes**, suggesting wide user coverage across recipes.
- Ratings are predominantly positive, with **72% of interactions rated 5 stars** and very few ratings of 1 or 2 stars.

![Distribution of Numerical Features](images/interactionsNumerical.png)


#### Interesting Findings:


- A significant proportion of reviews are highly positive, which may introduce a bias into the model. This skew suggests that users are generally satisfied with the recipes they interact with, but it also presents challenges for the model to predict lower ratings.
- A small fraction of interactions (169 entries) have missing reviews, but these do not affect the integrity of the primary rating column.
- Sparse interactions across users and recipes highlight the long-tail nature of the data, with some recipes and users having disproportionately high levels of activity.

![Distribution of Ratings](images/ratingDistribution.png)

---

## Predictive Task

Recommender systems have become an integral component of personalized user experiences, particularly in domains such as e-commerce, entertainment, and online platforms. In this study, we focus on the culinary domain, leveraging data from Food.com to design a recommendation engine that predicts user ratings for recipes. With over 1.1 million user-recipe interactions and metadata for over 230,000 recipes, the dataset offers a rich source of information to uncover user preferences and predict future interactions.

The primary task involves building a predictive model to estimate user ratings for unseen recipes based on historical interactions and recipe attributes. This task is not only a foundational challenge in recommender systems but also has practical implications for improving user engagement and satisfaction on recipe-sharing platforms. By incorporating both recipe-level features (e.g., preparation time, ingredients, number of steps) and user-recipe interactions, this study aims to identify key predictors of user satisfaction and design a model that can generalize effectively across diverse user groups.

To address the predictive task, we aim to predict **user ratings for recipes** by leveraging a combination of user-recipe interactions and recipe metadata. This task is critical for designing a personalized recommendation system that suggests recipes aligned with user preferences and past behavior. Below, we outline the steps for evaluating the model, selecting baselines, assessing predictions, and processing the data to extract relevant features.

### Evaluation of the Model

To evaluate the model's performance, we will employ a hybrid of the following metrics:

#### Root Mean Squared Error (RMSE):

RMSE measures the average magnitude of error between predicted ratings and actual ratings. It is suitable for evaluating numerical predictions like ratings and penalizes larger errors more heavily.

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y_i})^2}$$

The dataset shows a significant skew toward higher ratings (e.g., a majority of ratings are 4 or 5). This bias implies that small deviations in predictions for high ratings (e.g., predicting 4.2 instead of 5) might not matter as much, but large errors (e.g., predicting 1 instead of 5) could be critical.

RMSE might be better suited to capture these large deviations.

#### Mean Absolute Error (MAE)

MAE provides the average absolute difference between predicted and actual ratings, offering an intuitive understanding of prediction accuracy.

$$MAE = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y_i}|$$
While most ratings are concentrated around the 4-5 range, there are instances of low ratings (e.g., 1s or 2s). If outliers are genuine reflections of user dissatisfaction, RMSE could emphasize their impact. If they are noisy or random, MAE would be more appropriate as it does not overreact to such points.

### Baselines for Comparison

We will compare our model's performance against the Global Average Rating baseline. We would predict the global average rating for all recipes as a simple baseline.

$$\hat{y} = \frac{1}{n} \sum_{i=1}^n y_i$$

### Features and Data Processing

To ensure the model effectively predicts user ratings, we will carefully preprocess the data and select relevant features.

### Features Used:

**Recipe Metadata:**
- Preparation Time (`minutes`): Recipes with shorter preparation times may correlate with higher ratings for users with busy schedules.
- Number of Steps (`n_steps`): Indicates recipe complexity, which might influence user ratings.
- Number of Ingredients (`n_ingredients`): Similar to steps, this could signal simplicity or difficulty.

**User-Recipe Interaction Data:**
- User ID (`user_id`) and Recipe ID (`recipe_id`): Encoded as numerical features for similarity functions, like Jaccard, cosine, etc.
- Review Text (`review`): Sentiment analysis can extract insights about user satisfaction. This would require some feature extraction. 
- Rating (`rating`): Target variable for supervised learning.

### Assessment of Model Validity

To validate the model's predictions and ensure robustness:

#### Cross-Validation:
Employ K-fold cross-validation to mitigate overfitting and assess model performance on unseen data.

#### Error Analysis:
Analyze residuals to identify patterns where the model underperforms, such as specific user or recipe segments.

#### Baseline Comparison:
Ensure the model consistently outperforms baseline metrics like global average and collaborative filtering.
<!-- #endregion -->

```python

```
