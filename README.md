# Skincare Recommender System

## Abstract
Taking care of one’s skin is essential for both physical and mental wellbeing. Our model aims to create a recommender system that suggests products aligned with users' skincare concerns through a quick survey. We will use explit feedback like reviews from other users to identify suitable skincare products. Using datasets from Ulta and Sephora, we acknowledge potential biases in our model, such as the exclusion of lesser-known products, a focus on products used primarily by Westerners, and a predominance of data representing women or female-identifying individuals. Despite these biases, our ultimate goal is to enhance accessibility and knowledge of skincare for people of all skin types who aspire to improve their skincare routine.

## Datasets
- [Skinsort Dataset](https://www.kaggle.com/datasets/kazireyazulhasan/19000-skincare-products-database-of-skinsort): Contains information about skincare products.
- [Sephora Dataset](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews): Contains product information and reviews from Sephora.
- [Indonesia Dataset](https://www.kaggle.com/datasets/hafidahmusthaanah/skincare-review): Contains product information and reviews from Indonesia.

## Data Preprocessing
- Handled missing values and duplicates.
- Performed feature engineering and text preprocessing.
- Merged all rating data sets (Sephora and Indonesia) into one large Reviews Dataset.
- Removed word-based reviews from the Reviews Dataset, and kept the first 50 reviews for each unique product.
- Created a similarity matrix to match ratings from our Reviews Dataset with products names on our Skinsort Dataset, setting the similarity threshold to 0.4 to filter out uncorrelated product names between Skinsort and the Reviews Dataset.
- Merged and filtered the data to exclude irrelevant products such as face, eye, and hair makeup products.
- Encoded ingediants, brand, results and type.
- One-Hot-Encoded ingrediants, afteruse, brand and type
- dropped null values held in our data set
- Created histograms, pairplots, and counterplots.

## Milestone 3
- Finished major preprocessing by one hot encoding ingredients, afteruse effects, brand, type
- Built our first model using k-nearest neighbors to recommend products based on the features that we one hot encoded
- Printed the top 5 matches for the product using our model

## Conclusion (First Model)
Our first model demonstrates the ability to generate recommendations based on the input product's features. The input product appears as the closest match to itself, which is expected because it shares all its features with itself. However, we noticed that some recommended products are quite different from the input product. For example, when we input "glycolic acid 7 toning solution," one of the recommendations is a lip balm. We believe this discrepancy arises because we have numerous categories that are one-hot encoded, and the model is evaluating the similarities across all these categories. To improve our model, we could assign weights to different categories so that we can ensure that the type of product carries more importance than the ingredients. This way, recommendations will be more aligned with the primary function of the product. We can also perform dimensionality reduction by applying techniques like PCA to reduce the dimensionality of the feature space and focus on the important features. 

*After office hours we found out that we should change our task because our dataset does not have a ground truth so we are now trying to change our model to predict the prices using features from the dataset*

## Milestone Updated
- Used our Sephora dataset and one-hot encoded the product types.
- Cleaned the dataset by removing columns with missing values (NaNs).
- Utilized parameters such as ratings, like count, brand, and type to predict the prices.
- Split the dataset into training and testing sets.
- Calculated the mean squared error.
- Plotted scatter plots for coefficients.

## Conclusion (Second Model Updated)
Our second model uses ratings, like count, brand, and type of the product to predict the price of the product through linear regression. We calculated the mean squared error and found that both the training error and the testing error are quite high. To improve the model, we propose trying polynomial regression to see if it provides a better fit for the data or using perform feature engineering like incorporating interactions between features or adding new relevant parameters to estimate the prices more accurately.

## Jupyter Notebooks
- [Skincare Recommender Notebook](https://github.com/nicoleli04/CSE151A-Project/blob/main/Skincare_Recommender.ipynb)
- [new one](https://colab.research.google.com/drive/19qF-54tqK9wD9ATOslw1Lj7xjF-jtLO7?usp=sharing)

## Group Members
- Ilia Aballa
- Kiran Keertipati 
- Leena Khattat
- Nicole Li 
- Wang Liu
- Evelyn Mares-moreno 

