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
- Encoded countries and type.
- Created histograms, pairplots, and counterplots.

## Jupyter Notebooks
- [Skincare Recommender Notebook](https://github.com/nicoleli04/CSE151A-Project/blob/main/Skincare_Recommender.ipynb)

## Group Members
- Ilia Aballa - aballai
- Kiran Keertipati - KiranKeertipati
- Leena Khattat
- Nicole Li - nicoleli04
- Wang Liu - flanneryxiao
- Evelyn Mares-moreno - emaresmoreno

