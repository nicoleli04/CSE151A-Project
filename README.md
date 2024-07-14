# Skincare Recommender System

## Abstract
Taking care of one’s skin is essential for both physical and mental wellbeing. Our model aims to create a recommender system that suggests products aligned with users' skincare concerns through a quick survey. We will use explit feedback like reviews from other users to identify suitable skincare products. Using datasets from Ulta and Sephora, we acknowledge potential biases in our model, such as the exclusion of lesser-known products, a focus on products used primarily by Westerners, and a predominance of data representing women or female-identifying individuals. Despite these biases, our ultimate goal is to enhance accessibility and knowledge of skincare for people of all skin types who aspire to improve their skincare routine.

## Datasets
- [Skinsort Dataset](https://www.kaggle.com/datasets/kazireyazulhasan/19000-skincare-products-database-of-skinsort): Contains information about skincare products.
- **__________**: Contains user reviews for skincare products.

## Data Preprocessing
- Handled missing values and duplicates.
- Performed feature engineering and text preprocessing.
- We encoded countries and type
- created histograms, pairplots and counterplots
- merged and filtered the data to not include makeup products
- created similarity matrix to match ratings with products names in skinsort(product data set)
- merged all rating data sets to into one large data set
- cleaned data from the new large merged data set
- We created a threshold in order to filter out uncoorelated product names between skinsort (product data set) and the large review data set

## Jupyter Notebooks
- [Skincare Recommender Notebook](https://github.com/nicoleli04/CSE151A-Project/blob/main/Skincare_Recommender.ipynb)

  [tester](https://colab.research.google.com/drive/1BZMUc67qGvuqpEeNsp83_C0oyVc02UhN#scrollTo=8sfUzruRQy05)
## Group Members
- Ilia Aballa
- Kiran Keertipati 
- Leena Khattat
- Nicole Li 
- Wang Liu
- Evelyn Mares-moreno 

