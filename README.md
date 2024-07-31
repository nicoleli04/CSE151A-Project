# Skincare Recommender System

#MILESTONE 4 NOTEBOOK:#
[MileStone 4 nb](https://colab.research.google.com/drive/1UZuKvfh_-BFqsnquGwFd7OzI7wOTJZH0#scrollTo=MTfIhu3hAh8P)

# WORK IN PROGRESS #

# Writeup Table of Contents:
1. [Introduction](#introduction)
2. [Figures](#figures)
3. [Methods Section](#methods)
4. [Data Exploration](#data)
5. [Preprocessing](#preprocessing)
6. [Model 1: Recommendation System](#model1)
7. [Model 2: Linear Regression (Price and Reviews)](#model2)
8. [Logistic Regression (Acne Product Recommender)](#model3)
9. [Results Section](#results)
10. [Discussion Section](#discussion)
11. [Conclucsion](#conclusion)
12. [Statement of Collaboration](#collab)

## Introduction: <a name="introduction"></a>
Taking care of one’s skin is essential for both physical and mental wellbeing, however finding the right skincare products can be challenging given the large variety available. Products differ in many categories such as ingredients, purpose, and price. Along with that, their effectiveness is subjective, as it is dependent on the user's own skin type and allergens. The goal of our project is to address this challenge by creating a recommendation system that suggests skincare products based on user concerns and preferences. 

This project also aims to predict product prices and customer reviews, providing valuable insights into the cost dynamics of the skincare market. 


This project is particularly interesting because it combines elements of machine learning with skincare. By leveraging datasets from well-know beuaty retailers and by focussing on recommendations, this project has the potential to make skincare advice more accessible and personalized. Overall the broader impact of this project includes the personilzation of skincare, increased accessibility and economic insights.


Prior to our first group meeting, all members were responsible for thinking of an idea and finding datasets that might be useful for that idea. A group member came up with an idea to create a Skincare related project, and the rest of the group was interested in moving forward with the idea. The datasets the member found were presented and seemed promising, as they (at the time) seemed large enough to work with, containing columns detailing brand, product name, price, reviews, country of origin, and other information related to the product. Doing a topic on Skincare seemed like a unique topic that our members were interested in working with, and we were leaning towards creating a recommendation system that presented products based on a user's response to a survey, where we'd ask them for things like skin type, ingredients, allergens, and specific skin issues that they would want to address like dryness, acne, and so on. In our minds, we were to make a model that was meant to predict what product would be the best for our user to use. 

We were unsure of what exact model we would want to use, as a recommendation system was not yet covered in the class, and we realized that we would have to do individual research and attend office hours to get a better grasp of how to approach the project. At office hours, Professor Solares told us that we would have to find a way to vectorize our data, and emphasized the importance of reviews in our process. 


## Figures: <a name="figures"></a>

Below is a figure that shows a tree with the different types of Recommender Systems, from the research paper [A systematic review and research perspective on recommender systems](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00592-5)

![image](https://github.com/user-attachments/assets/03af8f99-80ee-41e8-a8c8-b13b49d76752)

The approach that seemed the most relevant to our problem is a model-based filtering technique where we use both association techniques and clustering approaches to see what products are the most similar to one another.

Here is another figure that we used to help understand how our model should work, from (this Kaggle notebook)[https://www.kaggle.com/code/dyahnurlita/skincare-recommendation-system-using-cf-method]. While they also implemented a recommendation system, theirs is different from ours as they use reviews and rating as the main method for recommending a product, while ours is different, focusing instead on other things like ingredients and use of the product. 
![image](https://github.com/user-attachments/assets/48c185fc-91fd-4ebe-a118-c757d205f0f0)






## Methods Section: <a name="methods"></a>


### Data Exploration <a name="data"></a>


### Preprocessing <a name="preprocessing"></a>


### Model 1: Recommendation System <a name="model1"></a>


### Model 2: Linear Regression (Price and Reviews) <a name="model2"></a>


### Model 3: Logistic Regression (Acne product recommender) <a name="model3"></a>

## Results Section:  <a name="results"></a>

## Discussion Section  <a name="discussion"></a>

## Conclusion <a name="conclusion"></a>

## Statement of Collaboration  <a name="collab"></a>
Overall, all members contributed to the project, as the project was mostly worked on with multiple people working together at a time. We all joined several zoom meetings together, or met in-person as a team to work on the project. We would share our screen and take turns programming different sections, often with feedback or edits from the others. We brainstormed all the ideas for the project together, bouncing ideas off of and changing ideas based on everybody's feedback. We also took turns going to office hours to discuss with TAs, and sometimes all of us attended together. For the more specific tasks, they are listed below. 

Evelyn Mares-moreno: Talked to tutors, wrote code for Milestone 4 and updated group members about these changes 

Ilia Aballa: Uploaded early datasets, helped to organize some meetings/reserve study room, completed writeup

Kiran Keertipati: Organized many meetings, met with Brian one on one to discuss the project, main screen-sharer for group coding sessions

Leena Khattat: Helped with encoding implementation, implemented LogReg model for Milestone 4

Nicole Li: Created Github Repo, helped write the similarity matrix 

Wang Liu: Helped to upload large files to Github and remove unnecessary files, helped write similarity matrix 

All: Helped clean/organize/understand data, participated in group coding sessions and brainstorming, took turns submitting Milestones on Gradescope, updated the README in some way over the four milestones

-----------------------------------------------------------------------------------------------------------------------------------
## README Prior to Writeup: ##

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
- [Skincare Recommender with Two Models](https://github.com/nicoleli04/CSE151A-Project/blob/main/Two_Model_Predictor_Skincare_Recommender.ipynb)

## Group Members
- Ilia Aballa
- Kiran Keertipati 
- Leena Khattat
- Nicole Li 
- Wang Liu
- Evelyn Mares-moreno

