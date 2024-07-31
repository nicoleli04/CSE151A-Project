# Skincare Recommender System

# MILESTONE 4 NOTEBOOK: #

[MileStone 4 nb](https://colab.research.google.com/drive/1UZuKvfh_-BFqsnquGwFd7OzI7wOTJZH0#scrollTo=MTfIhu3hAh8P)

# WORK IN PROGRESS #

# Writeup Table of Contents:
1. [Introduction](#introduction)
2. [Figures](#figures)
3. [Methods Section](#methods)
   1. [Data Exploration](#data)
   2. [Preprocessing](#preprocessing)
   3. [Model 1: Recommendation System](#model1)
   4. [Model 2: Linear Regression (Price and Reviews)](#model2)
   5. [Model 3: Logistic Regression (Acne Product Recommender)](#model3)
   6. [Model 4: Neural Network (Acne Product Recommender](#model4)
10. [Results Section](#results)
    1. [Data Exploration](#data2)
    2. [Preprocessing](#preprocessing2)
    3. [Model 1: Recommendation System](#model12)
    4. [Model 2: Linear Regression (Price and Reviews)](#model22)
    5. [Model 3: Logistic Regression (Acne Product Recommender)](#model32)
    6. [Model 4: Neural Network (Acne Product Recommender](#model42)
12. [Discussion Section](#discussion)
13. [Conclucsion](#conclusion)
14. [Statement of Collaboration](#collab)

## Introduction: <a name="introduction"></a>
Taking care of one’s skin is essential for both physical and mental wellbeing, however finding the right skincare products can be challenging given the large variety available. Products differ in many categories such as ingredients, purpose, and price. Along with that, their effectiveness is subjective, as it is dependent on the user's own skin type and allergens. The goal of our project is to address this challenge by creating a recommendation system that suggests skincare products based on user concerns and preferences. 

This project also aims to predict product prices and customer reviews, providing valuable insights into the cost dynamics of the skincare market. 


This project is particularly interesting because it combines elements of machine learning with skincare. By leveraging datasets from well-know beuaty retailers and by focussing on recommendations, this project has the potential to make skincare advice more accessible and personalized. Overall the broader impact of this project includes the personilzation of skincare, increased accessibility and economic insights.


Prior to our first group meeting, all members were responsible for thinking of an idea and finding datasets that might be useful for that idea. A group member came up with an idea to create a Skincare related project, and the rest of the group was interested in moving forward with the idea. The datasets the member found were presented and seemed promising, as they (at the time) seemed large enough to work with, containing columns detailing brand, product name, price, reviews, country of origin, and other information related to the product. Doing a topic on Skincare seemed like a unique topic that our members were interested in working with, and we were leaning towards creating a recommendation system that presented products based on a user's response to a survey, where we'd ask them for things like skin type, ingredients, allergens, and specific skin issues that they would want to address like dryness, acne, and so on. In our minds, we were to make a model that was meant to predict what product would be the best for our user to use. 

We were unsure of what exact model we would want to use, as a recommendation system was not yet covered in the class, and we realized that we would have to do individual research and attend office hours to get a better grasp of how to approach the project. At office hours, Professor Solares told us that we would have to find a way to vectorize our data, and emphasized the importance of reviews in our process. 


## Figures: <a name="figures"></a>
Here is the first figure that we used to help understand how our model should work, from [this Kaggle notebook](https://www.kaggle.com/code/dyahnurlita/skincare-recommendation-system-using-cf-method). We found it because we were searching for similar notebooks as we did not want to accidentally copy someone else's work. While this user also implemented a recommendation system, theirs is different from ours as they use reviews and rating as the main method for recommending a product, while ours is different, focusing instead on other things like ingredients and use of the product. This graphic was helpful in developing our understanding of the reviews based recommendation system. 

![image](https://github.com/user-attachments/assets/48c185fc-91fd-4ebe-a118-c757d205f0f0)


Below is a figure that shows a tree with the different types of Recommender Systems, from the research paper [A systematic review and research perspective on recommender systems](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-022-00592-5). The approach that seemed the most relevant to our problem is a model-based filtering technique where we use both association techniques and clustering approaches to see what products are the most similar to one another. However, there are also elements of memory-based filtering based on items. So, in the end our model could be described as a hybrid model using both model and memory-based filtering techniques.

![image](https://github.com/user-attachments/assets/03af8f99-80ee-41e8-a8c8-b13b49d76752) 

## Methods Section: <a name="methods"></a>
### Data Exploration <a name="data"></a>
In exploring our data, we used the following methods:
~~~ 
# read in datasets
sephora = pd.read_csv('/content/CSE151A-Project/SephoraData.csv')
indonesia = pd.read_csv('/content/CSE151A-Project/IndonesiaReviews.csv')
skinsort = pd.read_csv('/content/CSE151A-Project/SkinsortData.csv')
~~~

~~~
# preview datasets
sephora
indonesia
skinsort
~~~

~~~
# check number of prices with 0
sephora.columns
sum(sephora['price_usd'] == 0)
sephora['price_usd'].value_counts()
~~~

~~~
sum(indonesia['Price'] == 0)
~~~

~~~
# reviewing 'afteruse' column
skinsort['afteruse'][8]
~~~


### Preprocessing <a name="preprocessing"></a>
In preprocessing our data, we used the following methods:

#### Section 1: 
~~~
sns.pairplot(merged_df, hue ='Country', palette='RdBu')
~~~

~~~
merged_df.dropna(subset=['Results'], inplace=True)
merged_df.shape
~~~

~~~
merged_df.describe()
merged_df.info()
~~~

~~~
merged_df['Type'].dropna()
~~~

~~~
# renaming columns to match one another 
indonesia.rename(columns = {"Product": "name", "OverallRating": "rating"}, inplace = True)
sephora.rename(columns = {"product_name": "name", "price_usd":"Price"}, inplace = True)

# grabbing only name, rating, and price from each dataframe
s = sephora[['name', 'rating', 'Price']] 
i = indonesia[['name', 'rating', 'Price']]

reviews = pd.concat([s, i]) #concatenate the two datasets into the reviews dataset
reviews = reviews.drop_duplicates(subset=['name']) #drop duplicates
reviews.shape
reviews.value_counts('name')
~~~

#### Section 2:
The following is the similarity matrix code we used:
~~~
from sklearn.feature_extraction.text import TfidfVectorizer #to vectorize
from sklearn.metrics.pairwise import cosine_similarity #to create the similarity matrix
import re #regular expression operations

products_df = pd.DataFrame(skinsort)
reviews_df = pd.DataFrame(reviews)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

products_df['name'] = products_df['name'].apply(preprocess_text)
reviews_df['name'] = reviews_df['name'].apply(preprocess_text)

vectorizer = TfidfVectorizer()
tfidf_products = vectorizer.fit_transform(products_df['name'])
tfidf_reviews = vectorizer.transform(reviews_df['name'])

similarity_matrix = cosine_similarity(tfidf_reviews, tfidf_products)

matched_products = []
similarity_threshold = 0.4  # Set your desired similarity threshold here

sim_matrix_copy = similarity_matrix.copy()

for i in range(len(reviews_df)):
    # Find the index of the highest similarity score for the current review
    best_match_idx = np.argmax(sim_matrix_copy[i])
    # Check if the highest similarity score is above the threshold
    if sim_matrix_copy[i, best_match_idx] >= similarity_threshold:
        # Append the matched product from products_df
        matched_products.append(products_df['name'].iloc[best_match_idx])
        # Set the highest similarity score to a very low value to prevent re-matching
        sim_matrix_copy[:, best_match_idx] = -1
    else:
        # If no match is found above the threshold, append None
        matched_products.append(None)

reviews_df['Matched_Product'] = matched_products

merged_df = products_df.merge(reviews_df.drop(columns='name'), left_on='name', right_on='Matched_Product', how='left')

merged_df = merged_df.dropna(subset=['Matched_Product', 'rating'])

merged_df = merged_df.drop_duplicates('name').drop_duplicates('Matched_Product').drop(columns=['Matched_Product'])

merged_df
merged_df.to_csv('finalskincarelist_df.csv', index=False)
merged_df.head()
~~~

Other preprocessing methods: 
~~~
merged_df.rename(columns = {"brand": "Brand", "name": "Name", "type": "Type", "country": "Country", "ingridients":"Ingredients", "afterUse": "Results", "rating": "Rating"}, inplace = True)
merged_df.head()
~~~

~~~
merged_df.dropna(subset=['Results'], inplace=True)
merged_df.shape
~~~

~~~
merged_df.describe()
~~~

~~~
merged_df.info()
~~~

~~~
merged_df["Type"].dropna()
~~~

~~~
merged_df.loc[merged_df['Type'] == np.nan]
~~~

~~~
merged_df['Type'].value_counts()
~~~

~~~
merged_df["Country"].fillna("Unknown", inplace=True)
merged_df["Country"].value_counts()
~~~


We used the following methods for Encoding our Data:
~~~
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
merged_df['Type_label'] = le.fit_transform(merged_df['Type'])
merged_df['Country_label'] = le.fit_transform(merged_df['Country'])
~~~

~~~
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()

one_hot_encoded = encoder.fit_transform(merged_df[['Brand']]).toarray()

one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(['Brand']))

merged_df = merged_df.join(one_hot_df)

merged_df.drop('Brand', axis=1, inplace=True)

merged_df.head()
~~~

~~~
from sklearn.preprocessing import MultiLabelBinarizer

def str_to_list(x):
  return x.strip().split(',')

merged_df['Results'] = merged_df['Results'].dropna().apply(str_to_list)

mlb = MultiLabelBinarizer()

one_hot_encoded = mlb.fit_transform(merged_df['Results'])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_)

merged_df = merged_df.join(one_hot_df)
~~~

~~~

from sklearn.preprocessing import MultiLabelBinarizer

def str_to_list(x):
  return x.strip().split(',')

merged_df['Ingredients'] = merged_df['Ingredients'].dropna().apply(str_to_list)


mlb1 = MultiLabelBinarizer()

one_hot_encoded1 = mlb1.fit_transform(merged_df['Ingredients'])

one_hot_df1 = pd.DataFrame(one_hot_encoded1, columns=mlb1.classes_)

merged_df = merged_df.join(one_hot_df1)
~~~

~~~
encoder_type = OneHotEncoder()

one_hot_encoded_type = encoder_type.fit_transform(merged_df[['Type']]).toarray()

one_hot_df_type = pd.DataFrame(one_hot_encoded_type, columns=encoder_type.get_feature_names_out(['Type']))

merged_df = merged_df.join(one_hot_df_type)

merged_df.drop('Type', axis=1, inplace=True)

merged_df.head()
~~~

~~~
encoded = merged_df.drop(columns=["Country", "Ingredients", "Results", "Type_label", "Type_nan"]).dropna()
encoded.columns
~~~


### Model 1: Recommendation System <a name="model1"></a>
The following methods were used for training our first model:
~~~
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(merged_df.drop(columns=["Name"]), merged_df['Name'], test_size=0.2, random_state=42)


from sklearn.neighbors import NearestNeighbors

recommender = NearestNeighbors(metric='cosine')
recommender.fit(encoded.drop(columns=["Name"]))
~~~

~~~
product_index = 0

num_recommendations = 5

_, recommendations = recommender.kneighbors(encoded.drop(columns=["Name"]).iloc[[product_index]], n_neighbors=num_recommendations)

recommended_products = merged_df.iloc[recommendations[0]]['Name']

print("OG product:" +  merged_df.loc[product_index,"Name"] + "\nBrand:" + merged_df.loc[product_index,"Name"] )
print("Recommended Products:")
for product in recommended_products:
    print(product + "\nBrand:" + merged_df.loc[merged_df['Name'] == product, 'Name'].iloc[0])
~~~

~~~
product_index = 0

num_recommendations = 5

_, recommendations = recommender.kneighbors(encoded.drop(columns=["Name"]).iloc[[product_index]], n_neighbors=num_recommendations)

recommended_indices = recommendations[0]

print("OG Product:")
print(merged_df.iloc[product_index])

print("\nRecommended Products:")
for index in recommended_indices:
    print(merged_df.iloc[index])
    print("\n")
~~~

### Model 2: Linear Regression (Price and Reviews) <a name="model2"></a>
The following methods were used for training our second model using PRICE:
~~~
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split    
~~~

~~~
df = sephora
df.columns
~~~

~~~
df_final = df[["name","loves_count", "brand_id", "out_of_stock", "rating", "primary_category","Price", 'reviews', 'online_only', 'new', 'child_count']]
df_final.head()
~~~

~~~
encoder_type = OneHotEncoder()

one_hot_encoded_type = encoder_type.fit_transform(df_final[['primary_category']]).toarray()

one_hot_df_type = pd.DataFrame(one_hot_encoded_type, columns=encoder_type.get_feature_names_out(['primary_category']))

df_final= df_final.join(one_hot_df_type)

df_final.drop('primary_category', axis=1, inplace=True)

df_final.head()
~~~

~~~
df_final.set_index("name", inplace=True)
df_final
~~~

~~~
df_final.columns
df_final.dropna(inplace=True)
~~~

~~~
X = df_final.drop(columns=['Price'])
X = X.drop(columns = X.columns[:2])
y = df_final['Price']
~~~

~~~
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
~~~

~~~
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

reg = LinearRegression()
regmodel = reg.fit(X_train, y_train)
~~~

~~~
reg.coef_

yhat_train = reg.predict(X_train)
yhat_test = reg.predict(X_test)

display(min(yhat_train))
display(max(yhat_train))
~~~

~~~
sns.scatterplot(x = list(range(0,len(regmodel.coef_))), y = regmodel.coef_)

x = list(range(0,len(regmodel.coef_)))

plt.plot(x, regmodel.coef_, color = 'm')
~~~

~~~
print('\nMean squared error: %.2f' % mean_squared_error(y_train, yhat_train))
print('\nMean squared error: %.2f' % mean_squared_error(y_test, yhat_test))
~~~

~~~
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import operator as operater

print(X.shape)
print(y.shape)

# Extract the first column of X for scattering
X_scatter = X.iloc[:, 0]  # Assuming you want to plot against the first feature

plt.scatter(X_scatter, y, s=10)

sort_axis = operater.itemgetter(0)
# Use X_scatter for sorting as well
sorted_zip = sorted(zip(X_scatter, y_train), key=sort_axis)
X_train, y_train = zip(*sorted_zip)
plt.plot(X_train, y_train, color='m')
plt.show()
~~~

~~~
import matplotlib.pyplot as plt
import operator as operater

print(X.shape)
print(y.shape)
plt.scatter(X_scatter, y, s=10)

sort_axis = operater.itemgetter(0)
sorted_zip = sorted(zip(X, y_train), key=sort_axis)
X_train, y_train = zip(*sorted_zip)
plt.plot(X_train, y_train, color='m')
plt.show
~~~

Model using Rating:
~~~
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
~~~

~~~
data.isnull().sum()
data.dropna(subset=["rating"],inplace=True)
data.isnull().sum()
~~~

~~~
X = data[["brand_id", "loves_count","out_of_stock", "new", "online_only","limited_edition"]]

y = data["rating"]
X
~~~

~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42 )
~~~

~~~
reg = LinearRegression()
regmodel = reg.fit(X_train, y_train)

yhat_train = reg.predict(X_train)
yhat_test = reg.predict(X_test)
~~~

~~~
mse = mean_squared_error(y_test, yhat_test)
print("MSE test:",mse)
mse_train = mean_squared_error(y_train, yhat_train)
print("MSE train:",mse_train)
~~~

~~~
sns.heatmap(data[["brand_id", "loves_count","out_of_stock", "new", "online_only","rating"]].corr(), annot = True, vmin=-1, vmax=1, center=0)
~~~

~~~
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(x = list(range(0,len(regmodel.coef_))), y = regmodel.coef_)
x = list(range(0,len(regmodel.coef_)))
plt.plot(x, regmodel.coef_, color = 'm')
~~~

~~~
plt.figure(figsize=(10, 6))
plt.scatter(y_test, yhat_test, alpha=0.7)
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.show()
~~~

~~~
from sklearn.preprocessing import PolynomialFeatures
polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(X)
print('Polynomial Features: [1,x,x**2]')
print(x_poly[0])
~~~

~~~
x_poly_train = x_poly[:-20]
y_train = y[:-20]

x_poly_test = x_poly[-20:]
y_test = y[-20:]
~~~

~~~
model = LinearRegression()
model.fit(x_poly_train, y_train)
yhat_train_pred = model.predict(x_poly_train)
yhat_test_pred = model.predict(x_poly_test)

print("Model weights: ")
print(model.coef_)
~~~


### Model 3: Logistic Regression (Acne Product Recommender) <a name="model3"></a>
**Please see [this notebook](https://colab.research.google.com/drive/1UZuKvfh_-BFqsnquGwFd7OzI7wOTJZH0#scrollTo=rRzWQjMi3oKr) to see all of the redone Preprocessing and Exploration methods, as it differs slightly from the previous models, not included here to prevent the ReadME from being too long.**

~~~
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
~~~

~~~
product_names = acne_df['Name']
X = acne_df[['Ingredients']]
y = acne_df['Acne Fighting']
~~~

~~~
from sklearn.preprocessing import MultiLabelBinarizer

def str_to_list(x):
   return x.strip().split(',')


X['Ingredients'] = X['Ingredients'].dropna().apply(str_to_list)
~~~

~~~
mlb = MultiLabelBinarizer()

one_hot_encoded = mlb.fit_transform(X['Ingredients'])

one_hot_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_)

X = one_hot_df
X.head()
~~~

~~~
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
~~~

~~~
model = LogisticRegression()
model.fit(X_train, y_train)
~~~

~~~
yhat_test = model.predict(X_test)
yhat_train = model.predict(X_train)
~~~

Model Accuracy and Confusion Matrix:
~~~
accuracy = accuracy_score(y_test, yhat_test)
conf_matrix = confusion_matrix(y_test, yhat_test)
class_report = classification_report(y_test, yhat_test)

print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)
~~~

~~~
sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=['Non-Acne Fighting', 'Acne Fighting'], yticklabels=['Non-Acne Fighting', 'Acne Fighting'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for LogReg Model')
plt.show()
~~~

~~~
sns.scatterplot(x = list(range(0,len(model.coef_[0]))),y = model.coef_[0])
~~~

~~~
coefs = model.coef_[0]
important_coefs = [i for i in range(len(coefs)) if coefs[i] > 2]
important_ingredients = mlb.classes_[important_coefs]
print([(i, j) for i, j in zip(important_ingredients, coefs[important_coefs])])
~~~

### Model 4: Neural Network (Acne Product Recommender) <a name="model4"></a>
**Please see [this notebook](https://colab.research.google.com/drive/1UZuKvfh_-BFqsnquGwFd7OzI7wOTJZH0#scrollTo=rRzWQjMi3oKr) to see all of the redone Preprocessing and Exploration methods, as it differs slightly from the previous models, not included here to prevent the ReadME from being too long.**

~~~
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report
~~~

~~~
def create_model():
    model = Sequential()
    model.add(Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=32, activation='relu'))
    model.add(Dense(units=8, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
~~~

~~~
model_nn = create_model()
history = model_nn.fit(X_train, y_train, epochs=10, batch_size=10)
~~~

~~~
plt.plot(history.history['loss'], label='train_loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
~~~

~~~
y_test_pred = model_nn.predict(X_test)
y_test_pred = (y_test_pred > 0.5).astype(int)
print(classification_report(y_test, y_test_pred))
~~~


## Results Section:  <a name="results"></a>
**Results include input in the screenshot, so that the reader can understand what line actually produced the following results.**

### Results from the Data Exploration Methods: <a name="data2"></a>
Data Exp 1: 

![image](https://github.com/user-attachments/assets/7577bcfa-80e2-4a4c-9695-fa3691e6e12d)


Data Exp 2: 

![image](https://github.com/user-attachments/assets/1dd126fc-e3bf-4b6b-8b88-a0d1a9554103)


Data Exp 3: 

![image](https://github.com/user-attachments/assets/23f60dc0-ca84-4381-b6d6-bb74735902bd)


Data Exp 4: 

![image](https://github.com/user-attachments/assets/ffe7d9dc-0ac8-4bc5-ade1-b7e538b8f531)


Data Exp 5: 

![image](https://github.com/user-attachments/assets/68e8ee5e-f01c-4b46-ab92-6620f41cdb2a)


Data Exp 6: 

![image](https://github.com/user-attachments/assets/f7e2c367-8edc-4051-83ac-43a0a10e1cb4)



### Results from Preprocessing: <a name="preprocessing2"></a>
PP 1:

![image](https://github.com/user-attachments/assets/c214f053-bdea-4ab4-be76-321334894f64)


PP 2:

![image](https://github.com/user-attachments/assets/310e398e-53ff-4fc4-84df-b8a68b590558)


PP 3:

![image](https://github.com/user-attachments/assets/4efd0413-30c4-4b64-9e7b-be8b189c46c3)


PP 4:

![image](https://github.com/user-attachments/assets/b28e3dbc-2784-4989-9e13-0914aa728b82)


PP 5:

![image](https://github.com/user-attachments/assets/7353f59a-5cea-436e-bc0f-7ab1e48aa564)


PP 6:

![image](https://github.com/user-attachments/assets/00513194-c211-4d77-b635-0c72c62985d8)


PP 7:

![image](https://github.com/user-attachments/assets/cb5fe713-88f5-4c03-b96b-79ea5e4ff39d)


PP 8:

![image](https://github.com/user-attachments/assets/965f8287-76ea-4bb7-9630-19db7d2e6fde)


PP 9:

![image](https://github.com/user-attachments/assets/8938745f-9b74-442b-99a0-12228d35a005)


PP 10:

![image](https://github.com/user-attachments/assets/9066d5e7-e22f-4795-b25f-97da1d318d47)


PP 11:

![image](https://github.com/user-attachments/assets/2261bbb6-34cc-48ce-b4a9-f02078a6b815)


PP 12:

![image](https://github.com/user-attachments/assets/7670232d-e205-4287-94a3-86b3d8f9e663)


PP 13:

![image](https://github.com/user-attachments/assets/3e323763-5960-40f3-ab83-d8dbdf1f2fa6)


PP 14:

![image](https://github.com/user-attachments/assets/10ea3bff-d298-4a33-b63f-d68ac83af7cc)


PP 15:

![image](https://github.com/user-attachments/assets/5c61a969-cd19-4b3e-9044-67e04fb64d6d)


PP 16:

![image](https://github.com/user-attachments/assets/1fca022d-1688-4e26-96f9-00d0aaa1110b)


PP 17:

![image](https://github.com/user-attachments/assets/f2bdab60-6297-4076-ac26-56f16929e217)


PP 18:

![image](https://github.com/user-attachments/assets/0a8063b8-904c-4e6b-908f-1a14d680964e)


PP 19:

![image](https://github.com/user-attachments/assets/45d8357b-7e0c-4689-83f8-c9d987b7b1d2)


### Results from Model 1 <a name="model12"></a>
Model1 1:

![image](https://github.com/user-attachments/assets/70d80747-417d-401f-92c3-9e82e9e22828)


Model1 2:

![image](https://github.com/user-attachments/assets/422f1074-e2dd-4302-b732-0de66d6a1778)


Model1 3:

![image](https://github.com/user-attachments/assets/6bdff7ff-0047-47ac-a2ba-0625c26a207a)


Model1 4:

![image](https://github.com/user-attachments/assets/706e3395-9598-4fc9-90ea-d6915fabde1a)


### Results from Model 2 <a name="model22"></a>
#### Results from Linear Regression on Price: 

LinRegPrice 1:

![image](https://github.com/user-attachments/assets/4e0f9ac6-f88d-474a-8676-ec6b32bc2436)


LinRegPrice 2:

![image](https://github.com/user-attachments/assets/35271b08-4939-4273-bbeb-3994d1147acf)


LinRegPrice 3:

![image](https://github.com/user-attachments/assets/f5d2fc20-8cfa-4747-b4b7-da106e21f036)


#### Results from Linear Regression on Rating: 
LinRegRat 1:

![image](https://github.com/user-attachments/assets/e059d0aa-1bce-4a01-ace8-ab7c1a31c070)


LinRegRat 2:

![image](https://github.com/user-attachments/assets/1af4535b-018c-4378-9cbb-6c5ff3f2a40f)


LinRegRat 3:

![image](https://github.com/user-attachments/assets/ab83aa29-578f-4c3b-8184-7228afd1f3d4)


LinRegRat 4:

![image](https://github.com/user-attachments/assets/91309e96-dd49-495a-bf3a-564234271877)


LinRegRat 5:

![image](https://github.com/user-attachments/assets/e11950f8-3a52-40c1-9505-0243f7949396)


LinRegRat 6:

![image](https://github.com/user-attachments/assets/611de3ab-b4e1-4a7f-8f16-754125b29ac1)


LinRegRat 7:

![image](https://github.com/user-attachments/assets/f16a7caa-8209-4ac7-89c7-7ba99c2b4c5a)



### Results from Model 3 <a name="model32"></a>
LogReg 1:

![image](https://github.com/user-attachments/assets/887657be-8ca8-4d3e-a027-be00062a2604)


LogReg 2:

![image](https://github.com/user-attachments/assets/0d1a7d0f-f47d-4a9f-a4a2-677b0c2e1744)


LogReg 3:

![image](https://github.com/user-attachments/assets/2ab4887b-f0e2-456e-b4d4-044561e6db63)


LogReg 4:

![image](https://github.com/user-attachments/assets/04b8011c-d5aa-4de6-9cad-15e5ea7ba366)


LogReg 5:

![image](https://github.com/user-attachments/assets/648a2b4c-8960-427b-be49-25da1eacb074)



### Results from Model 4 <a name="model42"></a>
NN 1:

![image](https://github.com/user-attachments/assets/4c13e0fa-8527-4a46-a5cc-29eb849ceee7)


NN 2:

![image](https://github.com/user-attachments/assets/903f146e-6fad-490e-8294-a0f2072cf089)


NN 3:

![image](https://github.com/user-attachments/assets/d199057e-caa3-40ef-81f4-e23bdaee1afa)


## Discussion Section  <a name="discussion"></a>

### Data Exploration Discussion
In our data exploration, we were simply trying to understand our data better, leading to a lot of viewing the previews of our sephora, indonesia, and skinsort dataframes. We also wanted to see how many prices were listed as 0, and wanted to understand what effects each of the products had by reviewing the “afteruse” column. 

In our exploration, we would have improved by taking a deeper look into our data, and taking the discrepancies more seriously. There were some datasets that had ratings while others didn’t, and we didn’t anticipate how long it would actually take to properly merge these datasets. We also would have improved by trying harder to plot our data, in order to see if the columns are actually that related to one another. 

### Preprocessing Discussion
In our preprocessing step, we started out with a pairplot in order to visualize the data better, starting off with Country. This was not entirely helpful, so we decided to move on to dropping all our null values to clean up our datasets. We then used the describe and info functions to better see what is happening in our data. We then realized that many of the columns were not named in the exact same way, so we changed these names so that merging the datasets later would be much easier. 

We also used a similarity matrix in an attempt to match reviews to certain products based on how similar they are in product name on reviews, and we also wanted to attach all of the reviews to the product. 

Other preprocessing methods included checking the shape of our data, trying to locate where our null values are, and using value_counts() to see how many instances of each entry we have. 

In hindsight, using the similarity matrix to try to “force” products with reviews might not have been the best decision, as in order for our model to work, we had to give a pretty lax threshold for similarity, causing dissimilar products to be matched with each other. It’s possible that we could have put in more effort in researching other methods to merge our datasets together. 

### Model 1 Discussion


### Model 2 Discussion


### Model 3 Discussion


### Model 4 Discussion



## Conclusion <a name="conclusion"></a>





## Statement of Collaboration  <a name="collab"></a>
Overall, all members contributed to the project, as the project was mostly worked on with multiple people working together at a time. We all joined several zoom meetings together, or met in-person as a team to work on the project. We would share our screen and take turns programming different sections, often with feedback or edits from the others. We brainstormed all the ideas for the project together, bouncing ideas off of and changing ideas based on everybody's feedback. We also took turns going to office hours to discuss with TAs, and sometimes all of us attended together. For the more specific tasks, they are listed below alphabetically by last name: 

- Ilia Aballa: Uploaded early datasets, helped organize some meetings/reserve study room, attempted to plot graphs for Milestone 3, completed writeup 

- Kiran Keertipati: Organized many meetings, met with Brian one on one to discuss the project, main screen-sharer for group coding sessions

- Leena Khattat: Helped with encoding implementation, implemented LogReg model for Milestone 4

- Nicole Li: Created Github Repo, helped write the similarity matrix 

- Wang Liu: Developed the similarity-based dataset merging process, contributed to the acne prediction task.

- Evelyn Mares-moreno: Talked to tutors, wrote code for Milestone 4 and updated group members about these changes 

**All:**  Helped clean/organize/understand data, participated in group coding sessions and brainstorming, took turns submitting Milestones on Gradescope, updated the README in some way over the four milestones

-----------------------------------------------------------------------------------------------------------------------------------
## README Prior to Final Submission Writeup: ##

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

## Group Members, listed alphabetically by last name
- Ilia Aballa
- Kiran Keertipati 
- Leena Khattat
- Nicole Li 
- Wang Liu
- Evelyn Mares-moreno

