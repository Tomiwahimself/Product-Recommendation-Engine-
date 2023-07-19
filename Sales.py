import pandas as pd
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from surprise import SVD 
from surprise import Reader, Dataset, KNNBasic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the data
data = pd.read_csv('/Users/tomiwa/Documents/Portfolio/Product Recommendation Engine/Amazon.csv')

# Data Preprocessing
data['discounted_price'] = data['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)
data['actual_price'] = data['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)

# Replace '|' in the 'rating' column with the most frequent rating
most_frequent_rating = data['rating'].mode()[0]
data['rating'] = data['rating'].replace('|', most_frequent_rating)
data['rating'] = data['rating'].astype(float)
# EDA Process

# Distribution of Ratings
def visualize_data ():
    plt.figure(figsize=(8, 5))
    sns.countplot(x='rating', data=data)
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.show()

        # Number of Ratings per User
    user_ratings_counts = data['user_id'].value_counts()
    plt.figure(figsize=(10, 5))
    sns.histplot(user_ratings_counts, bins=50)
    plt.title('Number of Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Count')
    plt.show()

        # Number of Ratings per Product
    product_ratings_counts = data['product_id'].value_counts()
    plt.figure(figsize=(10, 5))
    sns.histplot(product_ratings_counts, bins=50)
    plt.title('Number of Ratings per Product')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Count')
    plt.show() 
    # Visualize Price Data
    price_range_labels = ['0-5000', '5000-10000', '10000-15000', '15000-20000', '20000-25000', '25000-30000', '30000-35000', '35000-40000', '40000-45000', '45000-50000']
    price_range_bins = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000]
    data['discounted_price_range'] = pd.cut(data['discounted_price'], bins=price_range_bins, labels=price_range_labels)
    data['actual_price_range'] = pd.cut(data['actual_price'], bins=price_range_bins, labels=price_range_labels)

    # Visualize Grouped Price Data
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x='discounted_price_range', bins=len(price_range_labels), label='Discounted Price', alpha=0.7)
    sns.histplot(data=data, x='actual_price_range', bins=len(price_range_labels), label='Actual Price', alpha=0.7)
    plt.xlabel('Price Range (₹)')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
    plt.legend()
    plt.show()
# Collaborative Filtering
collab_data = data[['user_id', 'product_id', 'rating']]
train_data, test_data = train_test_split(collab_data, test_size=0.2, random_state=42)
reader = Reader(rating_scale=(1, 5))
train_dataset = Dataset.load_from_df(train_data, reader)
collab_algo = SVD()
collab_algo.fit(train_dataset.build_full_trainset())
# Build the KNNBasic algorithm with Item-Based Collaborative Filtering
knn_algo = KNNBasic(sim_options={'user_based': False})
knn_algo.fit(train_dataset.build_full_trainset())

# Content-Based Filtering
content_data = data[['product_id', 'product_name', 'category', 'about_product']]
content_data['about_product'] = content_data['about_product'].fillna('')
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(content_data['about_product'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def content_based_recommendations(product_id, cosine_sim=cosine_sim):
    if product_id not in content_data['product_id'].values:
        print("Product ID not found in the dataset.")
        return None
    
    # Create a copy of the 'content_data' DataFrame to avoid the SettingWithCopyWarning
    content_data_copy = content_data.copy()

    # Use .loc to assign values without raising the SettingWithCopyWarning
    content_data.loc[:, 'about_product'] = content_data['about_product'].fillna('')

    idx = content_data_copy[content_data_copy['product_id'] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10 similar products
    product_indices = [i[0] for i in sim_scores]
    return content_data_copy.loc[product_indices, 'product_id']


# Function to get hybrid recommendations
def hybrid_recommendations(user_id, product_id):
    # Get collaborative filtering recommendations using KNNBasic (Item-Based)
    collab_rec = [pred.iid for pred in knn_algo.test([[user_id, product_id, 4]])]

    # Get content-based filtering recommendations
    content_rec = content_based_recommendations(product_id)

    # Combine the results and remove duplicates
    recommendations = collab_rec + content_rec
    recommendations = list(set(recommendations))

    # Remove the input product_id from recommendations if it exists
    recommendations = [rec for rec in recommendations if rec != product_id]

    return recommendations

if __name__ == '__main__':
    # Initial model training and content-based similarity matrix creation
    collab_data = data[['user_id', 'product_id', 'rating']]
    train_data, _ = train_test_split(collab_data, test_size=0.2, random_state=42)
    reader = Reader(rating_scale=(1, 5))
    train_dataset = Dataset.load_from_df(train_data, reader)
    collab_algo = SVD()
    collab_algo.fit(train_dataset.build_full_trainset())

    content_data = data[['product_id', 'product_name', 'category', 'about_product']]
    content_data['about_product'] = content_data['about_product'].fillna('')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(content_data['about_product'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Example usage: Get hybrid recommendations for a specific user and product
    user_id = 'AG3D6O4STAQKAY2UVGEUV46KN35Q,AHMY5CWJMMK5BJRBBSNLYT3ONILA,AHCTC6ULH4XB6YHDY6PCH2R772LQ,AGYHHIERNXKA6P5T7CZLXKVPT7IQ,AG4OGOFWXJZTQ2HKYIOCOY3KXF2Q,AENGU523SXMOS7JPDTW52PNNVWGQ,AEQJHCVTNINBS4FKTBGQRQTGTE5Q,AFC3FFC5PKFF5PMA52S3VCHOZ5FQ'  # Replace 'your_user_id_here' with an actual user ID
    product_id = 'B082LSVT4B'  # Replace 'your_product_id_here' with an actual product ID
    recommendations = hybrid_recommendations(user_id, product_id)
    if recommendations is not None:
        print("Hybrid Recommender Recommendations:")
        print(recommendations)

