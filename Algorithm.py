import pandas as pd
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate
from surprise.prediction_algorithms import SVD, KNNBasic, NMF, SlopeOne

# Load the data
data = pd.read_csv('/Users/tomiwa/Documents/Portfolio/Product Recommendation Engine/Amazon.csv')

# Assuming 'df' is your DataFrame containing the dataset
discount_price_stats = data['discounted_price'].describe()
actual_price_stats = data['actual_price'].describe()

print("Discount Price Statistics:")
print(discount_price_stats)

print("\nActual Price Statistics:")
print(actual_price_stats)

# Data Preprocessing: Convert 'discounted_price' column to numeric without currency symbol and commas
data['discounted_price'] = data['discounted_price'].str.replace('₹', '').str.replace(',', '').astype(float)

# Data Preprocessing: Convert 'actual_price' column to numeric without currency symbol and commas
data['actual_price'] = data['actual_price'].str.replace('₹', '').str.replace(',', '').astype(float)

most_frequent_rating = data['rating'].mode()[0]
data['rating'] = data['rating'].replace('|', most_frequent_rating)
data['rating'] = data['rating'].astype(float)
# Collaborative Filtering
collab_data = data[['user_id', 'product_id', 'rating']]

# Create the Surprise Reader and Dataset
reader = Reader(rating_scale=(1, 5))
train_dataset = Dataset.load_from_df(collab_data, reader)

# List of collaborative filtering algorithms to test
algorithms = [
    SVD(),
    KNNBasic(sim_options={'user_based': True}),
    KNNBasic(sim_options={'user_based': False}),
    NMF(),
    SlopeOne()
]

# Test and evaluate each algorithm using cross-validation
for algorithm in algorithms:
    results = cross_validate(algorithm, train_dataset, measures=['RMSE'], cv=5, verbose=True)
    print(f"Algorithm: {algorithm.__class__.__name__}")
    print(f"Mean RMSE: {results['test_rmse'].mean()}")
    print("---------------------------")
    
    
    # Deciding the Algorithm 
   # Based on the RMSE scores obtained from evaluating different collaborative filtering algorithms, here are the results:

#1. SVD: Mean RMSE - 0.2853
#2. KNNBasic (User-Based): Mean RMSE - 0.2826
#3. KNNBasic (Item-Based): Mean RMSE - 0.2829
#4. NMF: Mean RMSE - 0.2968
#5. SlopeOne: Mean RMSE - 0.2826

#Among these algorithms, the one with the lowest mean RMSE (which indicates better accuracy)
#is KNNBasic with Item-Based collaborative filtering. Therefore, the KNNBasic (Item-Based) algorithm seems 
#to be the best choice for this particular recommendation system based on the evaluation results. However, 
#it's essential to keep in mind that the choice of the algorithm may vary depending on the specific dataset 
#and use case. You might want to experiment with other algorithms and hyperparameter tuning to ensure the best 
#results for your specific application.