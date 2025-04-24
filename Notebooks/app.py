import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, ndcg_score
from scipy.sparse import hstack
from xgboost import XGBRanker
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
df = pd.read_csv('merged_df.xls')

# Split category_code into main_category and subcategory
df[['main_category', 'subcategory']] = df['category_code'].str.split('.', n=1, expand=True)

# Scale the price column
scaler = joblib.load('scaler.pkl')
df['price_scaled'] = scaler.transform(df[['price']])

# Map interaction scores
interaction_map = {'view': 1, 'cart': 2, 'purchase': 3}
df['interaction_score'] = df['event_type'].map(interaction_map)

# Combine categorical features into a single text column
df['combined_text'] = df['event_type'] + ' ' + df['category_code'].fillna('') + ' ' + \
                      df['brand'].fillna('') + ' ' + df['main_category'].fillna('') + ' ' + \
                      df['subcategory'].fillna('')

# Drop unnecessary columns
df.drop(columns=['price', 'category_id'], inplace=True)

# Load the saved TF-IDF vectorizer and transform the combined_text
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = tfidf_vectorizer.transform(df['combined_text'])

# Combine TF-IDF matrix with price_scaled
price_features = df['price_scaled'].values.reshape(-1, 1)
combined_features = hstack([tfidf_matrix, price_features])

# Compute user-product stats
user_product_stats = df.groupby(['user_id', 'product_id']).agg(
    total_interactions=('interaction_score', 'count'),
    avg_interaction_score=('interaction_score', 'mean'),
    last_interaction_score=('interaction_score', 'last')
).reset_index()
df = df.merge(user_product_stats, on=['user_id', 'product_id'], how='left')

# Load the LabelEncoder and encode categorical variables
label_enc = joblib.load('label_enc.pkl')
df['brand'] = label_enc.fit_transform(df['brand'])
df['category_code'] = label_enc.fit_transform(df['category_code'])

# Train-test split
# Changed 'price_scaled' to 'price' to match the model's expected feature name
features = ['category_code', 'brand', 'price', 'total_interactions', 'avg_interaction_score', 'last_interaction_score']
# Rename the column in df to match the expected feature name
df.rename(columns={'price_scaled': 'price'}, inplace=True)
X = df[['user_id', 'product_id'] + features]
y = df['interaction_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split data for TF-IDF and reset indices
df_train = df.loc[X_train.index].copy().reset_index(drop=True)
df_test = df.loc[X_test.index].copy().reset_index(drop=True)

# Compute TF-IDF features for train and test
tfidf_train = tfidf_vectorizer.transform(df_train['combined_text'])
tfidf_test = tfidf_vectorizer.transform(df_test['combined_text'])

# Combine with price_scaled for train and test (now using 'price')
price_train_scaled = df_train['price'].values.reshape(-1, 1)
price_test_scaled = df_test['price'].values.reshape(-1, 1)
combined_features_train = hstack([tfidf_train, price_train_scaled])
combined_features_test = hstack([tfidf_test, price_test_scaled])

# Compute user profiles for TF-IDF (using both training and test data)
user_interactions_train = df_train[df_train['event_type'].isin(['view', 'purchase'])]
user_interactions_test = df_test[df_test['event_type'].isin(['view', 'purchase'])]
user_profiles = {}

# First, process training data
for user_id, group in user_interactions_train.groupby('user_id'):
    product_indices = group.index
    aggregated_tfidf = np.mean(tfidf_train[product_indices].toarray(), axis=0)
    aggregated_price = np.mean(group['price'])
    user_profile = np.hstack([aggregated_tfidf, aggregated_price])
    user_profiles[user_id] = user_profile

# Then, process test data and update profiles if user exists in both
for user_id, group in user_interactions_test.groupby('user_id'):
    product_indices = group.index
    aggregated_tfidf = np.mean(tfidf_test[product_indices].toarray(), axis=0)
    aggregated_price = np.mean(group['price'])
    user_profile = np.hstack([aggregated_tfidf, aggregated_price])
    if user_id in user_profiles:
        # Combine profiles by averaging if user exists in both sets
        existing_profile = user_profiles[user_id]
        user_profiles[user_id] = (existing_profile + user_profile) / 2
    else:
        user_profiles[user_id] = user_profile

# Load the saved XGBRanker and Random Forest models
best_model = joblib.load('xgb_model.pkl')
rf_model = joblib.load('rf_model.pkl')

# Define evaluation functions
def recommend_and_evaluate_tfidf(user_id, user_profiles, k=5):
    """
    Recommend and evaluate top K products for a user using TF-IDF similarity.
    Uses both train and test data combined for recommendations.
    
    Args:
        user_id (int): ID of the user.
        user_profiles (dict): Precomputed user profiles (TF-IDF + price_scaled).
        k (int): Number of recommendations.
    
    Returns:
        dict: Recommendations and evaluation metrics.
    """
    if user_id not in user_profiles:
        return {"error": f"No profile found for user {user_id}"}
    
    # Get the user's profile vector
    user_profile_vector = user_profiles[user_id]
    
    # Compute similarity between user's profile and all product vectors (using combined train and test data)
    product_similarity_scores = cosine_similarity([user_profile_vector], combined_features.toarray())[0]
    
    # Get indices of top K similar products
    user_indices = df[df['user_id'] == user_id].index.values
    top_indices = [idx for idx in np.argsort(product_similarity_scores)[::-1] if idx not in user_indices][:k]
    
    # Get recommended products
    recommendations = df.iloc[top_indices][['product_id', 'brand', 'subcategory']].copy()
    recommendations['similarity_score'] = product_similarity_scores[top_indices]
    
    # Ground truth interaction scores for recommended products
    user_data = df[df['user_id'] == user_id]
    true_scores = [user_data[user_data['product_id'] == pid]['interaction_score'].iloc[0] 
                   if pid in user_data['product_id'].values else 0 for pid in recommendations['product_id']]
    
    # Binary relevance (relevant if interaction_score >= 2, i.e., cart or purchase)
    binary_true = np.array([1 if score >= 2 else 0 for score in true_scores])
    
    # Compute metrics
    precision = precision_score(binary_true, binary_true * 0 + 1, zero_division=0)
    all_true_scores = user_data['interaction_score'].values
    all_pred_scores = product_similarity_scores[user_indices]
    ndcg = ndcg_score([all_true_scores], [all_pred_scores], k=k) if len(all_true_scores) > 1 else 0
    
    return {
        "recommendations": recommendations,
        "precision@k": precision,
        "ndcg@k": ndcg,
        "true_scores": true_scores,
        "predicted_scores": recommendations['similarity_score'].values
    }

def recommend_and_evaluate_xgboost(user_id, X_train, X_test, y_train, y_test, model, product_info, label_encoder, k=5):
    # Check test set first
    user_data_test = X_test[X_test['user_id'] == user_id].copy()
    if not user_data_test.empty:
        user_data = user_data_test
        y_data = y_test
    else:
        # Fall back to training set if not found in test set
        user_data_train = X_train[X_train['user_id'] == user_id].copy()
        if user_data_train.empty:
            return {"error": f"No data available for user {user_id}"}
        user_data = user_data_train
        y_data = y_train
    
    user_data['interaction_score'] = y_data.loc[user_data.index]
    features = user_data.drop(columns=['user_id', 'product_id', 'interaction_score'])
    user_data['predicted_score'] = model.predict(features)
    
    true_scores = user_data['interaction_score'].values
    recommendations = user_data.sort_values(by='predicted_score', ascending=False)
    top_k = recommendations[['product_id', 'predicted_score']].head(k)
    
    unique_product_info = product_info[['product_id', 'category_code']].drop_duplicates(subset='product_id')
    top_k['product_name'] = top_k['product_id'].map(unique_product_info.set_index('product_id')['category_code']).apply(lambda x: label_encoder.inverse_transform([int(x)])[0])
    
    top_k_true_scores = recommendations['interaction_score'].head(k).values
    binary_true = np.array([1 if score >= 2 else 0 for score in top_k_true_scores])
    
    precision = precision_score(binary_true, binary_true * 0 + 1, zero_division=0)
    ndcg = ndcg_score([true_scores], [user_data['predicted_score'].values], k=k) if len(true_scores) > 1 else 0
    
    return {
        "recommendations": top_k,
        "precision@k": precision,
        "ndcg@k": ndcg
    }

def recommend_and_evaluate_rf(user_id, X_train, X_test, y_train, y_test, model, product_info, label_encoder, k=5):
    # Check test set first
    user_data_test = X_test[X_test['user_id'] == user_id].copy()
    if not user_data_test.empty:
        user_data = user_data_test
        y_data = y_test
    else:
        # Fall back to training set if not found in test set
        user_data_train = X_train[X_train['user_id'] == user_id].copy()
        if user_data_train.empty:
            return {"error": f"No data available for user {user_id}"}
        user_data = user_data_train
        y_data = y_train
    
    user_data['interaction_score'] = y_data.loc[user_data.index]
    # Ensure the features match the model's expected input
    features = ['category_code', 'brand', 'price', 'total_interactions', 'avg_interaction_score', 'last_interaction_score']
    user_features = user_data[features]
    user_data['predicted_score'] = model.predict(user_features)
    
    true_scores = user_data['interaction_score'].values
    recommendations = user_data.sort_values(by='predicted_score', ascending=False)
    top_k = recommendations[['product_id', 'predicted_score']].head(k)
    
    unique_product_info = product_info[['product_id', 'category_code']].drop_duplicates(subset='product_id')
    top_k['product_name'] = top_k['product_id'].map(unique_product_info.set_index('product_id')['category_code']).apply(lambda x: label_encoder.inverse_transform([int(x)])[0])
    
    top_k_true_scores = recommendations['interaction_score'].head(k).values
    binary_true = np.array([1 if score >= 2 else 0 for score in top_k_true_scores])
    
    precision = precision_score(binary_true, binary_true * 0 + 1, zero_division=0)
    ndcg = ndcg_score([true_scores], [user_data['predicted_score'].values], k=k) if len(true_scores) > 1 else 0
    
    return {
        "recommendations": top_k,
        "precision@k": precision,
        "ndcg@k": ndcg
    }

# Streamlit app
st.title("Personalized Product Recommendation Dashboard")

st.write("Enter your user ID to get personalized product recommendations from three models: TF-IDF, XGBRanker, and Random Forest.")

# Input user ID
user_id = st.number_input("User ID", min_value=0, step=1, value=516207684)

if st.button("Get Recommendations"):
    # TF-IDF Recommendations
    st.subheader("TF-IDF Recommendations")
    tfidf_result = recommend_and_evaluate_tfidf(user_id, user_profiles, k=5)
    if "error" not in tfidf_result:
        st.write("**Recommendations:**")
        st.dataframe(tfidf_result["recommendations"])
        st.write(f"**Precision@5:** {tfidf_result['precision@k']:.3f}")
        st.write(f"**NDCG@5:** {tfidf_result['ndcg@k']:.3f}")
    else:
        st.write(tfidf_result["error"])

    # XGBRanker Recommendations
    st.subheader("XGBRanker Recommendations")
    xgb_result = recommend_and_evaluate_xgboost(user_id, X_train, X_test, y_train, y_test, best_model, df, label_enc, k=5)
    if "error" not in xgb_result:
        st.write("**Recommendations:**")
        st.dataframe(xgb_result["recommendations"])
        st.write(f"**Precision@5:** {xgb_result['precision@k']:.3f}")
        st.write(f"**NDCG@5:** {xgb_result['ndcg@k']:.3f}")
    else:
        st.write(xgb_result["error"])

    # Random Forest Recommendations
    st.subheader("Random Forest Recommendations")
    rf_result = recommend_and_evaluate_rf(user_id, X_train, X_test, y_train, y_test, rf_model, df, label_enc, k=5)
    if "error" not in rf_result:
        st.write("**Recommendations:**")
        st.dataframe(rf_result["recommendations"])
        st.write(f"**Precision@5:** {rf_result['precision@k']:.3f}")
        st.write(f"**NDCG@5:** {rf_result['ndcg@k']:.3f}")
    else:
        st.write(rf_result["error"])