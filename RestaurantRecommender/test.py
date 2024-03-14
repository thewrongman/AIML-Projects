import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
st.set_page_config(page_title="YaYu Retreat's", layout = 'wide', page_icon = 'logo1.png', initial_sidebar_state = 'auto')

data = pd.read_csv('BangaloreZomatoData.csv')

data.drop(data[['Timing', 'URL', 'PeopleKnownFor', 'IsHomeDelivery', 'isTakeaway', 'isIndoorSeating', 'isVegOnly',
                'Dinner Reviews', 'Delivery Ratings', 'Delivery Reviews', 'KnownFor', 'PopularDishes']], axis=1,
          inplace=True)
data['Area'] = data['Area'].str.replace(', Bangalore', '')
data.replace('-', np.nan, inplace=True)
numeric_features = ['Dinner Ratings', 'AverageCost']
categorical_features = ['Area', 'Cuisines', 'Full_Address']
data[categorical_features] = data[categorical_features].replace('-', np.nan)

data[categorical_features] = data[categorical_features].fillna('Unknown')
for feature in numeric_features:
    data[feature] = data[feature].apply(lambda x: float(x) if pd.notnull(x) else None)
    data[feature].fillna(data[feature].mean(), inplace=True)

encoder = OneHotEncoder(drop='first', sparse=False)
encoded_categorical_features = encoder.fit_transform(data[categorical_features])
all_features = np.hstack((encoded_categorical_features, data[numeric_features].values))
scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)

# Mdoel or KMeans clustering
optimal_clusters = 4  
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=1000, n_init=20, random_state=0)
data['cluster'] = kmeans.fit_predict(all_features)

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Cuisines'].fillna(''))

# Streamlit UI
st.title("🍽️ Bangalore Restaurant Recommender 🌟")
st.sidebar.header("User Preferences")

# User input for area and preference
user_area = st.sidebar.text_input("Enter the desired area:", "").strip().title()

user_input = st.sidebar.text_input("Describe your preference:", "").strip().title()

if user_area and user_input:
    
    user_vector = tfidf_vectorizer.transform([user_input])
    recommendation_scores = cosine_similarity(user_vector, tfidf_matrix)
    restaurant_indices_in_area = data[data['Area'] == user_area].index

    recommendations = [(idx, recommendation_scores[0][idx]) for idx in restaurant_indices_in_area]
    recommendations.sort(key=lambda x: x[1], reverse=True)

    st.subheader(f"🌟 Top 3 Recommended Restaurants in {user_area} 🌟")
    for idx, _ in recommendations[:3]:
        restaurant = data.iloc[idx]
        st.write(f"--------------------------{restaurant['Name']}------------------------")
        st.write(f"Average Cost: ₹{restaurant['AverageCost']:.2f}")
        st.write(f"Rating: ⭐{restaurant['Dinner Ratings']:.2f}")
        st.write(f"Cuisine: 🍲{restaurant['Cuisines']}")
        st.write(f"Full Address: 📍{restaurant['Full_Address']}\n")
        st.write(f"----------------------------------------------------------------------")

    # Visualization: Cluster Distribution
    # st.sidebar.subheader("Cluster Distribution")
    # fig = px.histogram(data, x="cluster", title="Restaurant Clusters Distribution",
    #                    labels={"cluster": "Cluster Number", "count": "Number of Restaurants"})
    # st.sidebar.plotly_chart(fig)


