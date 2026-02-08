# Intelligent Recommendation System Powered by AI & ML
# Movie Recommendation System (Content-Based Filtering)

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Step 1: Create Movie Dataset
# -----------------------------
movies = {
    "Title": [
        "The Matrix",
        "John Wick",
        "Inception",
        "Interstellar",
        "The Notebook",
        "Titanic",
        "Avengers"
    ],
    "Genre": [
        "Action Sci-Fi",
        "Action Thriller",
        "Sci-Fi Thriller",
        "Sci-Fi Drama",
        "Romance Drama",
        "Romance Drama",
        "Action Sci-Fi"
    ]
}

df = pd.DataFrame(movies)

print("Movie Dataset:\n")
print(df)

# ---------------------------------
# Step 2: Convert Text to Vectors
# ---------------------------------
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(df["Genre"])

# ---------------------------------
# Step 3: Calculate Similarity
# ---------------------------------
similarity = cosine_similarity(genre_matrix)

# ---------------------------------
# Step 4: Recommendation Function
# ---------------------------------
def recommend_movie(movie_name, top_n=3):
    if movie_name not in df["Title"].values:
        print("\nMovie not found! Please check the name.")
        return

    index = df[df["Title"] == movie_name].index[0]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(
        similarity_scores, key=lambda x: x[1], reverse=True
    )

    print(f"\nRecommended movies for '{movie_name}':\n")
    count = 0
    for i in similarity_scores:
        if df.iloc[i[0]]["Title"] != movie_name:
            print(df.iloc[i[0]]["Title"])
            count += 1
        if count == top_n:
            break

# ---------------------------------
# Step 5: Test the Rec
# ---------------------------------
recommend_movie("Inception")
