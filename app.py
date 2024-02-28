from flask import Flask, render_template, request
from joblib import load
from fuzzywuzzy import process
from googlesearch import search
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from flask import Flask, render_template, request
import pandas as pd
from fuzzywuzzy import process
from googlesearch import search
import nltk
nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

df = pd.read_csv("Food_Recipe.csv")
df = df[['name', 'description', 'cuisine', 'ingredients_name', 'instructions', 'image_url']]

def preprocess_text(text):
    # Add additional text preprocessing steps if needed
    return text

# Preprocessing pipeline
text_preprocessor = FunctionTransformer(func=preprocess_text)

# Combine text features into a single feature
df['text_data'] = df['description'] + ' ' + df['ingredients_name'] + ' ' + df['instructions']

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'))

# Create a pipeline for text data
text_pipeline = Pipeline([
    ('preprocessor', text_preprocessor),
    ('tfidf_vectorizer', tfidf_vectorizer),
])

# Fill NaN values with an empty string
df['text_data'] = df['text_data'].fillna('')

# Fit the model
model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
model.fit(text_pipeline.fit_transform(df['text_data']))

def get_google_results(query, num_results=3):
    results = []
    for j in search(query, num_results=num_results):
        results.append(j)
    return results

def get_cooking_instructions_and_suggestions_with_google(dish_name):
    dish_name_lower = dish_name.lower()

    matches = process.extractOne(dish_name_lower, df['name'].str.lower())

    if matches and matches[1] >= 70:
        best_match = matches[0]
        dish_index = df[df['name'].str.lower() == best_match].index[0]

        distances, indices = model.kneighbors(text_pipeline.transform([df.iloc[dish_index]['instructions']]))

        cooking_instructions = f"Cooking Instructions for {df.iloc[dish_index]['name']}:\n{df.iloc[dish_index]['instructions']}"

        suggestions_cuisine = []
        for index in indices.flatten()[1:]:
            suggestions_cuisine.append(df.iloc[index]['name'])

        suggestions_diet = []
        cuisine_query = df.iloc[dish_index]['cuisine']
        google_results = get_google_results(f"{cuisine_query} recipes of low calories", num_results=3)
        for result in google_results:
            suggestions_diet.append(result)

        return {
            'cooking_instructions': cooking_instructions,
            'suggestions_cuisine': suggestions_cuisine,
            'suggestions_diet': suggestions_diet,
            'recipe_url': df.iloc[dish_index]['image_url']
        }

    else:
        return {'message': "No close matches found."}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        dish_name_input = request.form['dish_name']
        result = get_cooking_instructions_and_suggestions_with_google(dish_name_input)

        if 'suggestions_cuisine' not in result:
            result['suggestions_cuisine'] = []
        if 'suggestions_diet' not in result:
            result['suggestions_diet'] = []

        suggestions_cuisine = result['suggestions_cuisine']
        suggestions_diet = result['suggestions_diet']

        return render_template('result.html', result=result,
                               suggestions_cuisine=suggestions_cuisine,
                               suggestions_diet=suggestions_diet)

if __name__ == '__main__':
    app.run(debug=True)
