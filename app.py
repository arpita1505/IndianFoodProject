import streamlit as st
import pandas as pd
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Indian Food AI", layout="centered")

# =========================
# LIGHT MODE + BLUE HEADING
# =========================
st.markdown("""
<style>
body {
    background-color: white;
    color: black;
}
h1 {
    color: #1f77b4 !important;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("IndianFoodDatasetCSV.csv")
df.columns = df.columns.str.strip()

df['TranslatedIngredients'] = df['TranslatedIngredients'].fillna('').str.lower()
df['RecipeName'] = df['RecipeName'].fillna('').str.lower()

# =========================
# MODEL
# =========================
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['TranslatedIngredients'])
y = df['Diet']

model = LinearSVC()
model.fit(X, y)

# =========================
# FUNCTIONS
# =========================
def recommend_food(user_input):
    user_vec = tfidf.transform([user_input.lower()])
    similarity = cosine_similarity(user_vec, X)[0]

    top_indices = similarity.argsort()[-5:][::-1]

    results = df.iloc[top_indices][['RecipeName', 'Cuisine', 'Diet']].copy()
    results['Match %'] = [round(similarity[i] * 100, 2) for i in top_indices]

    return results.sort_values(by="Match %", ascending=False)

def analyze_food_quality(user_input):
    user_input = user_input.lower()
    tags = []

    if any(x in user_input for x in ["paneer", "chicken", "egg", "dal"]):
        tags.append("💪 High Protein")
    if any(x in user_input for x in ["rice", "potato", "bread"]):
        tags.append("⚡ High Carbs")
    if any(x in user_input for x in ["butter", "ghee", "cream"]):
        tags.append("⚠️ High Fat")
    if any(x in user_input for x in ["vegetable", "spinach", "salad"]):
        tags.append("🟢 Healthy")

    if not tags:
        tags.append("⚖️ Balanced")

    return tags

# =========================
# HEADER
# =========================
st.markdown("<h1>🍴 Indian Food Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Step-by-Step Smart Food Analysis</p>", unsafe_allow_html=True)

st.markdown("---")

# =========================
# INPUT
# =========================
user_input = st.text_input("Enter Ingredients", "paneer, butter, tomato")

analyze = st.button("Analyze Food")

# =========================
# TIMELINE OUTPUT
# =========================
if analyze and user_input:

    # STEP 1
    st.subheader("Step 1: Processing Input")
    with st.spinner("Analyzing ingredients..."):
        time.sleep(1.5)

    # STEP 2
    vec = tfidf.transform([user_input.lower()])
    prediction = model.predict(vec)[0]

    st.subheader("Step 2: Predicted Diet")
    st.success(prediction)
    time.sleep(1.2)

    # STEP 3
    recs = recommend_food(user_input)
    recs['RecipeName'] = recs['RecipeName'].str.title()
    top = recs.iloc[0]

    st.subheader("Step 3: Best Match")
    st.write(f"**{top['RecipeName']}**")
    st.progress(int(top['Match %']))
    st.caption(f"{top['Match %']}% match")
    time.sleep(1.2)

    # STEP 4
    st.subheader("Step 4: Recommendations")
    for _, row in recs.iterrows():
        st.write(f"**{row['RecipeName']}**")
        st.progress(int(row['Match %']))
        st.caption(f"{row['Match %']}% match")
        st.markdown("")

    time.sleep(1.2)

    # STEP 5
    tags = analyze_food_quality(user_input)

    st.subheader("Step 5: Insights")
    for tag in tags:
        st.success(tag)