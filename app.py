import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity

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
# RECOMMENDATION WITH %
# =========================
def recommend_food(user_input):
    user_vec = tfidf.transform([user_input.lower()])
    similarity = cosine_similarity(user_vec, X)[0]

    top_indices = similarity.argsort()[-5:][::-1]

    results = df.iloc[top_indices][['RecipeName', 'Cuisine', 'Diet']].copy()

    results['Match %'] = [round(similarity[i] * 100, 2) for i in top_indices]

    results = results.sort_values(by="Match %", ascending=False)

    return results

# =========================
# FOOD INSIGHTS
# =========================
def analyze_food_quality(user_input):
    user_input = user_input.lower()

    tags = []
    insights = []

    if any(x in user_input for x in ["paneer", "chicken", "egg", "dal"]):
        tags.append("High Protein 💪")
        insights.append("Good source of protein")

    if any(x in user_input for x in ["rice", "potato", "bread"]):
        tags.append("High Carbs ⚡")
        insights.append("Energy-rich food")

    if any(x in user_input for x in ["butter", "ghee", "cream"]):
        tags.append("High Fat ⚠️")
        insights.append("May be heavy to digest")

    if any(x in user_input for x in ["vegetable", "spinach", "salad"]):
        tags.append("Healthy Choice 🟢")
        insights.append("Rich in fiber and vitamins")

    if not tags:
        tags.append("Balanced ⚖️")
        insights.append("Moderate nutritional profile")

    return tags, insights

# =========================
# UI
# =========================
st.set_page_config(page_title="Indian Food App")

st.markdown("""
    <h1 style='text-align: center;'>🍴 Indian Food Intelligence System</h1>
    <p style='text-align: center; color: gray;'>Diet Prediction + Smart Recommendations + Insights</p>
""", unsafe_allow_html=True)

# =========================
# INPUT
# =========================
user_input = st.text_area("Enter Ingredients (comma separated)", "onion, paneer, tomato")

# =========================
# BUTTON
# =========================
if st.button("Analyze Food"):

    if user_input:

        # =========================
        # PREDICTION
        # =========================
        vec = tfidf.transform([user_input.lower()])
        prediction = model.predict(vec)[0]

        st.success(f"Predicted Diet: {prediction}")

        st.markdown("---")

        # =========================
        # RECOMMENDATIONS
        # =========================
        st.subheader("🍽️ Similar Food Recommendations")

        recs = recommend_food(user_input)
        recs['RecipeName'] = recs['RecipeName'].str.title()

        # ⭐ BEST MATCH
        top_match = recs.iloc[0]
        st.success(f"⭐ Best Match: {top_match['RecipeName']} ({top_match['Match %']}%)")

        st.markdown("### Top Matches")

        # 🔥 SHOW WITH PROGRESS BARS
        for _, row in recs.iterrows():
            st.write(f"**{row['RecipeName']}** ({row['Cuisine']}, {row['Diet']})")
            st.progress(int(row['Match %']))
            st.write(f"Match: {row['Match %']}%")
            st.markdown("---")

        st.markdown("---")

        # =========================
        # FOOD INSIGHTS
        # =========================
        st.subheader("🥗 Food Insights")

        tags, insights = analyze_food_quality(user_input)

        st.write("### Health Tags")
        for tag in tags:
            st.success(tag)

        st.write("### Insights")
        for i in insights:
            st.write(f"• {i}")