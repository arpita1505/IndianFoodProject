import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Food AI", layout="wide")

# =========================
# CUSTOM CSS (ULTRA UI)
# =========================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1f1c2c, #928dab);
}

.main {
    background-color: transparent;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: white;
}

.subtitle {
    text-align: center;
    color: #ddd;
    margin-bottom: 30px;
}

.card {
    background: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 8px 20px rgba(0,0,0,0.3);
    margin-bottom: 20px;
    color: white;
}

.metric {
    font-size: 28px;
    font-weight: bold;
    color: #00ffcc;
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
# RECOMMENDATION
# =========================
def recommend_food(user_input):
    user_vec = tfidf.transform([user_input.lower()])
    similarity = cosine_similarity(user_vec, X)[0]

    top_indices = similarity.argsort()[-5:][::-1]

    results = df.iloc[top_indices][['RecipeName', 'Cuisine', 'Diet']].copy()
    results['Match %'] = [round(similarity[i] * 100, 2) for i in top_indices]

    return results.sort_values(by="Match %", ascending=False)

# =========================
# FOOD INSIGHTS
# =========================
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
st.markdown('<div class="title">🍴 Indian Food Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Powered Diet Prediction & Smart Recommendations</div>', unsafe_allow_html=True)

# =========================
# INPUT CENTER
# =========================
col1, col2, col3 = st.columns([1,2,1])

with col2:
    user_input = st.text_input("Enter Ingredients", "paneer, butter, tomato")
    analyze = st.button("✨ Analyze Food")

# =========================
# OUTPUT
# =========================
if analyze and user_input:

    vec = tfidf.transform([user_input.lower()])
    prediction = model.predict(vec)[0]

    recs = recommend_food(user_input)
    recs['RecipeName'] = recs['RecipeName'].str.title()

    tags = analyze_food_quality(user_input)

    # =========================
    # TOP CARDS
    # =========================
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="card">
            <h3>🥗 Diet Type</h3>
            <div class="metric">{prediction}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        top = recs.iloc[0]
        st.markdown(f"""
        <div class="card">
            <h3>⭐ Best Match</h3>
            <div class="metric">{top['RecipeName']}</div>
            <p>{top['Match %']}% Match</p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(int(top['Match %']))

    with c3:
        st.markdown('<div class="card"><h3>💡 Insights</h3>', unsafe_allow_html=True)
        for tag in tags:
            st.write(tag)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # =========================
    # RECOMMENDATIONS GRID
    # =========================
    st.markdown("## 🍽️ Recommendations")

    cols = st.columns(2)

    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="card">
                <h3>{row['RecipeName']}</h3>
                <p>{row['Cuisine']} | {row['Diet']}</p>
                <p><b>{row['Match %']}% Match</b></p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(row['Match %']))