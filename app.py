import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Smart Recipe System", layout="centered")

# =========================
# STYLE
# =========================
st.markdown("""
<style>
h1 { color: #1f77b4; text-align: center; }
.card {
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
    border: 1px solid #ddd;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("IndianFoodDatasetCSV.csv")
df.columns = df.columns.str.strip()

df['TranslatedIngredients'] = df['TranslatedIngredients'].fillna('').str.lower()
df['RecipeName'] = df['RecipeName'].fillna('')
df['Cuisine'] = df['Cuisine'].fillna('Other')
df['Diet'] = df['Diet'].fillna('Vegetarian')

# =========================
# CLEAN DIET COLUMN
# =========================
def clean_diet(diet):
    diet = str(diet).lower()
    if "non vegetarian" in diet:
        return "Non-Vegetarian"
    elif "vegetarian" in diet or "egg" in diet:
        return "Vegetarian"
    else:
        return "Vegetarian"

df["Cleaned_Diet"] = df["Diet"].apply(clean_diet)

# =========================
# TF-IDF (for similarity)
# =========================
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['TranslatedIngredients'])

# =========================
# NUTRITION LOGIC
# =========================
protein_rich = ["paneer", "chicken", "egg", "dal", "beans", "soy"]
fat_rich = ["butter", "oil", "cream", "cheese", "ghee"]
fibre_rich = ["spinach", "vegetable", "carrot", "beans", "broccoli"]

def get_nutrition(ingredients):
    protein = any(i in ingredients for i in protein_rich)
    fat = any(i in ingredients for i in fat_rich)
    fibre = any(i in ingredients for i in fibre_rich)

    return {
        "Protein": "High" if protein else "Low",
        "Fat": "High" if fat else "Low",
        "Fibre": "High" if fibre else "Low"
    }

# =========================
# MATCHING FUNCTION
# =========================
def get_recommendations(user_input, diet_filter, cuisine_filter, time_filter):

    user_vec = tfidf.transform([user_input])
    similarity = cosine_similarity(user_vec, X)[0]

    df_copy = df.copy()
    df_copy["score"] = similarity * 100

    # FILTER: DIET
    if diet_filter != "Both":
        df_copy = df_copy[df_copy["Cleaned_Diet"] == diet_filter]

    # FILTER: CUISINE
    if "Any Cuisine" not in cuisine_filter:
        df_copy = df_copy[df_copy["Cuisine"].isin(cuisine_filter)]

    # FILTER: TIME
    if time_filter == "Under 30 mins":
        df_copy = df_copy[df_copy["TotalTimeInMins"] <= 30]
    elif time_filter == "Under 1 hour":
        df_copy = df_copy[df_copy["TotalTimeInMins"] <= 60]

    return df_copy.sort_values(by="score", ascending=False).head(5)

# =========================
# HEADER
# =========================
st.markdown("<h1>🍴 Smart Recipe Recommendation</h1>", unsafe_allow_html=True)

# =========================
# INPUT SECTION
# =========================
user_input = st.text_input("Enter Ingredients", "paneer, butter, tomato")

diet = st.radio("Select Diet", ["Both", "Vegetarian", "Non-Vegetarian"])

cuisine_options = [
    "Indian", "Continental", "Chinese", "Italian", "Mexican",
    "Thai", "Mediterranean", "American", "Japanese", "Others"
]

selected_cuisine = st.multiselect(
    "Select Cuisine",
    ["Any Cuisine"] + cuisine_options,
    default=["Any Cuisine"]
)

time_filter = st.selectbox("Cooking Time", ["Any", "Under 30 mins", "Under 1 hour"])

# =========================
# PROCESS BUTTON
# =========================
if st.button("Find Recipes"):

    user_input_clean = user_input.lower()

    results = get_recommendations(user_input_clean, diet, selected_cuisine, time_filter)

    st.markdown("---")

    st.subheader(f"Top Recipes for: {user_input}")

    for _, row in results.iterrows():

        recipe_name = row["RecipeName"].title()
        recipe_ingredients = row["TranslatedIngredients"].split(",")
        score = round(row["score"], 2)
        time_total = row.get("TotalTimeInMins", "N/A")
        servings = row.get("Servings", "N/A")
        cuisine = row.get("Cuisine", "Other")
        diet_type = row["Cleaned_Diet"]
        url = row.get("URL", "#")

        user_ingredients = [i.strip() for i in user_input_clean.split(",")]

        matched = [i for i in recipe_ingredients if any(u in i for u in user_ingredients)]
        missing = [i for i in recipe_ingredients if i not in matched]

        nutrition = get_nutrition(recipe_ingredients)

        # =========================
        # CARD UI
        # =========================
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            st.subheader(recipe_name)

            st.write(f"🍽 Cuisine: {cuisine}")
            st.write(f"🥗 Diet: {diet_type}")
            st.write(f"📊 Match Score: {score}%")
            st.write(f"❗ Missing Ingredients: {len(missing)}")

            st.markdown("### 🧾 Ingredients")

            for i in matched:
                st.markdown(f"🟢 {i.strip()}")

            for i in missing:
                st.markdown(f"🔴 {i.strip()}")

            st.markdown("### ⏱ Details")
            st.write(f"Total Time: {time_total} mins")
            st.write(f"Servings: {servings}")

            st.markdown("### 🧬 Nutrition")
            st.write(f"💪 Protein: {nutrition['Protein']}")
            st.write(f"🧈 Fat: {nutrition['Fat']}")
            st.write(f"🥦 Fibre: {nutrition['Fibre']}")

            st.markdown(f"[🔗 View Recipe]({url})")

            st.markdown("</div>", unsafe_allow_html=True)
