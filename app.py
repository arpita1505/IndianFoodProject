import streamlit as st
import joblib
import pandas as pd

model = joblib.load("indian_food_diet_model.pkl")

st.set_page_config(page_title="Indian Food Diet Prediction", page_icon="🍽️")

st.title("Indian Food Diet Prediction System")
st.write("Enter recipe details below to predict the diet category.")

ingredients = st.text_area("Translated Ingredients", placeholder="rice, milk, sugar, cardamom, almonds")
cuisine = st.text_input("Cuisine", placeholder="indian")
course = st.text_input("Course", placeholder="dessert")
prep_time = st.number_input("Prep Time (mins)", min_value=0, value=10)
cook_time = st.number_input("Cook Time (mins)", min_value=0, value=20)
total_time = st.number_input("Total Time (mins)", min_value=0, value=30)

if st.button("Predict Diet"):
    if not ingredients.strip() or not cuisine.strip() or not course.strip():
        st.error("Please fill all text fields.")
    else:
        input_data = pd.DataFrame([{
            'TranslatedIngredients': ingredients.lower().strip(),
            'Cuisine': cuisine.lower().strip(),
            'Course': course.lower().strip(),
            'PrepTimeInMins': prep_time,
            'CookTimeInMins': cook_time,
            'TotalTimeInMins': total_time
        }])

        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Diet Category: {prediction}")