import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC

# Load dataset
df = pd.read_csv("IndianFoodDatasetCSV.csv")

# Select columns
df = df[[
    'TranslatedIngredients',
    'Cuisine',
    'Course',
    'PrepTimeInMins',
    'CookTimeInMins',
    'TotalTimeInMins',
    'Diet'
]].copy()

# Drop rows with missing target
df.dropna(subset=['Diet'], inplace=True)

# Fill missing values
df['TranslatedIngredients'] = df['TranslatedIngredients'].fillna('')
df['Cuisine'] = df['Cuisine'].fillna('unknown')
df['Course'] = df['Course'].fillna('unknown')

# Clean text
df['TranslatedIngredients'] = df['TranslatedIngredients'].astype(str).str.lower().str.strip()
df['Cuisine'] = df['Cuisine'].astype(str).str.lower().str.strip()
df['Course'] = df['Course'].astype(str).str.lower().str.strip()
df['Diet'] = df['Diet'].astype(str).str.lower().str.strip()

# Remove empty target values
df = df[df['Diet'] != '']

X = df.drop(columns=['Diet'])
y = df['Diet']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'TranslatedIngredients'),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Cuisine', 'Course']),
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), ['PrepTimeInMins', 'CookTimeInMins', 'TotalTimeInMins'])
    ]
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LinearSVC())
])

param_grid = {
    'preprocessor__text__max_features': [3000, 5000, 7000],
    'preprocessor__text__ngram_range': [(1,1), (1,2)],
    'classifier__C': [0.1, 1, 2, 5]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

joblib.dump(best_model, "indian_food_diet_model.pkl")
print("Model saved successfully as indian_food_diet_model.pkl")
print("Best params:", grid.best_params_)