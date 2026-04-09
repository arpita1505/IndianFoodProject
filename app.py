import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="RecipeIQ · Smart Recommendations",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# GLOBAL CSS  (warm saffron × deep charcoal theme)
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root variables ── */
:root {
    --saffron:   #F4A228;
    --saffron-lt:#FFF3DC;
    --ember:     #E05C2A;
    --bg:        #0F0E0C;
    --surface:   #1A1916;
    --surface2:  #252320;
    --border:    #2E2B26;
    --text:      #F0EBE3;
    --muted:     #8A847B;
    --green:     #5DBB7E;
    --red:       #E05C5C;
}

/* ── Base ── */
html, body, [class*="css"]  { font-family: 'DM Sans', sans-serif; background: var(--bg); color: var(--text); }
.stApp { background: var(--bg); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }

/* ── Inputs ── */
input, textarea, select { background: var(--surface2) !important; color: var(--text) !important; border-color: var(--border) !important; }
.stTextInput > div > div > input { background: var(--surface2) !important; color: var(--text) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; padding: 12px 16px !important; font-size: 15px !important; }
.stTextInput > div > div > input:focus { border-color: var(--saffron) !important; box-shadow: 0 0 0 3px rgba(244,162,40,0.18) !important; }

/* ── Radio ── */
.stRadio label { color: var(--text) !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--muted) !important; font-size: 13px; }

/* ── Multiselect ── */
.stMultiSelect > div > div { background: var(--surface2) !important; border-color: var(--border) !important; border-radius: 10px !important; }
.stMultiSelect span[data-baseweb="tag"] { background: var(--saffron) !important; color: #0F0E0C !important; font-weight: 600; border-radius: 6px; }

/* ── Selectbox ── */
.stSelectbox > div > div { background: var(--surface2) !important; border-color: var(--border) !important; border-radius: 10px !important; color: var(--text) !important; }

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, var(--saffron) 0%, var(--ember) 100%) !important;
    color: #0F0E0C !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    width: 100% !important;
    letter-spacing: 0.5px;
    transition: opacity 0.2s, transform 0.1s !important;
    cursor: pointer;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Custom classes ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(32px, 5vw, 54px);
    font-weight: 900;
    color: var(--text);
    line-height: 1.1;
    margin-bottom: 4px;
}
.hero-title span { color: var(--saffron); }

.hero-sub {
    color: var(--muted);
    font-size: 15px;
    font-weight: 300;
    margin-bottom: 32px;
    letter-spacing: 0.3px;
}

/* Recipe card */
.recipe-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.recipe-card:hover { border-color: var(--saffron); }
.recipe-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: linear-gradient(180deg, var(--saffron), var(--ember));
    border-radius: 4px 0 0 4px;
}

.recipe-name {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 8px;
    line-height: 1.3;
}

.recipe-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 16px;
}
.badge {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: var(--muted);
    font-weight: 500;
}
.badge.saffron { background: rgba(244,162,40,0.12); border-color: rgba(244,162,40,0.3); color: var(--saffron); }
.badge.green   { background: rgba(93,187,126,0.12); border-color: rgba(93,187,126,0.3); color: var(--green); }
.badge.red     { background: rgba(224,92,92,0.12);  border-color: rgba(224,92,92,0.3);  color: var(--red); }

.score-bar-wrap { margin: 12px 0 20px 0; }
.score-label {
    display: flex; justify-content: space-between;
    font-size: 12px; color: var(--muted); margin-bottom: 6px;
}
.score-bar-bg {
    background: var(--surface2);
    border-radius: 99px; height: 8px; overflow: hidden;
}
.score-bar-fill {
    height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--saffron), var(--ember));
    transition: width 0.8s ease;
}

.ing-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 8px;
    margin-top: 10px;
}
.ing-chip {
    padding: 6px 12px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    display: flex; align-items: center; gap: 6px;
}
.ing-chip.have { background: rgba(93,187,126,0.1); color: var(--green); border: 1px solid rgba(93,187,126,0.25); }
.ing-chip.need { background: rgba(224,92,92,0.08); color: var(--red);   border: 1px solid rgba(224,92,92,0.2);  }

.details-row {
    display: flex; gap: 24px; flex-wrap: wrap;
    margin: 16px 0;
    padding: 16px;
    background: var(--surface2);
    border-radius: 12px;
}
.detail-item { text-align: center; }
.detail-val { font-size: 18px; font-weight: 700; color: var(--text); }
.detail-key { font-size: 11px; color: var(--muted); margin-top: 2px; text-transform: uppercase; letter-spacing: 0.6px; }

.nutrition-row {
    display: flex; gap: 12px; flex-wrap: wrap;
    margin: 12px 0;
}
.nut-pill {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    border: 1px solid var(--border);
    background: var(--surface2);
    color: var(--muted);
}
.nut-pill.high { color: var(--green); border-color: rgba(93,187,126,0.3); background: rgba(93,187,126,0.08); }

.view-btn {
    display: inline-block;
    margin-top: 14px;
    padding: 10px 22px;
    background: transparent;
    border: 1px solid var(--saffron);
    color: var(--saffron);
    border-radius: 10px;
    font-size: 13px;
    font-weight: 600;
    text-decoration: none !important;
    transition: background 0.2s, color 0.2s;
    font-family: 'DM Sans', sans-serif;
}
.view-btn:hover { background: var(--saffron); color: #0F0E0C; }

.section-head {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: var(--muted);
    margin: 18px 0 10px 0;
}

.tip-box {
    background: rgba(244,162,40,0.08);
    border: 1px solid rgba(244,162,40,0.25);
    border-radius: 12px;
    padding: 14px 18px;
    color: var(--saffron);
    font-size: 13px;
    margin: 0 0 24px 0;
}

.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: var(--muted);
}
.empty-state .big { font-size: 48px; margin-bottom: 12px; }
.empty-state p { font-size: 15px; }

.sort-row {
    display: flex; gap: 10px; align-items: center;
    margin-bottom: 20px;
}
.results-header {
    font-family: 'Playfair Display', serif;
    font-size: 20px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
}
.results-sub { font-size: 13px; color: var(--muted); margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("IndianFoodDatasetCSV.csv")
    df.columns = df.columns.str.strip()
    df['TranslatedIngredients'] = df['TranslatedIngredients'].fillna('').str.lower()
    df['RecipeName']   = df['RecipeName'].fillna('Unknown Recipe')
    df['Cuisine']      = df['Cuisine'].fillna('Other')
    df['Diet']         = df['Diet'].fillna('Vegetarian')
    df['TotalTimeInMins'] = pd.to_numeric(df.get('TotalTimeInMins', 0), errors='coerce').fillna(0)
    df['Servings']     = df.get('Servings', pd.Series(['N/A'] * len(df))).fillna('N/A')
    df['URL']          = df.get('URL', pd.Series(['#'] * len(df))).fillna('#')
    df['Cleaned_Diet'] = df['Diet'].apply(_clean_diet)
    return df

def _clean_diet(d):
    d = str(d).lower()
    return "Non-Vegetarian" if "non vegetarian" in d else "Vegetarian"


@st.cache_resource
def build_tfidf(df):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X   = vec.fit_transform(df['TranslatedIngredients'])
    return vec, X

df  = load_data()
vec, X = build_tfidf(df)

CUISINES = sorted(df['Cuisine'].dropna().unique().tolist())


# ─────────────────────────────────────────
# NUTRITION HEURISTIC
# ─────────────────────────────────────────
_PROTEIN = {"paneer","chicken","egg","dal","beans","soy","lentil","tofu","fish","mutton","prawn"}
_FAT     = {"butter","oil","cream","cheese","ghee","coconut","cashew","almond"}
_FIBRE   = {"spinach","vegetable","carrot","beans","broccoli","cabbage","pea","mushroom","corn"}

def nutrition(ings: list[str]) -> dict:
    flat = " ".join(ings)
    return {
        "Protein": any(p in flat for p in _PROTEIN),
        "Fat":     any(f in flat for f in _FAT),
        "Fibre":   any(f in flat for f in _FIBRE),
    }


# ─────────────────────────────────────────
# CORE RECOMMENDATION ENGINE
# ─────────────────────────────────────────
def recommend(user_input: str, diet: str, cuisines: list, time_pref: str, sort_by: str, n: int = 8):
    user_ings = [i.strip() for i in user_input.split(",") if i.strip()]
    if not user_ings:
        return pd.DataFrame()

    dfc = df.copy()

    # ── Step 1: compute per-recipe match (plain string lists, NO html) ──
    def _match(row_ings_str):
        # Split on comma, clean each ingredient
        ings = [i.strip() for i in str(row_ings_str).split(",") if i.strip()]
        total = len(ings)
        if total == 0:
            return 0.0, 0.0, [], [], 0

        matched = []
        missing = []
        for ing in ings:
            ing_lower = ing.lower()
            # An ingredient is "matched" if any user ingredient word appears in it
            if any(u.lower() in ing_lower for u in user_ings):
                matched.append(ing)   # plain string, NOT html
            else:
                missing.append(ing)   # plain string, NOT html

        n_matched = len(matched)
        # raw ratio: matched / total recipe ingredients
        ratio = n_matched / total

        # weighted score: ratio × log-scaled ingredient count
        # penalises 1-ingredient recipes (ghee, water, etc.)
        import math
        weight = math.log(total + 1) / math.log(20)   # normalised; recipes with ~20 ings get weight≈1
        weighted = ratio * min(weight, 1.0) * 100

        return ratio * 100, weighted, matched, missing, total

    cols = ['match_score', 'weighted_score', 'matched', 'missing', 'total_ings']
    rows = dfc['TranslatedIngredients'].apply(lambda s: pd.Series(_match(s), index=cols))
    dfc  = pd.concat([dfc, rows], axis=1)

    # ── Step 2: only keep recipes where at least 1 ingredient matched ──
    dfc = dfc[dfc['match_score'] > 0]

    # ── Step 3: filters ──
    if diet != "Both":
        dfc = dfc[dfc["Cleaned_Diet"] == diet]

    if cuisines and "Any Cuisine" not in cuisines:
        dfc = dfc[dfc["Cuisine"].isin(cuisines)]

    if time_pref == "Under 30 mins":
        dfc = dfc[dfc["TotalTimeInMins"] <= 30]
    elif time_pref == "Under 1 hour":
        dfc = dfc[dfc["TotalTimeInMins"] <= 60]

    # ── Step 4: sort ──
    if sort_by == "Fewest Missing":
        dfc["missing_count"] = dfc["missing"].apply(len)
        dfc = dfc.sort_values(["missing_count", "weighted_score"], ascending=[True, False])
    elif sort_by == "Least Time":
        dfc = dfc[dfc["TotalTimeInMins"] > 0]
        dfc = dfc.sort_values(["TotalTimeInMins", "weighted_score"], ascending=[True, False])
    else:  # Best Match (default) — use weighted score
        dfc = dfc.sort_values("weighted_score", ascending=False)

    return dfc.head(n).reset_index(drop=True)


# ─────────────────────────────────────────
# SMART SUGGESTION  (unlock more recipes)
# ─────────────────────────────────────────
def smart_suggestion(user_ings: list[str]) -> str | None:
    counts = {}
    for _, row in df.iterrows():
        recipe_ings = [i.strip() for i in row['TranslatedIngredients'].split(",") if i.strip()]
        missing     = [i for i in recipe_ings if not any(u in i for u in user_ings)]
        for m in missing:
            counts[m] = counts.get(m, 0) + 1
    if not counts:
        return None
    top = sorted(counts, key=counts.get, reverse=True)[0]
    return f"💡 Add <b>{top}</b> to unlock <b>{counts[top]}</b> more recipes!"


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 24px'>
      <div style='font-family:Playfair Display,serif;font-size:22px;font-weight:900;color:#F4A228'>RecipeIQ</div>
      <div style='font-size:12px;color:#8A847B;margin-top:2px'>Smart Ingredient Matching</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🥬 Diet Preference**")
    diet = st.radio("", ["Both", "Vegetarian", "Non-Vegetarian"], label_visibility="collapsed")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("**🌍 Cuisine**")
    selected_cuisines = st.multiselect(
        "",
        ["Any Cuisine"] + CUISINES,
        default=["Any Cuisine"],
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("**⏱ Cooking Time**")
    time_pref = st.selectbox("", ["Any", "Under 30 mins", "Under 1 hour"], label_visibility="collapsed")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("**🔃 Sort Results By**")
    sort_by = st.selectbox("", ["Best Match", "Fewest Missing", "Least Time"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown(f"<div style='font-size:12px;color:#8A847B'>📚 {len(df):,} recipes indexed</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────
st.markdown("""
<div class='hero-title'>Find recipes from <span>what you have</span></div>
<div class='hero-sub'>Enter ingredients → get perfectly matched Indian recipes instantly</div>
""", unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1.2])
with col_input:
    user_input = st.text_input(
        "",
        placeholder="e.g. paneer, butter, tomato, onion, garam masala",
        label_visibility="collapsed"
    )
with col_btn:
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    search_clicked = st.button("🔍 Find Recipes")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# Tip
user_ings = [i.strip() for i in user_input.lower().split(",") if i.strip()]
if user_ings and len(user_ings) < 3:
    st.markdown("<div class='tip-box'>✨ <b>Tip:</b> Add more ingredients for better matches — try at least 3–5.</div>",
                unsafe_allow_html=True)

# ─── RESULTS ───
if search_clicked:
    if not user_input.strip():
        st.warning("Please enter at least one ingredient.")
    else:
        with st.spinner("Matching recipes…"):
            results = recommend(user_input.lower(), diet, selected_cuisines, time_pref, sort_by)

        if results.empty:
            st.markdown("""
            <div class='empty-state'>
              <div class='big'>🍽</div>
              <p>No recipes found. Try adding more ingredients or broadening your filters.</p>
            </div>""", unsafe_allow_html=True)
        else:
            # Smart suggestion
            suggestion = smart_suggestion(user_ings)
            if suggestion:
                st.markdown(f"<div class='tip-box'>{suggestion}</div>", unsafe_allow_html=True)

            st.markdown(f"<div class='results-header'>Top {len(results)} Matches</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='results-sub'>Sorted by <b>{sort_by}</b> · {diet} · {time_pref}</div>",
                        unsafe_allow_html=True)

            for _, row in results.iterrows():
                name       = str(row['RecipeName']).title()
                cuisine    = str(row.get('Cuisine', 'Other'))
                diet_type  = row['Cleaned_Diet']
                score      = round(float(row['match_score']), 1)   # raw % for display
                matched    = list(row['matched'])                   # plain strings
                missing    = list(row['missing'])                   # plain strings
                total_time = int(row['TotalTimeInMins']) if row['TotalTimeInMins'] else 0
                servings   = row.get('Servings', 'N/A')
                url        = str(row.get('URL', '#'))
                nut        = nutrition(row['TranslatedIngredients'].split(","))

                diet_color  = "green" if diet_type == "Vegetarian" else "red"
                score_color = "#5DBB7E" if score >= 60 else ("#F4A228" if score >= 30 else "#E05C5C")

                # Build ingredient chips HTML from plain ingredient strings
                matched_chips = "".join(
                    f"<span class='ing-chip have'>✔ {i.strip()[:30]}</span>"
                    for i in matched[:12]
                )
                missing_chips = "".join(
                    f"<span class='ing-chip need'>✖ {i.strip()[:30]}</span>"
                    for i in missing[:12]
                )
                missing_note = (f"<i style='font-size:12px;color:#8A847B'>…and {len(missing)-12} more</i>"
                                if len(missing) > 12 else "")

                # Nutrition pills
                nut_html = ""
                for k, v in nut.items():
                    cls  = "high" if v else ""
                    icon = "💪" if k == "Protein" else ("🧈" if k == "Fat" else "🥦")
                    nut_html += f"<span class='nut-pill {cls}'>{icon} {k}: {'High' if v else 'Low'}</span>"

                time_display = f"{total_time} mins" if total_time else "N/A"

                st.markdown(f"""
                <div class='recipe-card'>
                  <div class='recipe-name'>{name}</div>
                  <div class='recipe-meta'>
                    <span class='badge saffron'>🌍 {cuisine}</span>
                    <span class='badge {diet_color}'>{'🥗' if diet_type=='Vegetarian' else '🍖'} {diet_type}</span>
                    <span class='badge'>⏱ {time_display}</span>
                    <span class='badge'>🍽 {servings} servings</span>
                  </div>

                  <div class='score-bar-wrap'>
                    <div class='score-label'>
                      <span>Ingredient Match</span>
                      <span style='color:{score_color};font-weight:700'>{score}%</span>
                    </div>
                    <div class='score-bar-bg'>
                      <div class='score-bar-fill' style='width:{min(score,100)}%;background:linear-gradient(90deg,{score_color},{score_color}cc)'></div>
                    </div>
                    <div style='font-size:12px;color:#8A847B;margin-top:5px'>
                      ✅ {len(matched)} matched &nbsp;·&nbsp; ❌ {len(missing)} missing
                    </div>
                  </div>

                  <div class='section-head'>Ingredients</div>
                  <div class='ing-grid'>
                    {matched_chips}
                    {missing_chips}
                  </div>
                  {missing_note}

                  <div class='section-head'>Nutrition Profile</div>
                  <div class='nutrition-row'>{nut_html}</div>

                  <a class='view-btn' href='{url}' target='_blank'>🔗 View Full Recipe →</a>
                </div>
                """, unsafe_allow_html=True)

else:
    # Landing state
    st.markdown("""
    <div style='text-align:center;padding:64px 20px;'>
      <div style='font-size:64px;margin-bottom:16px'>🍛</div>
      <div style='font-size:17px;color:#8A847B;max-width:380px;margin:0 auto;line-height:1.6'>
        Type your available ingredients above and hit <b style='color:#F4A228'>Find Recipes</b> to discover what you can cook right now.
      </div>
    </div>
    """, unsafe_allow_html=True)
