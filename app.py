import streamlit as st
import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────
# PAGE CONFIG  — wide layout, sidebar hidden
# ─────────────────────────────────────────
st.set_page_config(
    page_title="RecipeIQ · Smart Recommendations",
    page_icon="🍛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

:root {
    --saffron:  #F4A228;
    --ember:    #E05C2A;
    --bg:       #0F0E0C;
    --surface:  #181614;
    --surface2: #222019;
    --surface3: #2A2722;
    --border:   #302D27;
    --text:     #F0EBE3;
    --muted:    #8A847B;
    --green:    #52B36E;
    --red:      #D95555;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg) !important;
    color: var(--text);
}
.stApp { background: var(--bg) !important; }

/* Hide Streamlit chrome + sidebar toggle */
#MainMenu, footer, header,
[data-testid="stSidebar"],
[data-testid="collapsedControl"] { display: none !important; }

/* Block container */
.block-container {
    max-width: 1260px !important;
    padding: 2.2rem 2rem 4rem !important;
    margin: 0 auto !important;
}

/* Inputs */
.stTextInput > div > div > input {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 11px 16px !important;
    font-size: 14px !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--saffron) !important;
    box-shadow: 0 0 0 3px rgba(244,162,40,0.14) !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--surface2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-size: 13px !important;
}

/* Multiselect */
.stMultiSelect > div > div {
    background: var(--surface2) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
}
.stMultiSelect span[data-baseweb="tag"] {
    background: rgba(244,162,40,0.15) !important;
    color: var(--saffron) !important;
    font-weight: 600;
    border-radius: 6px;
    font-size: 11px;
}

/* Radio */
.stRadio > div { gap: 2px !important; }
.stRadio label { color: var(--text) !important; font-size: 13px !important; }

/* Button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--saffron), var(--ember)) !important;
    color: #0F0E0C !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 11px 18px !important;
    width: 100% !important;
    cursor: pointer !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: opacity 0.15s !important;
}
div[data-testid="stButton"] > button:hover { opacity: 0.82 !important; }

hr { border-color: var(--border) !important; margin: 14px 0 !important; }

/* ── FILTER PANEL ── */
.filter-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 22px 18px 20px;
}
.f-logo {
    font-family: 'Playfair Display', serif;
    font-size: 19px;
    font-weight: 900;
    color: var(--saffron);
    margin-bottom: 1px;
}
.f-logo-sub { font-size: 10px; color: var(--muted); margin-bottom: 16px; letter-spacing: 0.4px; }
.f-label {
    font-size: 10px; font-weight: 700;
    letter-spacing: 1.1px; text-transform: uppercase;
    color: var(--muted); margin: 16px 0 7px 0;
}
.f-div { height: 1px; background: var(--border); margin: 16px 0; }
.f-foot { font-size: 10px; color: var(--muted); text-align: center; margin-top: 14px; }

/* ── HERO ── */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(24px, 3vw, 40px);
    font-weight: 900;
    color: var(--text);
    line-height: 1.15;
    margin-bottom: 5px;
    text-align: center;
}
.hero-title span { color: var(--saffron); }
.hero-sub {
    font-size: 13px; color: var(--muted);
    margin-bottom: 18px; text-align: center;
}

/* ── TIP BOX ── */
.tip-box {
    background: rgba(244,162,40,0.07);
    border: 1px solid rgba(244,162,40,0.2);
    border-radius: 9px;
    padding: 9px 14px;
    color: var(--saffron);
    font-size: 12px;
    margin: 10px 0 16px 0;
}

/* ── RESULTS HEADER ── */
.res-header { font-family:'Playfair Display',serif; font-size:17px; font-weight:700; color:var(--text); margin-bottom:2px; }
.res-sub    { font-size:11px; color:var(--muted); margin-bottom:14px; }

/* ── RECIPE CARD ── */
.recipe-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px 16px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.recipe-card:hover {
    border-color: rgba(244,162,40,0.45);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.recipe-card::before {
    content:''; position:absolute; top:0; left:0;
    width:3px; height:100%;
    background: linear-gradient(180deg, var(--saffron), var(--ember));
}
.r-name {
    font-family: 'Playfair Display', serif;
    font-size: 16px; font-weight: 700;
    color: var(--text); margin-bottom: 8px;
    line-height: 1.35; padding-left: 8px;
}
.r-meta { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:12px; padding-left:8px; }
.badge {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 20px; padding: 2px 9px;
    font-size: 11px; color: var(--muted); font-weight: 500;
}
.badge.s { background:rgba(244,162,40,0.09); border-color:rgba(244,162,40,0.25); color:var(--saffron); }
.badge.g { background:rgba(82,179,110,0.09); border-color:rgba(82,179,110,0.25); color:var(--green); }
.badge.r { background:rgba(217,85,85,0.09);  border-color:rgba(217,85,85,0.25);  color:var(--red); }

/* score row */
.score-row {
    display: flex; align-items: center; gap: 10px;
    padding-left: 8px; margin-bottom: 6px;
}
.score-lbl { font-size:10px; text-transform:uppercase; letter-spacing:0.8px; color:var(--muted); font-weight:600; }
.score-num { font-size:13px; font-weight:700; }
.bar-bg { width:180px; height:5px; background:var(--surface3); border-radius:99px; overflow:hidden; display:inline-block; }
.bar-fill { height:100%; border-radius:99px; }
.score-counts { font-size:11px; color:var(--muted); padding-left:8px; margin-bottom:12px; }

/* section head */
.s-head {
    font-size:10px; font-weight:700; letter-spacing:1px;
    text-transform:uppercase; color:var(--muted);
    margin: 12px 0 7px 8px;
}

/* chips */
.ing-wrap { display:flex; flex-wrap:wrap; gap:5px; padding-left:8px; }
.chip { padding:3px 9px; border-radius:6px; font-size:11px; font-weight:500; white-space:nowrap; }
.chip.h { background:rgba(82,179,110,0.09); color:var(--green); border:1px solid rgba(82,179,110,0.2); }
.chip.n { background:rgba(217,85,85,0.07);  color:var(--red);   border:1px solid rgba(217,85,85,0.17); }
.more-n { font-size:10px; color:var(--muted); font-style:italic; padding-left:8px; margin-top:3px; }

/* nutrition */
.nut-row { display:flex; flex-wrap:wrap; gap:5px; padding-left:8px; margin-top:3px; }
.nut-pill {
    padding:3px 10px; border-radius:20px; font-size:11px; font-weight:600;
    background:var(--surface2); border:1px solid var(--border); color:var(--muted);
}
.nut-pill.hi { color:var(--green); background:rgba(82,179,110,0.08); border-color:rgba(82,179,110,0.2); }

/* view link */
.view-link {
    display:inline-flex; align-items:center; gap:5px;
    margin: 12px 0 0 8px; padding: 7px 16px;
    background: transparent; border: 1px solid var(--saffron);
    color: var(--saffron) !important; border-radius:7px;
    font-size:11px; font-weight:600;
    text-decoration: none !important;
    transition: background 0.16s, color 0.16s;
    font-family: 'DM Sans', sans-serif;
}
.view-link:hover { background:var(--saffron); color:#0F0E0C !important; }

.empty-state { text-align:center; padding:70px 20px; color:var(--muted); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("IndianFoodDatasetCSV.csv")
    df.columns = df.columns.str.strip()
    df['TranslatedIngredients'] = df['TranslatedIngredients'].fillna('').str.lower()
    df['RecipeName']      = df['RecipeName'].fillna('Unknown Recipe')
    df['Cuisine']         = df['Cuisine'].fillna('Other')
    df['Diet']            = df['Diet'].fillna('Vegetarian')
    df['TotalTimeInMins'] = pd.to_numeric(df.get('TotalTimeInMins', 0), errors='coerce').fillna(0)
    df['Servings']        = df.get('Servings', pd.Series(['N/A'] * len(df))).fillna('N/A')
    df['URL']             = df.get('URL', pd.Series(['#'] * len(df))).fillna('#')
    df['Cleaned_Diet']    = df['Diet'].apply(
        lambda d: "Non-Vegetarian" if "non vegetarian" in str(d).lower() else "Vegetarian"
    )
    return df

@st.cache_resource
def build_tfidf(_df):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    X   = vec.fit_transform(_df['TranslatedIngredients'])
    return vec, X

df       = load_data()
_vec, _X = build_tfidf(df)
CUISINES = sorted(df['Cuisine'].dropna().unique().tolist())


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
_PROTEIN = {"paneer","chicken","egg","dal","beans","soy","lentil","tofu","fish","mutton","prawn"}
_FAT     = {"butter","oil","cream","cheese","ghee","coconut","cashew","almond"}
_FIBRE   = {"spinach","vegetable","carrot","beans","broccoli","cabbage","pea","mushroom","corn"}

def get_nutrition(s: str) -> dict:
    flat = s.lower()
    return {
        "Protein": any(p in flat for p in _PROTEIN),
        "Fat":     any(f in flat for f in _FAT),
        "Fibre":   any(f in flat for f in _FIBRE),
    }

def recommend(user_input, diet, cuisines, time_pref, sort_by, n=8):
    user_ings = [i.strip() for i in user_input.split(",") if i.strip()]
    if not user_ings:
        return pd.DataFrame()

    dfc = df.copy()

    def _match(s):
        ings  = [i.strip() for i in str(s).split(",") if i.strip()]
        total = len(ings)
        if total == 0:
            return 0.0, 0.0, [], [], 0
        matched, missing = [], []
        for ing in ings:
            (matched if any(u.lower() in ing.lower() for u in user_ings) else missing).append(ing)
        ratio    = len(matched) / total
        weight   = math.log(total + 1) / math.log(20)
        weighted = ratio * min(weight, 1.0) * 100
        return ratio * 100, weighted, matched, missing, total

    cols = ['match_score', 'weighted_score', 'matched', 'missing', 'total_ings']
    rows = dfc['TranslatedIngredients'].apply(lambda s: pd.Series(_match(s), index=cols))
    dfc  = pd.concat([dfc, rows], axis=1)
    dfc  = dfc[dfc['match_score'] > 0]

    if diet != "Both":
        dfc = dfc[dfc["Cleaned_Diet"] == diet]
    if cuisines and "Any Cuisine" not in cuisines:
        dfc = dfc[dfc["Cuisine"].isin(cuisines)]
    if time_pref == "Under 30 mins":
        dfc = dfc[dfc["TotalTimeInMins"] <= 30]
    elif time_pref == "Under 1 hour":
        dfc = dfc[dfc["TotalTimeInMins"] <= 60]

    if sort_by == "Fewest Missing":
        dfc["_mc"] = dfc["missing"].apply(len)
        dfc = dfc.sort_values(["_mc", "weighted_score"], ascending=[True, False])
    elif sort_by == "Least Time":
        dfc = dfc[dfc["TotalTimeInMins"] > 0]
        dfc = dfc.sort_values(["TotalTimeInMins", "weighted_score"], ascending=[True, False])
    else:
        dfc = dfc.sort_values("weighted_score", ascending=False)

    return dfc.head(n).reset_index(drop=True)

def smart_suggestion(user_ings):
    counts = {}
    for _, row in df.iterrows():
        for ing in [i.strip() for i in row['TranslatedIngredients'].split(",") if i.strip()]:
            if not any(u.lower() in ing for u in user_ings):
                counts[ing] = counts.get(ing, 0) + 1
    if not counts:
        return None
    top = sorted(counts, key=counts.get, reverse=True)[0]
    return f"💡 Add <b>{top}</b> to unlock <b>{counts[top]}</b> more recipes!"


# ─────────────────────────────────────────
# LAYOUT  — left filter panel + right content
# ─────────────────────────────────────────
col_filter, _gap, col_main = st.columns([1, 0.05, 2.5])

# ══════════════════════════
# FILTER PANEL
# ══════════════════════════
with col_filter:
    st.markdown("<div class='filter-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='f-logo'>RecipeIQ</div><div class='f-logo-sub'>Smart Ingredient Matching</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='f-div'></div>", unsafe_allow_html=True)

    st.markdown("<div class='f-label'>Diet Preference</div>", unsafe_allow_html=True)
    diet = st.radio("diet", ["Both", "Vegetarian", "Non-Vegetarian"], label_visibility="collapsed")

    st.markdown("<div class='f-div'></div>", unsafe_allow_html=True)
    st.markdown("<div class='f-label'>Cuisine</div>", unsafe_allow_html=True)
    selected_cuisines = st.multiselect("cuisine", ["Any Cuisine"] + CUISINES,
                                       default=["Any Cuisine"], label_visibility="collapsed")

    st.markdown("<div class='f-div'></div>", unsafe_allow_html=True)
    st.markdown("<div class='f-label'>Cooking Time</div>", unsafe_allow_html=True)
    time_pref = st.selectbox("time", ["Any", "Under 30 mins", "Under 1 hour"],
                             label_visibility="collapsed")

    st.markdown("<div class='f-div'></div>", unsafe_allow_html=True)
    st.markdown("<div class='f-label'>Sort By</div>", unsafe_allow_html=True)
    sort_by = st.selectbox("sort", ["Best Match", "Fewest Missing", "Least Time"],
                           label_visibility="collapsed")

    st.markdown("<div class='f-div'></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='f-foot'>📚 {len(df):,} recipes indexed</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════
# MAIN CONTENT
# ══════════════════════════
with col_main:
    st.markdown("""
    <div class='hero-title'>Find recipes from <span>what you have</span></div>
    <div class='hero-sub'>Enter ingredients separated by commas to discover recipes</div>
    """, unsafe_allow_html=True)

    inp_c, btn_c = st.columns([5, 1.1])
    with inp_c:
        user_input = st.text_input("ing", value="",
                                   placeholder="e.g. paneer, onion, tomato, garam masala",
                                   label_visibility="collapsed")
    with btn_c:
        st.markdown("<div style='height:2px'></div>", unsafe_allow_html=True)
        search_clicked = st.button("🔍 Find Recipes")

    user_ings = [i.strip() for i in user_input.lower().split(",") if i.strip()]
    if user_ings and len(user_ings) < 3:
        st.markdown("<div class='tip-box'>✨ <b>Tip:</b> Add at least 3–5 ingredients for better matches.</div>",
                    unsafe_allow_html=True)

    if search_clicked:
        if not user_input.strip():
            st.warning("Please enter at least one ingredient.")
        else:
            with st.spinner("Matching recipes…"):
                results = recommend(user_input.lower(), diet, selected_cuisines, time_pref, sort_by)

            if results.empty:
                st.markdown("""<div class='empty-state'>
                  <div style='font-size:44px;margin-bottom:10px'>🍽</div>
                  <p>No recipes found. Try more ingredients or broader filters.</p>
                </div>""", unsafe_allow_html=True)
            else:
                sug = smart_suggestion(user_ings)
                if sug:
                    st.markdown(f"<div class='tip-box'>{sug}</div>", unsafe_allow_html=True)

                st.markdown(f"<div class='res-header'>Top {len(results)} Matches</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='res-sub'>Sorted by <b>{sort_by}</b> &nbsp;·&nbsp; {diet} &nbsp;·&nbsp; {time_pref}</div>",
                    unsafe_allow_html=True)

                for _, row in results.iterrows():
                    name      = str(row['RecipeName']).title()
                    cuisine   = str(row.get('Cuisine', 'Other'))
                    diet_type = row['Cleaned_Diet']
                    score     = round(float(row['match_score']), 1)
                    matched   = list(row['matched'])
                    missing   = list(row['missing'])
                    t_time    = int(row['TotalTimeInMins']) if row['TotalTimeInMins'] else 0
                    servings  = row.get('Servings', 'N/A')
                    url       = str(row.get('URL', '#'))
                    nut       = get_nutrition(row['TranslatedIngredients'])

                    d_cls  = "g" if diet_type == "Vegetarian" else "r"
                    d_icon = "🥗" if diet_type == "Vegetarian" else "🍖"
                    s_col  = "#52B36E" if score >= 60 else ("#F4A228" if score >= 30 else "#D95555")
                    t_disp = f"{t_time} mins" if t_time else "N/A"

                    have_chips = "".join(
                        f"<span class='chip h'>✔ {i.strip()[:34]}</span>"
                        for i in matched[:14])
                    need_chips = "".join(
                        f"<span class='chip n'>✖ {i.strip()[:34]}</span>"
                        for i in missing[:10])
                    extra = len(missing) - 10
                    more  = f"<div class='more-n'>…and {extra} more missing</div>" if extra > 0 else ""

                    nut_html = "".join(
                        f"<span class='nut-pill {'hi' if v else ''}'>"
                        f"{'💪' if k=='Protein' else ('🧈' if k=='Fat' else '🥦')} "
                        f"{k}: {'High' if v else 'Low'}</span>"
                        for k, v in nut.items()
                    )

                    st.markdown(f"""
                    <div class='recipe-card'>
                      <div class='r-name'>{name}</div>
                      <div class='r-meta'>
                        <span class='badge s'>🌍 {cuisine}</span>
                        <span class='badge {d_cls}'>{d_icon} {diet_type}</span>
                        <span class='badge'>⏱ {t_disp}</span>
                        <span class='badge'>🍽 {servings} servings</span>
                      </div>
                      <div class='score-row'>
                        <span class='score-lbl'>Match</span>
                        <span class='score-num' style='color:{s_col}'>{score}%</span>
                        <div class='bar-bg'><div class='bar-fill' style='width:{min(score,100)}%;background:{s_col}'></div></div>
                      </div>
                      <div class='score-counts'>✅ {len(matched)} matched &nbsp;·&nbsp; ❌ {len(missing)} missing</div>
                      <div class='s-head'>Ingredients</div>
                      <div class='ing-wrap'>{have_chips}{need_chips}</div>
                      {more}
                      <div class='s-head'>Nutrition</div>
                      <div class='nut-row'>{nut_html}</div>
                      <a class='view-link' href='{url}' target='_blank'>🔗 View Full Recipe →</a>
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='text-align:center;padding:80px 20px;'>
          <div style='font-size:50px;margin-bottom:14px'>🍛</div>
          <div style='font-size:14px;color:#8A847B;max-width:320px;margin:0 auto;line-height:1.75;'>
            Type your available ingredients and press
            <b style='color:#F4A228'>Find Recipes</b> to get started.
          </div>
        </div>
        """, unsafe_allow_html=True)
