import streamlit as st
import joblib
import pandas as pd
import numpy as np
import base64

# =========================
# FUNÇÃO: imagem → base64
# =========================
def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# =========================
# CARREGAR IMAGENS DO PC
# =========================
mars = img_to_base64("web/assets/planets/mars.png")
earth = img_to_base64("web/assets/planets/earth.png")
saturn = img_to_base64("web/assets/planets/saturn.png")
jupiter = img_to_base64("web/assets/planets/jupiter.png")

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="ExoSpace AI",
    page_icon="🪐",
    layout="centered"
)

# =========================
# CSS + ANIMAÇÕES
# =========================
st.markdown("""
<style>

.stApp {
    background: radial-gradient(circle at center, #0b0f2a 0%, #000000 100%);
    overflow-x: hidden;
}

/* estrelas animadas */
.stars, .stars:after {
    position: fixed;
    width: 2px;
    height: 2px;
    background: transparent;
    box-shadow:
        20px 40px white,
        80px 120px white,
        200px 300px white,
        400px 500px white,
        600px 200px white,
        900px 400px white,
        1200px 600px white;
    animation: moveStars 60s linear infinite;
}

.stars:after {
    content: "";
    top: 2000px;
}

@keyframes moveStars {
    from {transform: translateY(0);}
    to {transform: translateY(-2000px);}
}

/* planetas */
.planet {
    position: fixed;
    width: 120px;
    opacity: 0.85;
    z-index: 0;
    animation: float 10s ease-in-out infinite;
    filter: drop-shadow(0 0 10px rgba(100,200,255,0.5));
}

/* posições */
.p1 { top: 20px; left: 20px; }
.p2 { top: 60px; right: 40px; width: 140px; animation-duration: 14s; }
.p3 { bottom: 40px; left: 60px; width: 110px; animation-duration: 12s; }
.p4 { bottom: 50px; right: 60px; width: 160px; animation-duration: 16s; }

/* flutuação */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-25px); }
    100% { transform: translateY(0px); }
}

/* UI acima */
.block-container {
    position: relative;
    z-index: 2;
}

/* neon */
h1, h2, h3 {
    color: #7fd1ff;
    text-shadow: 0 0 10px #1f8fff;
}

</style>
""", unsafe_allow_html=True)

# =========================
# ELEMENTOS VISUAIS
# =========================
st.markdown('<div class="stars"></div>', unsafe_allow_html=True)

st.markdown(f"""
<img class="planet p1" src="data:image/png;base64,{mars}">
<img class="planet p2" src="data:image/png;base64,{earth}">
<img class="planet p3" src="data:image/png;base64,{saturn}">
<img class="planet p4" src="data:image/png;base64,{jupiter}">
""", unsafe_allow_html=True)

# =========================
# TÍTULO
# =========================
st.title("🪐 ExoSpace AI")
st.subheader("Classificador de Exoplanetas com IA")

st.divider()

# =========================
# MODELO
# =========================
data = joblib.load("model/exospaceai.pkl")
model = data["model"]
features = data["features"]
imputer = data.get("imputer", None)

# =========================
# INPUTS
# =========================
st.header("🔢 Parâmetros")

col1, col2 = st.columns(2)

with col1:
    pl_bmasse = st.slider("Massa do planeta", 0.01, 20.0, 1.0)
    pl_orbsmax = st.slider("Semi-eixo maior da órbita", 0.01, 10.0, 1.0)
    pl_orbper = st.slider("Período orbital", 1, 2000, 365)

with col2:
    pl_bmasse = st.number_input("Massa (manual)", 0.01, 20.0, pl_bmasse, format="%.4f")
    pl_orbsmax = st.number_input("Órbita (manual)", 0.01, 10.0, pl_orbsmax, format="%.4f")
    pl_orbper = st.number_input("Período (manual)", 1, 2000, pl_orbper)

st.header("🌟 Estrela")

col3, col4 = st.columns(2)

with col3:
    st_mass = st.slider("Massa da estrela", 0.1, 3.0, 1.0)
    st_rad = st.slider("Raio da estrela", 0.1, 5.0, 1.0)
    st_teff = st.slider("Temperatura (K)", 2000, 10000, 5800)

with col4:
    st_mass = st.number_input("Massa (manual)", 0.1, 3.0, st_mass, format="%.3f")
    st_rad = st.number_input("Raio (manual)", 0.1, 5.0, st_rad, format="%.3f")
    st_teff = st.number_input("Temp (manual)", 2000, 10000, st_teff)

# =========================
# PREVISÃO
# =========================
if st.button("🚀 Classificar"):

    novo = pd.DataFrame([{
        "pl_bmasse": pl_bmasse,
        "pl_orbsmax": pl_orbsmax,
        "pl_orbper": pl_orbper,
        "st_mass": st_mass,
        "st_rad": st_rad,
        "st_teff": st_teff
    }])

    # feature engineering
    novo["density_proxy"] = novo["pl_bmasse"] / (novo["pl_orbsmax"] ** 3)
    novo["surface_gravity"] = novo["pl_bmasse"] / (novo["pl_orbsmax"] ** 2)
    novo["log_mass"] = np.log1p(novo["pl_bmasse"])
    novo["log_orbsmax"] = np.log1p(novo["pl_orbsmax"])
    novo["stellar_influence"] = novo["st_mass"] / (novo["pl_orbsmax"] + 1e-6)

    novo = novo[features]

    if imputer:
        novo = pd.DataFrame(imputer.transform(novo), columns=features)

    pred = model.predict(novo)[0]
    conf = np.max(model.predict_proba(novo)[0])

    st.success(f"🪐 Classe: {pred}")
    st.info(f"🎯 Confiança: {conf:.2f}")
    st.progress(float(conf))
