from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import streamlit as st
import altair as alt

# =========================
# Config
# =========================
st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🏥",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "obesity_pipeline.joblib"
DATA_PATH = BASE_DIR / "Obesity.csv"

# =========================
# Dark Theme CSS
# =========================
DARK_CSS = """
<style>
:root{
  --bg:#0b1220;
  --text:#e6eefc;
  --muted:#a7b6d7;
  --border:rgba(255,255,255,.08);
  --brand:#18a0fb;
}

.stApp{
  background:
    radial-gradient(1200px 800px at 20% 10%, rgba(24,160,251,0.10), transparent 60%),
    radial-gradient(1000px 700px at 80% 0%, rgba(24,194,156,0.08), transparent 55%),
    var(--bg);
  color:var(--text);
}

.card{
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px;
}
.card-title{
  display:flex; gap:10px; align-items:center;
  font-weight:800; font-size:18px; margin-bottom:10px;
}
.section-bar{
  background: linear-gradient(90deg, rgba(24,160,251,0.35), rgba(24,160,251,0.12));
  border: 1px solid rgba(24,160,251,0.25);
  border-radius: 14px;
  padding: 12px 16px;
  font-weight: 900;
  font-size: 18px;
  margin: 6px 0 14px 0;
}
.badge{
  display:inline-block;
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(24,160,251,0.15);
  border: 1px solid rgba(24,160,251,0.25);
  color: var(--text);
  font-size: 12px;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div{
  background: rgba(255,255,255,0.03) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

.stButton > button{
  border-radius:12px !important;
  border:1px solid rgba(24,160,251,0.30) !important;
  background: linear-gradient(180deg, rgba(24,160,251,0.30), rgba(24,160,251,0.12)) !important;
  color: var(--text) !important;
  font-weight: 800 !important;
}
.stButton > button:hover{
  border-color: rgba(24,160,251,0.55) !important;
  transform: translateY(-1px);
}

div[data-testid="stDataFrame"]{
  border-radius:14px;
  border:1px solid var(--border);
  overflow:hidden;
}
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# =========================
# Helpers
# =========================
def section(title: str, icon: str = "🧩"):
    st.markdown(f'<div class="section-bar">{icon} {title}</div>', unsafe_allow_html=True)

def card_open(title: str, icon: str = "📌", badge: Optional[str] = None):
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="card">
          <div class="card-title">{icon} {title} {badge_html}</div>
        """,
        unsafe_allow_html=True
    )

def card_close():
    st.markdown("</div>", unsafe_allow_html=True)

def bmi(height_m: float, weight_kg: float) -> float:
    if height_m <= 0:
        return 0.0
    return weight_kg / (height_m ** 2)

def safe_pct(x: float) -> str:
    try:
        return f"{x:.1f}%"
    except Exception:
        return "—"

# =========================
# Load model
# =========================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo não encontrado em: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

model = load_model()

# =========================
# Load data for Insights
# =========================
@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)

    # BMI
    if "Height" in df.columns and "Weight" in df.columns:
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    # Friendly labels
    if "Gender" in df.columns:
        df["Gender_PT"] = df["Gender"].map({"Male": "Masculino", "Female": "Feminino"}).fillna(df["Gender"].astype(str))

    if "family_history" in df.columns:
        df["family_history_PT"] = df["family_history"].map({"yes": "Sim", "no": "Não"}).fillna(df["family_history"].astype(str))

    if "NObeyesdad" in df.columns:
        map_ob = {
            "Insufficient_Weight": "Peso Insuficiente",
            "Normal_Weight": "Peso Normal",
            "Overweight_Level_I": "Sobrepeso Nível I",
            "Overweight_Level_II": "Sobrepeso Nível II",
            "Obesity_Type_I": "Obesidade Tipo I",
            "Obesity_Type_II": "Obesidade Tipo II",
            "Obesity_Type_III": "Obesidade Tipo III",
        }
        df["Obesity_PT"] = df["NObeyesdad"].map(map_ob).fillna(df["NObeyesdad"].astype(str))

    if "Age" in df.columns:
        bins = [0, 20, 30, 40, 50, 200]
        labels = ["<20", "20–29", "30–39", "40–49", "50+"]
        df["Faixa_Etaria"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False)

    return df

df_data = load_data()

# =========================
# Header + Tabs
# =========================
st.title("🏥 Sistema Preditivo de Obesidade")

tab_inicio, tab_pred, tab_insights, tab_sobre = st.tabs(
    ["🏠 Início", "🧠 Predição", "📈 Insights e Métricas", "ℹ️ Sobre"]
)

# =========================
# TAB: Início
# =========================
with tab_inicio:
    section("Bem-vindo ao Sistema", "🏠")

    colA, colB = st.columns([1.4, 1], gap="large")
    with colA:
        card_open("Objetivo", "🎯")
        st.write(
            "Esta aplicação foi desenvolvida para **auxiliar na estimativa do nível de obesidade** usando "
            "**Machine Learning**, combinando informações de perfil, hábitos e estilo de vida. "
            "O resultado é uma **referência analítica** e deve ser interpretado junto com a avaliação profissional."
        )
        card_close()

        st.write("")
        card_open("Como usar", "🚀")
        st.markdown(
            """
            1. Acesse **🧠 Predição**
            2. Preencha os dados do paciente
            3. Clique em **Fazer Predição**
            4. Analise o resultado e as probabilidades por classe
            """
        )
        card_close()

    with colB:
        card_open("Recursos", "🧩")
        st.markdown(
            """
            - **Acurácia do modelo:** **95%**  
            - **Usabilidade agradável**, com interface organizada e leitura clara para apoiar a tomada de decisão
            """
        )
        card_close()

# =========================
# TAB: Predição (PT na tela / EN no input)
# =========================
with tab_pred:
    section("Predição de Nível de Obesidade", "🧠")

    card_open("Informações do Profissional e Paciente (opcional)", "🧑‍⚕️")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        profissional = st.text_input("Nome do Profissional (opcional)", placeholder="Ex: Dra. Ana Silva")
    with c2:
        registro = st.text_input("Registro do Conselho (opcional)", placeholder="Ex: CRM 123456")
    with c3:
        paciente = st.text_input("Nome do Paciente (opcional)", placeholder="Ex: Maria Santos")
    card_close()

    st.write("")
    section("Dados do Paciente", "📋")

    # Mapeamentos PT -> EN (visível em PT, enviado em EN)
    map_gender = {"Masculino": "Male", "Feminino": "Female"}
    map_yesno = {"Sim": "yes", "Não": "no"}

    map_caec = {
        "Não": "no",
        "Às vezes": "Sometimes",
        "Frequentemente": "Frequently",
        "Sempre": "Always",
    }
    map_calc = {
        "Não": "no",
        "Às vezes": "Sometimes",
        "Frequentemente": "Frequently",
        "Sempre": "Always",
    }
    map_mtrans = {
        "Carro": "Automobile",
        "Moto": "Motorbike",
        "Bicicleta": "Bike",
        "Transporte público": "Public_Transportation",
        "Caminhando": "Walking",
    }

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        card_open("Dados Demográficos", "🧾")
        gender_pt = st.selectbox("Gênero", list(map_gender.keys()))
        gender = map_gender[gender_pt]
        age = st.number_input("Idade", min_value=1, max_value=120, value=30, step=1)
        card_close()

    with col2:
        card_open("Medidas", "📏")
        height = st.number_input("Altura (metros)", min_value=1.00, max_value=2.30, value=1.70, step=0.01)
        weight = st.number_input("Peso (kg)", min_value=20.0, max_value=250.0, value=70.0, step=0.1)
        imc = bmi(height, weight)
        st.markdown(f"**IMC:** `{imc:.2f}`")
        card_close()

    with col3:
        card_open("Histórico", "📚")
        family_pt = st.selectbox("Histórico familiar de excesso de peso", list(map_yesno.keys()))
        family_history = map_yesno[family_pt]

        smoke_pt = st.selectbox("Fuma?", ["Não", "Sim"])
        smoke = "yes" if smoke_pt == "Sim" else "no"

        scc_pt = st.selectbox("Monitora calorias ingeridas?", ["Não", "Sim"])
        scc = "yes" if scc_pt == "Sim" else "no"
        card_close()

    st.write("")
    section("Hábitos e Estilo de Vida", "🌿")

    colA, colB, colC = st.columns(3, gap="large")
    with colA:
        card_open("Alimentação", "🍽️")
        favc_pt = st.selectbox("Consumo frequente de alimentos muito calóricos?", ["Sim", "Não"])
        favc = map_yesno[favc_pt]

        fcvc = st.slider("Consumo de vegetais (1 baixo → 3 alto)", 1.0, 3.0, 2.0, 0.1)
        ncp = st.slider("Refeições principais por dia", 1.0, 4.0, 3.0, 0.1)

        caec_pt = st.selectbox("Belisca/come entre as refeições?", list(map_caec.keys()))
        caec = map_caec[caec_pt]
        card_close()

    with colB:
        card_open("Hidratação", "💧")
        ch2o = st.slider("Água por dia (1 baixa → 3 alta)", 1.0, 3.0, 2.0, 0.1)
        card_close()

        st.write("")
        card_open("Álcool", "🍺")
        calc_pt = st.selectbox("Consumo de álcool", list(map_calc.keys()))
        calc = map_calc[calc_pt]
        card_close()

    with colC:
        card_open("Rotina", "⏱️")
        faf = st.slider("Atividade física (0 baixa → 3 alta)", 0.0, 3.0, 1.0, 0.1)
        tue = st.slider("Tempo de tela (0 baixo → 2 alto)", 0.0, 2.0, 1.0, 0.1)

        mtrans_pt = st.selectbox("Meio de transporte", list(map_mtrans.keys()))
        mtrans = map_mtrans[mtrans_pt]
        card_close()

    st.write("")
    section("Predição", "🔮")

    left, right = st.columns([1, 2], gap="large")
    with left:
        run_pred = st.button("✨ Fazer Predição", use_container_width=True)
    with right:
        st.caption("Resultado gerado a partir do pipeline treinado. Use como apoio à decisão.")

    if run_pred:
        input_data = pd.DataFrame([{
            "Gender": gender,
            "Age": age,
            "Height": height,
            "Weight": weight,
            "family_history": family_history,
            "FAVC": favc,
            "FCVC": fcvc,
            "NCP": ncp,
            "CAEC": caec,
            "SMOKE": smoke,
            "CH2O": ch2o,
            "SCC": scc,
            "FAF": faf,
            "TUE": tue,
            "CALC": calc,
            "MTRANS": mtrans
        }])

        pred = model.predict(input_data)[0]

        card_open("Resultado", "✅", badge="Predição concluída")
        st.markdown(f"### Nível previsto: **{pred}**")
        st.write(f"**Paciente:** {paciente or '—'}  |  **Profissional:** {profissional or '—'}  |  **Registro:** {registro or '—'}")
        st.write(f"**IMC calculado:** `{imc:.2f}`")
        card_close()

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0]
            classes = model.classes_
            proba_df = (
                pd.DataFrame({"Classe": classes, "Probabilidade": proba})
                .sort_values("Probabilidade", ascending=False)
                .reset_index(drop=True)
            )
            st.write("")
            card_open("Probabilidades por classe", "📊", badge="Apoio à decisão")
            st.dataframe(proba_df, use_container_width=True)
            card_close()

# =========================
# TAB: Insights e Métricas (Altair - garante aparecer)
# =========================
with tab_insights:
    section("Insights e Métricas", "📈")

    if df_data.empty:
        st.warning(
            "Não encontrei o arquivo **Obesity.csv** na pasta do projeto. "
            "Coloque o CSV ao lado do `app.py` (ou ajuste `DATA_PATH`)."
        )
    else:
        # Métricas
        total = len(df_data)
        imc_medio = df_data["BMI"].mean() if "BMI" in df_data.columns else None
        idade_media = df_data["Age"].mean() if "Age" in df_data.columns else None

        if "Obesity_PT" in df_data.columns:
            over_ob = ~df_data["Obesity_PT"].isin(["Peso Normal", "Peso Insuficiente"])
            taxa = float(over_ob.mean() * 100)
        else:
            taxa = None

        m1, m2, m3, m4 = st.columns(4, gap="large")
        with m1:
            card_open("Total de Registros", "🧾")
            st.markdown(f"## {total}")
            card_close()
        with m2:
            card_open("IMC Médio", "📏")
            st.markdown(f"## {imc_medio:.2f}" if imc_medio is not None else "## —")
            card_close()
        with m3:
            card_open("Idade Média", "🎂")
            st.markdown(f"## {idade_media:.1f} anos" if idade_media is not None else "## —")
            card_close()
        with m4:
            card_open("Taxa Sobrepeso/Obesidade", "⚠️")
            st.markdown(f"## {taxa:.1f}%" if taxa is not None else "## —")
            card_close()

        st.write("")
        section("Distribuição dos Níveis", "📊")

        left, right = st.columns(2, gap="large")

        if "Obesity_PT" in df_data.columns:
            # Bar chart distribution
            dist = df_data["Obesity_PT"].value_counts().reset_index()
            dist.columns = ["Nível", "Frequência"]

            bar = (
                alt.Chart(dist)
                .mark_bar()
                .encode(
                    x=alt.X("Nível:N", sort="-y", title="Nível de Obesidade"),
                    y=alt.Y("Frequência:Q", title="Frequência"),
                    tooltip=["Nível", "Frequência"]
                )
                .properties(height=360)
            )

            # Pie-ish: use arc chart
            pie = (
                alt.Chart(dist)
                .mark_arc(innerRadius=60)
                .encode(
                    theta=alt.Theta(field="Frequência", type="quantitative"),
                    color=alt.Color(field="Nível", type="nominal", legend=alt.Legend(title="Nível")),
                    tooltip=["Nível", "Frequência"]
                )
                .properties(height=360)
            )

            with left:
                card_open("Distribuição de Níveis (Barras)", "📊")
                st.altair_chart(bar, use_container_width=True)
                card_close()

            with right:
                card_open("Proporção de Níveis (Rosca)", "🧿")
                st.altair_chart(pie, use_container_width=True)
                card_close()
        else:
            st.info("Coluna `NObeyesdad` não encontrada no CSV. Não foi possível montar os gráficos por classe.")

        st.write("")
        section("Relações e Perfil", "🧠")

        c1, c2 = st.columns(2, gap="large")

        # Scatter Age vs BMI
        if all(col in df_data.columns for col in ["Age", "BMI", "Obesity_PT"]):
            scatter_df = df_data[["Age", "BMI", "Obesity_PT"]].dropna()

            scatter = (
                alt.Chart(scatter_df)
                .mark_circle(size=55, opacity=0.55)
                .encode(
                    x=alt.X("Age:Q", title="Idade"),
                    y=alt.Y("BMI:Q", title="IMC"),
                    color=alt.Color("Obesity_PT:N", title="Nível"),
                    tooltip=["Age", "BMI", "Obesity_PT"]
                )
                .properties(height=360)
            )

            with c1:
                card_open("Idade x IMC", "🔎")
                st.altair_chart(scatter, use_container_width=True)
                card_close()

        # Bar Gender x Obesity
        if all(col in df_data.columns for col in ["Gender_PT", "Obesity_PT"]):
            gdf = (
                df_data.groupby(["Gender_PT", "Obesity_PT"])
                .size()
                .reset_index(name="Frequência")
            )

            gender_bar = (
                alt.Chart(gdf)
                .mark_bar()
                .encode(
                    x=alt.X("Gender_PT:N", title="Gênero"),
                    y=alt.Y("Frequência:Q", title="Frequência"),
                    color=alt.Color("Obesity_PT:N", title="Nível"),
                    tooltip=["Gender_PT", "Obesity_PT", "Frequência"]
                )
                .properties(height=360)
            )

            with c2:
                card_open("Distribuição por Gênero", "👥")
                st.altair_chart(gender_bar, use_container_width=True)
                card_close()

        st.write("")
        section("Hábitos e Histórico", "🌿")

        c3, c4 = st.columns(2, gap="large")

        # Family history
        if all(col in df_data.columns for col in ["family_history_PT", "Obesity_PT"]):
            fdf = (
                df_data.groupby(["family_history_PT", "Obesity_PT"])
                .size()
                .reset_index(name="Frequência")
            )

            fam_bar = (
                alt.Chart(fdf)
                .mark_bar()
                .encode(
                    x=alt.X("family_history_PT:N", title="Histórico familiar"),
                    y=alt.Y("Frequência:Q", title="Frequência"),
                    color=alt.Color("Obesity_PT:N", title="Nível"),
                    tooltip=["family_history_PT", "Obesity_PT", "Frequência"]
                )
                .properties(height=360)
            )

            with c3:
                card_open("Impacto do Histórico Familiar", "🧬")
                st.altair_chart(fam_bar, use_container_width=True)
                card_close()

        # Physical activity FAF bins
        if all(col in df_data.columns for col in ["FAF", "Obesity_PT"]):
            tmp = df_data[["FAF", "Obesity_PT"]].dropna().copy()
            tmp["Atividade_Física"] = pd.cut(
                tmp["FAF"],
                bins=[-0.1, 1.0, 2.0, 3.1],
                labels=["Baixa (0–1)", "Média (1–2)", "Alta (2–3)"]
            )
            fafdf = tmp.groupby(["Atividade_Física", "Obesity_PT"]).size().reset_index(name="Frequência")

            faf_bar = (
                alt.Chart(fafdf)
                .mark_bar()
                .encode(
                    x=alt.X("Atividade_Física:N", title="Atividade física (FAF)"),
                    y=alt.Y("Frequência:Q", title="Frequência"),
                    color=alt.Color("Obesity_PT:N", title="Nível"),
                    tooltip=["Atividade_Física", "Obesity_PT", "Frequência"]
                )
                .properties(height=360)
            )

            with c4:
                card_open("Atividade Física x Obesidade", "🏃")
                st.altair_chart(faf_bar, use_container_width=True)
                card_close()

        st.write("")
        section("Dados (opcional)", "🗂️")

        with st.expander("Ver amostra do dataset"):
            st.dataframe(df_data.head(30), use_container_width=True)

        csv_bytes = df_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Baixar dataset (CSV)",
            data=csv_bytes,
            file_name="obesity_dataset.csv",
            mime="text/csv",
            use_container_width=True
        )

# =========================
# TAB: Sobre
# =========================
with tab_sobre:
    section("Sobre o Sistema", "ℹ️")

    colA, colB = st.columns([1.4, 1], gap="large")
    with colA:
        card_open("Visão geral", "📘")
        st.write(
            "Este projeto reúne duas frentes em uma única aplicação: "
            "**predição do nível de obesidade** por meio de um pipeline de Machine Learning e "
            "uma área de **análise exploratória** para entender padrões do dataset (idade, IMC, hábitos e histórico). "
            "A proposta é entregar uma ferramenta clara, objetiva e fácil de navegar para fins acadêmicos."
        )
        card_close()

        st.write("")
        card_open("Autores", "👤")
        st.markdown(
            """
            - **Franco Colmán**
            - **Hugo Duran**
            """
        )
        card_close()

    with colB:
        card_open("Informações técnicas", "🗂️")
        st.markdown(
            """
            - **Entrada do modelo:** 16 variáveis (mantidas em **inglês** no pipeline)  
            - **Saída:** 7 classes de obesidade  
            - **Modelo (pipeline):** `model/obesity_pipeline.joblib`  
            - **Base de dados:** `Obesity.csv`  
            """
        )
        card_close()
