from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
import streamlit as st
import altair as alt
import numpy as np
import matplotlib.pyplot as plt

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
def _resolve_target_col(df: pd.DataFrame) -> Optional[str]:
    """
    Padroniza a coluna alvo para os gráficos.
    Preferência: Obesity > Obesity_level > NObeyesdad > NObeyesdad (variações)
    """
    candidates = ["Obesity", "Obesity_level", "NObeyesdad", "NObeyesdad ", "NObeyesdad\r"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback por aproximação
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("obesity", "obesity_level", "nobeyesdad"):
            return c
    return None

@st.cache_data
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()

    df = pd.read_csv(DATA_PATH)

    # Padroniza coluna alvo em "Obesity" (para visualização e cálculos)
    target_col = _resolve_target_col(df)
    if target_col is not None and target_col != "Obesity":
        df["Obesity"] = df[target_col]
    elif target_col == "Obesity":
        # já está ok
        pass

    # BMI
    if "Height" in df.columns and "Weight" in df.columns:
        df["BMI"] = df["Weight"] / (df["Height"] ** 2)

    # Traduções (apenas visualização)
    if "Gender" in df.columns:
        df["Gender_PT"] = df["Gender"].map({"Male": "Masculino", "Female": "Feminino"}).fillna(df["Gender"].astype(str))

    if "family_history" in df.columns:
        df["family_history_PT"] = df["family_history"].map({"yes": "Sim", "no": "Não"}).fillna(df["family_history"].astype(str))

    # Faixas etárias (bins do seu requisito)
    if "Age" in df.columns:
        bins = [0, 18, 25, 30, 35, 40, 50, 100]
        labels = ["0–17", "18–24", "25–29", "30–34", "35–39", "40–49", "50–99"]
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
            "Esta aplicação foi desenvolvida para **estimar o nível de obesidade** por meio de "
            "**Machine Learning**, usando informações de perfil e hábitos. "
            "O resultado é uma **referência analítica** e deve ser interpretado junto com a avaliação clínica."
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
            - Interface com **boa usabilidade**, organizada e fácil de interpretar para apoiar a decisão
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

    map_caec = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
    map_calc = {"Não": "no", "Às vezes": "Sometimes", "Frequentemente": "Frequently", "Sempre": "Always"}
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
# TAB: Insights e Métricas (SOMENTE os gráficos solicitados)
# =========================
with tab_insights:
    section("Insights e Métricas", "📈")

    if df_data.empty:
        st.warning(
            "Não encontrei o arquivo **Obesity.csv** na pasta do projeto. "
            "Coloque o CSV ao lado do `app.py` (ou ajuste `DATA_PATH`)."
        )
    else:
        # validações mínimas
        required_target = "Obesity" in df_data.columns
        if not required_target:
            st.error(
                "Não encontrei a coluna alvo para os gráficos. "
                "Verifique se o CSV possui **Obesity** ou **Obesity_level**."
            )
        else:
            # =========================
            # 01 + 02 — Distribuição (contagem e %)
            # =========================
            section("01 + 02 — Distribuição do nível de obesidade", "📊")

            vc_count = df_data["Obesity"].value_counts(dropna=False)
            vc_pct = df_data["Obesity"].value_counts(normalize=True, dropna=False) * 100

            dist_df = pd.DataFrame({
                "Obesity": vc_count.index.astype(str),
                "Contagem": vc_count.values,
                "Percentual": vc_pct.reindex(vc_count.index).values
            })

            c1, c2 = st.columns(2, gap="large")
            with c1:
                card_open("01 — Distribuição (contagem)", "📊")
                fig, ax = plt.subplots()
                ax.bar(dist_df["Obesity"], dist_df["Contagem"])
                ax.set_title("Distribuição do nível de obesidade (contagem)")
                ax.set_xlabel("Obesity")
                ax.set_ylabel("Contagem")
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig, clear_figure=True)
                card_close()

            with c2:
                card_open("02 — Distribuição (% do total)", "📈")
                fig, ax = plt.subplots()
                ax.bar(dist_df["Obesity"], dist_df["Percentual"])
                ax.set_title("Distribuição do nível de obesidade (% do total)")
                ax.set_xlabel("Obesity")
                ax.set_ylabel("%")
                ax.tick_params(axis="x", rotation=45)
                st.pyplot(fig, clear_figure=True)
                card_close()

            # =========================
            # 05 — Scatter Peso x Altura por nível
            # =========================
            section("05 — Dispersão Peso × Altura por nível", "🔎")

            if all(c in df_data.columns for c in ["Height", "Weight", "Obesity"]):
                card_open("05 — Height (X) x Weight (Y) por Obesity", "🧭")
                fig, ax = plt.subplots()
                for cls, g in df_data.dropna(subset=["Height", "Weight", "Obesity"]).groupby("Obesity"):
                    ax.scatter(g["Height"], g["Weight"], label=str(cls), alpha=0.6)
                ax.set_title("Dispersão: Peso × Altura por nível de obesidade")
                ax.set_xlabel("Height (m)")
                ax.set_ylabel("Weight (kg)")
                ax.legend(title="Obesity", bbox_to_anchor=(1.02, 1), loc="upper left")
                st.pyplot(fig, clear_figure=True)
                card_close()
            else:
                st.info("Não foi possível montar o gráfico 05 (precisa de Height, Weight e Obesity).")

            # =========================
            # 07 — Gender x Obesity (100% empilhado)
            # =========================
            section("07 — Gender × Obesity (100% empilhado)", "👥")

            if all(c in df_data.columns for c in ["Gender", "Obesity"]):
                ct = pd.crosstab(df_data["Gender"], df_data["Obesity"], normalize="index") * 100
                ct = ct.fillna(0)

                card_open("07 — Composição por gênero (100%)", "📚")
                fig, ax = plt.subplots()
                bottom = np.zeros(len(ct))
                x = np.arange(len(ct.index))

                for col in ct.columns:
                    vals = ct[col].values
                    ax.bar(x, vals, bottom=bottom, label=str(col))
                    bottom += vals

                ax.set_title("Gender × Obesity (100% empilhado)")
                ax.set_xlabel("Gender")
                ax.set_ylabel("% dentro de cada gênero")
                ax.set_xticks(x)
                ax.set_xticklabels([str(v) for v in ct.index], rotation=0)
                ax.legend(title="Obesity", bbox_to_anchor=(1.02, 1), loc="upper left")
                st.pyplot(fig, clear_figure=True)
                card_close()
            else:
                st.info("Não foi possível montar o gráfico 07 (precisa de Gender e Obesity).")

            # =========================
            # 08 — Heatmap faixa etária x Obesity (contagem)
            # =========================
            section("08 — Heatmap de faixa etária × Obesity (contagem)", "🧊")

            if all(c in df_data.columns for c in ["Age", "Obesity"]):
                # bins conforme seu requisito
                bins = [0, 18, 25, 30, 35, 40, 50, 100]
                labels = ["0–17", "18–24", "25–29", "30–34", "35–39", "40–49", "50–99"]
                faixa = pd.cut(df_data["Age"], bins=bins, labels=labels, right=False)
                heat = pd.crosstab(df_data["Obesity"], faixa)

                card_open("08 — Contagem por Obesity x Faixa Etária", "🔥")
                fig, ax = plt.subplots()
                im = ax.imshow(heat.values, aspect="auto")
                ax.set_title("Heatmap: faixa etária × Obesity (contagem)")
                ax.set_xlabel("Faixa etária")
                ax.set_ylabel("Obesity")

                ax.set_xticks(np.arange(len(heat.columns)))
                ax.set_xticklabels([str(c) for c in heat.columns], rotation=45, ha="right")
                ax.set_yticks(np.arange(len(heat.index)))
                ax.set_yticklabels([str(i) for i in heat.index])

                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig, clear_figure=True)
                card_close()
            else:
                st.info("Não foi possível montar o gráfico 08 (precisa de Age e Obesity).")

            # =========================
            # 14 — Heatmap correlação numéricas
            # =========================
            section("14 — Heatmap de correlação (numéricas)", "🧮")

            numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE", "BMI"]
            available = [c for c in numeric_cols if c in df_data.columns]

            if len(available) >= 2:
                corr = df_data[available].corr()

                card_open("14 — Correlação de Pearson", "🧾")
                fig, ax = plt.subplots()
                im = ax.imshow(corr.values, aspect="auto")
                ax.set_title("Heatmap de correlação (Pearson)")

                ax.set_xticks(np.arange(len(available)))
                ax.set_xticklabels(available, rotation=45, ha="right")
                ax.set_yticks(np.arange(len(available)))
                ax.set_yticklabels(available)

                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig, clear_figure=True)
                card_close()
            else:
                st.info("Não foi possível montar o gráfico 14 (faltam colunas numéricas suficientes).")

            # =========================
            # 16 — Radar perfil médio normalizado por Obesity
            # =========================
            section("16 — Radar: perfil médio normalizado por nível", "🕸️")

            radar_vars = ["FCVC", "NCP", "CH2O", "FAF", "TUE"]
            if all(c in df_data.columns for c in ["Obesity", *radar_vars]):
                means = df_data.groupby("Obesity")[radar_vars].mean(numeric_only=True)

                # min-max por variável (sobre as médias por classe)
                mins = means.min(axis=0)
                maxs = means.max(axis=0)
                denom = (maxs - mins).replace(0, np.nan)
                means_norm = (means - mins) / denom
                means_norm = means_norm.fillna(0)

                categories = radar_vars
                angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                angles += angles[:1]  # fecha o radar

                card_open("16 — Perfil médio normalizado (0–1)", "📡")
                fig = plt.figure()
                ax = plt.subplot(111, polar=True)

                for cls in means_norm.index:
                    values = means_norm.loc[cls].tolist()
                    values += values[:1]
                    ax.plot(angles, values, label=str(cls))
                    ax.fill(angles, values, alpha=0.08)

                ax.set_title("Radar: perfil médio normalizado por Obesity")
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(categories)
                ax.set_yticklabels([])

                ax.legend(bbox_to_anchor=(1.25, 1.05), loc="upper left", title="Obesity")
                st.pyplot(fig, clear_figure=True)
                card_close()
            else:
                st.info("Não foi possível montar o gráfico 16 (precisa de Obesity e FCVC/NCP/CH2O/FAF/TUE).")

        # (opcional) dataset
        st.write("")
        section("Dados (opcional)", "🗂️")
        with st.expander("Ver amostra do dataset"):
            st.dataframe(df_data.head(30), use_container_width=True)

# =========================
# TAB: Sobre
# =========================
with tab_sobre:
    section("Sobre o Sistema", "ℹ️")

    colA, colB = st.columns([1.4, 1], gap="large")
    with colA:
        card_open("Visão geral", "📘")
        st.write(
            "Este projeto integra um **modelo preditivo** para estimar o nível de obesidade e uma área de "
            "**visualização analítica**, com gráficos que ajudam a entender a distribuição das classes e relações "
            "entre variáveis do dataset. O foco é oferecer uma experiência objetiva e amigável para fins acadêmicos."
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
            - **Saída:** classes de obesidade  
            - **Modelo (pipeline):** `model/obesity_pipeline.joblib`  
            - **Base de dados:** `Obesity.csv`  
            """
        )
        card_close()
