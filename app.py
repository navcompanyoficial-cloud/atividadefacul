"""
Projeto BI + IA — Vendas Anuais de Asfalto por Município
Grupo 16 — Pergunta 16: Outliers por UF
Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")

st.set_page_config(
    page_title="BI Asfalto — Grupo 16",
    page_icon="🛣️",
    layout="wide",
)

# ──────────────────────────────────────────────
# ETL (cached)
# ──────────────────────────────────────────────
@st.cache_data
def load_and_transform():
    df_raw = pd.read_csv(
        "vendas-anuais-de-asfalto-por-municipio.csv",
        sep=";",
        encoding="utf-8",
    )
    df = df_raw.copy()
    df.columns = [
        "ANO", "GRANDE_REGIAO", "UF", "PRODUTO",
        "CODIGO_IBGE", "MUNICIPIO", "VENDAS_KG",
    ]
    df["ANO"] = df["ANO"].astype(int)
    df["CODIGO_IBGE"] = df["CODIGO_IBGE"].astype(str)
    df["VENDAS_KG"] = pd.to_numeric(df["VENDAS_KG"], errors="coerce").fillna(0).astype(int)
    df["MUNICIPIO"] = df["MUNICIPIO"].str.strip()
    df["UF"] = df["UF"].str.strip()
    df["GRANDE_REGIAO"] = df["GRANDE_REGIAO"].str.strip()
    df.drop(columns=["PRODUTO"], inplace=True)
    df.loc[df["VENDAS_KG"] < 0, "VENDAS_KG"] = 0
    df["METODOLOGIA"] = df["ANO"].apply(
        lambda x: "VENDAS+CONSUMO" if x <= 2006 else "SOMENTE VENDAS"
    )
    # Drop rows where UF is NaN
    df = df.dropna(subset=["UF", "GRANDE_REGIAO"]).copy()
    return df


df_full = load_and_transform()

# ──────────────────────────────────────────────
# Sidebar — filtros
# ──────────────────────────────────────────────
st.sidebar.title("Filtros")

ano_min, ano_max = int(df_full["ANO"].min()), int(df_full["ANO"].max())
ano_range = st.sidebar.slider(
    "Período (ANO)",
    min_value=ano_min,
    max_value=ano_max,
    value=(ano_min, ano_max),
)

regioes_disponiveis = sorted(df_full["GRANDE_REGIAO"].unique())
regioes_sel = st.sidebar.multiselect(
    "Região",
    options=regioes_disponiveis,
    default=regioes_disponiveis,
)

ufs_disponiveis = sorted(
    df_full[df_full["GRANDE_REGIAO"].isin(regioes_sel)]["UF"].unique()
)
ufs_sel = st.sidebar.multiselect(
    "UF",
    options=ufs_disponiveis,
    default=ufs_disponiveis,
)

# Dados filtrados
df = df_full[
    (df_full["ANO"] >= ano_range[0])
    & (df_full["ANO"] <= ano_range[1])
    & (df_full["GRANDE_REGIAO"].isin(regioes_sel))
    & (df_full["UF"].isin(ufs_sel))
].copy()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Registros filtrados: **{len(df):,}** de {len(df_full):,}"
)

# ──────────────────────────────────────────────
# Título
# ──────────────────────────────────────────────
st.title("🛣️ BI + IA — Vendas Anuais de Asfalto por Município")
st.markdown(
    "**Grupo 16 — Pergunta 16:** Outliers por UF &nbsp;|&nbsp; "
    f"Período: {ano_range[0]}–{ano_range[1]}"
)

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visão Geral Brasil",
    "🗺️ Recorte Geográfico",
    "📈 Série Temporal",
    "📋 Métricas",
    "🔍 Pergunta 16 — Outliers",
])

# ══════════════════════════════════════════════
# TAB 1 — Visão Geral Brasil
# ══════════════════════════════════════════════
with tab1:
    st.header("Visão Geral Brasil")

    vendas_ano = df.groupby("ANO")["VENDAS_KG"].sum().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1) Vendas totais por ano
    axes[0, 0].bar(vendas_ano["ANO"], vendas_ano["VENDAS_KG"] / 1e9, color="steelblue")
    axes[0, 0].axvline(x=2006.5, color="red", linestyle="--", alpha=0.7, label="Mudança metod. (2007)")
    axes[0, 0].set_title("Vendas Totais por Ano")
    axes[0, 0].set_ylabel("Bilhões kg")
    axes[0, 0].legend(fontsize=8)

    # 2) Vendas acumuladas por região
    vendas_por_regiao = df.groupby("GRANDE_REGIAO")["VENDAS_KG"].sum().sort_values(ascending=True)
    axes[0, 1].barh(vendas_por_regiao.index, vendas_por_regiao.values / 1e9, color="coral")
    axes[0, 1].set_title("Vendas Acumuladas por Região")
    axes[0, 1].set_xlabel("Bilhões kg")

    # 3) Top 10 UFs
    vendas_por_uf = df.groupby("UF")["VENDAS_KG"].sum().nlargest(10).sort_values(ascending=True)
    axes[1, 0].barh(vendas_por_uf.index, vendas_por_uf.values / 1e9, color="seagreen")
    axes[1, 0].set_title("Top 10 UFs por Vendas Acumuladas")
    axes[1, 0].set_xlabel("Bilhões kg")

    # 4) Nº municípios com vendas por ano
    mun_por_ano = (
        df[df["VENDAS_KG"] > 0]
        .groupby("ANO")["CODIGO_IBGE"]
        .nunique()
        .reset_index()
    )
    axes[1, 1].plot(
        mun_por_ano["ANO"], mun_por_ano["CODIGO_IBGE"],
        marker="o", color="purple", markersize=4,
    )
    axes[1, 1].set_title("Nº de Municípios com Vendas > 0 por Ano")
    axes[1, 1].set_ylabel("Qtd. Municípios")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ══════════════════════════════════════════════
# TAB 2 — Recorte Geográfico
# ══════════════════════════════════════════════
with tab2:
    st.header("Recorte Geográfico")

    col_a, col_b = st.columns(2)

    # 1) Evolução por região
    with col_a:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        for regiao in sorted(df["GRANDE_REGIAO"].dropna().unique()):
            dados_r = df[df["GRANDE_REGIAO"] == regiao].groupby("ANO")["VENDAS_KG"].sum()
            ax1.plot(
                dados_r.index, dados_r.values / 1e9,
                marker=".", label=str(regiao).replace("REGIÃO ", ""),
            )
        ax1.set_title("Evolução de Vendas por Região")
        ax1.set_ylabel("Bilhões kg")
        ax1.legend(title="Região", fontsize=7)
        ax1.axvline(x=2006.5, color="red", linestyle="--", alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close(fig1)

    # 2) Heatmap UF x Década
    with col_b:
        df_temp = df.copy()
        df_temp["DECADA"] = (df_temp["ANO"] // 10) * 10
        heatmap_data = (
            df_temp.groupby(["UF", "DECADA"])["VENDAS_KG"]
            .sum()
            .unstack(fill_value=0)
            / 1e9
        )
        heatmap_data = heatmap_data.loc[
            heatmap_data.sum(axis=1).nlargest(15).index
        ]
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax2)
        ax2.set_title("Top 15 UFs — Vendas por Década (bi kg)")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close(fig2)

# ══════════════════════════════════════════════
# TAB 3 — Série Temporal
# ══════════════════════════════════════════════
with tab3:
    st.header("Série Temporal & Insights")

    vendas_ano_ts = df.groupby("ANO")["VENDAS_KG"].sum().reset_index()
    vendas_ano_ts["MEDIA_MOVEL_3A"] = (
        vendas_ano_ts["VENDAS_KG"].rolling(window=3, center=False).mean()
    )
    vendas_ano_ts["VARIACAO_PCT"] = vendas_ano_ts["VENDAS_KG"].pct_change() * 100

    col1, col2 = st.columns(2)

    with col1:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        ax3.plot(
            vendas_ano_ts["ANO"], vendas_ano_ts["VENDAS_KG"] / 1e9,
            marker="o", markersize=4, label="Vendas",
        )
        ax3.plot(
            vendas_ano_ts["ANO"], vendas_ano_ts["MEDIA_MOVEL_3A"] / 1e9,
            color="red", linewidth=2, label="Média Móvel 3A",
        )
        ax3.axvline(x=2006.5, color="gray", linestyle="--", alpha=0.7, label="Mudança metod.")
        ax3.set_title("Série Temporal + Média Móvel 3 Anos")
        ax3.set_ylabel("Bilhões kg")
        ax3.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with col2:
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        cores = [
            "green" if v >= 0 else "red"
            for v in vendas_ano_ts["VARIACAO_PCT"].fillna(0)
        ]
        ax4.bar(
            vendas_ano_ts["ANO"],
            vendas_ano_ts["VARIACAO_PCT"].fillna(0),
            color=cores, alpha=0.8,
        )
        ax4.axhline(y=0, color="black", linewidth=0.5)
        ax4.set_title("Variação % Ano a Ano")
        ax4.set_ylabel("Variação (%)")
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

# ══════════════════════════════════════════════
# TAB 4 — Métricas
# ══════════════════════════════════════════════
with tab4:
    st.header("Métricas")

    # KPIs — usando dados COMPLETOS (sem filtro)
    vendas_ano_full = df_full.groupby("ANO")["VENDAS_KG"].sum().reset_index()
    total_vendas_brasil = df_full["VENDAS_KG"].sum()
    ultimo_ano = vendas_ano_full["ANO"].max()
    penultimo_ano = ultimo_ano - 1
    vendas_ultimo = vendas_ano_full[vendas_ano_full["ANO"] == ultimo_ano]["VENDAS_KG"].values[0]
    vendas_penultimo = vendas_ano_full[vendas_ano_full["ANO"] == penultimo_ano]["VENDAS_KG"].values[0]
    variacao = ((vendas_ultimo - vendas_penultimo) / vendas_penultimo) * 100

    k1, k2, k3 = st.columns(3)
    k1.metric("Total Vendas (1992-2024)", f"{total_vendas_brasil / 1e9:.2f} bi kg")
    k2.metric(f"Vendas {ultimo_ano}", f"{vendas_ultimo / 1e9:.2f} bi kg")
    k3.metric(f"Variação {penultimo_ano}→{ultimo_ano}", f"{variacao:+.1f}%")

    st.markdown("---")

    # Gráficos de métricas — usando dados FILTRADOS
    col_m1, col_m2 = st.columns(2)

    # Métrica 3 — Participação % por região (area chart)
    with col_m1:
        st.subheader("Participação % por Região")
        vendas_regiao_ano = df.groupby(["ANO", "GRANDE_REGIAO"])["VENDAS_KG"].sum().reset_index()
        total_por_ano = df.groupby("ANO")["VENDAS_KG"].sum().reset_index().rename(
            columns={"VENDAS_KG": "TOTAL_ANO"}
        )
        vendas_regiao_ano = vendas_regiao_ano.merge(total_por_ano, on="ANO")
        vendas_regiao_ano["PCT"] = (
            vendas_regiao_ano["VENDAS_KG"] / vendas_regiao_ano["TOTAL_ANO"]
        ) * 100

        pivot_regiao = vendas_regiao_ano.pivot(
            index="ANO", columns="GRANDE_REGIAO", values="PCT"
        )

        fig5, ax5 = plt.subplots(figsize=(8, 5))
        pivot_regiao.plot(kind="area", stacked=True, alpha=0.8, colormap="Set2", ax=ax5)
        ax5.set_ylabel("Participação (%)")
        ax5.set_title("Participação % por Região")
        ax5.legend(title="Região", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

    # Métrica 4 — Participação % top 10 UFs
    with col_m2:
        st.subheader("Participação % Top 10 UFs")
        vendas_uf_ano = df.groupby(["ANO", "UF"])["VENDAS_KG"].sum().reset_index()
        vendas_uf_ano = vendas_uf_ano.merge(total_por_ano, on="ANO")
        vendas_uf_ano["PCT"] = (
            vendas_uf_ano["VENDAS_KG"] / vendas_uf_ano["TOTAL_ANO"]
        ) * 100
        top10_ufs = df.groupby("UF")["VENDAS_KG"].sum().nlargest(10).index.tolist()
        dados_top10 = vendas_uf_ano[vendas_uf_ano["UF"].isin(top10_ufs)]

        fig6, ax6 = plt.subplots(figsize=(8, 5))
        for uf in top10_ufs:
            d = dados_top10[dados_top10["UF"] == uf]
            ax6.plot(d["ANO"], d["PCT"], marker=".", label=uf)
        ax6.set_ylabel("Participação (%)")
        ax6.set_title("Participação % Top 10 UFs")
        ax6.legend(title="UF", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)

    col_m3, col_m4 = st.columns(2)

    # Métrica 5 — Ranking top 10 UFs
    with col_m3:
        st.subheader("Evolução do Ranking Top 10 UFs")
        vendas_uf_ano["RANKING"] = (
            vendas_uf_ano.groupby("ANO")["VENDAS_KG"]
            .rank(ascending=False, method="min")
            .astype(int)
        )
        ranking_top10 = vendas_uf_ano[vendas_uf_ano["UF"].isin(top10_ufs)].pivot(
            index="ANO", columns="UF", values="RANKING"
        )

        fig7, ax7 = plt.subplots(figsize=(8, 5))
        for uf in top10_ufs:
            if uf in ranking_top10.columns:
                ax7.plot(ranking_top10.index, ranking_top10[uf], marker=".", label=uf, linewidth=2)
        ax7.invert_yaxis()
        ax7.set_yticks(range(1, 11))
        ax7.set_title("Evolução do Ranking")
        ax7.set_ylabel("Posição")
        ax7.legend(title="UF", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close(fig7)

    # Métrica 6 — Média Móvel 3 anos
    with col_m4:
        st.subheader("Vendas Brasil + Média Móvel 3 Anos")
        vendas_ano_m = df.groupby("ANO")["VENDAS_KG"].sum().reset_index()
        vendas_ano_m["MM3"] = vendas_ano_m["VENDAS_KG"].rolling(window=3).mean()

        fig8, ax8 = plt.subplots(figsize=(8, 5))
        ax8.bar(
            vendas_ano_m["ANO"], vendas_ano_m["VENDAS_KG"] / 1e9,
            alpha=0.5, color="steelblue", label="Vendas Anuais",
        )
        ax8.plot(
            vendas_ano_m["ANO"], vendas_ano_m["MM3"] / 1e9,
            color="red", linewidth=2.5, label="Média Móvel 3A",
        )
        ax8.axvline(x=2006.5, color="gray", linestyle="--", alpha=0.7, label="Mudança metod.")
        ax8.set_ylabel("Bilhões kg")
        ax8.set_title("Média Móvel 3 Anos")
        ax8.legend(fontsize=8)
        plt.tight_layout()
        st.pyplot(fig8)
        plt.close(fig8)

# ══════════════════════════════════════════════
# TAB 5 — Pergunta 16: Outliers por UF
# ══════════════════════════════════════════════
with tab5:
    st.header("Pergunta 16 — Outliers por UF")
    st.markdown(
        "> *Para cada UF, analise distribuição de vendas municipais por ano "
        "(mediana, dispersão). Aplique detecção de outliers para achar "
        "municípios 'fora da curva' e quantifique quanto eles explicam do total da UF.*"
    )
    st.markdown(
        "**Em outras palavras:** dentro de cada estado, a maioria dos municípios compra "
        "pouco asfalto. Mas existem algumas cidades que compram **muito mais** que as outras "
        "— essas são os \"outliers\" (fora da curva). Nesta seção, identificamos quem são "
        "esses municípios e mostramos o quanto eles representam do total de vendas do estado."
    )

    # ---------- helper: detectar outliers IQR ----------
    def detectar_outliers_iqr(grupo):
        Q1 = grupo["VENDAS_KG"].quantile(0.25)
        Q3 = grupo["VENDAS_KG"].quantile(0.75)
        IQR = Q3 - Q1
        limite_sup = Q3 + 1.5 * IQR
        limite_inf = Q1 - 1.5 * IQR
        grupo = grupo.copy()
        grupo["IS_OUTLIER"] = (grupo["VENDAS_KG"] < limite_inf) | (
            grupo["VENDAS_KG"] > limite_sup
        )
        grupo["LIMITE_SUPERIOR"] = limite_sup
        return grupo

    df_outliers = df.groupby(["ANO", "UF"], group_keys=False).apply(detectar_outliers_iqr)

    total_registros = len(df_outliers)
    total_outliers = int(df_outliers["IS_OUTLIER"].sum())

    st.markdown("### 6.1 Como as vendas se distribuem dentro de cada estado?")
    st.markdown(
        "Abaixo mostramos, para cada UF, dois indicadores:\n"
        "- **Mediana**: o valor \"do meio\" das vendas municipais — metade dos municípios vende menos que isso, metade vende mais.\n"
        "- **Coeficiente de Variação (CV%)**: mede o quanto as vendas variam entre os municípios. "
        "Quanto maior o CV, maior a diferença entre os municípios que vendem pouco e os que vendem muito."
    )

    # Estatísticas por UF/Ano
    stats_uf_ano = (
        df.groupby(["ANO", "UF"])["VENDAS_KG"]
        .agg(["count", "sum", "mean", "median", "std"])
        .reset_index()
    )
    stats_uf_ano.columns = [
        "ANO", "UF", "QTD_MUNICIPIOS", "TOTAL_VENDAS",
        "MEDIA", "MEDIANA", "DESVIO_PADRAO",
    ]
    stats_uf_ano["CV"] = (
        (stats_uf_ano["DESVIO_PADRAO"] / stats_uf_ano["MEDIA"])
        .replace([np.inf, -np.inf], 0) * 100
    )

    resumo_uf = (
        stats_uf_ano.groupby("UF")
        .agg(
            MEDIANA_MEDIA=("MEDIANA", "mean"),
            CV_MEDIO=("CV", "mean"),
            TOTAL_GERAL=("TOTAL_VENDAS", "sum"),
        )
        .sort_values("TOTAL_GERAL", ascending=False)
        .reset_index()
    )

    top15 = resumo_uf.head(15)

    fig9, axes9 = plt.subplots(1, 2, figsize=(14, 5))
    axes9[0].barh(top15["UF"], top15["MEDIANA_MEDIA"] / 1e3, color="steelblue")
    axes9[0].set_xlabel("Mediana média (mil kg)")
    axes9[0].set_title("Mediana Média de Vendas Municipais por UF")
    axes9[0].invert_yaxis()

    axes9[1].barh(top15["UF"], top15["CV_MEDIO"], color="coral")
    axes9[1].set_xlabel("Coeficiente de Variação Médio (%)")
    axes9[1].set_title("Dispersão (CV%) das Vendas Municipais por UF")
    axes9[1].invert_yaxis()

    plt.suptitle("Distribuição por UF", fontsize=13, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig9)
    plt.close(fig9)

    # ---------- 6.2 Detecção de outliers — resumo ----------
    st.markdown("### 6.2 Quantos outliers existem nos dados?")
    st.markdown(
        "Usamos o **método IQR (Intervalo Interquartil)** para identificar outliers. "
        "Funciona assim: para cada estado em cada ano, calculamos a faixa \"normal\" de vendas "
        "entre os municípios. Qualquer município que venda **muito acima** dessa faixa é "
        "considerado um **outlier** — ou seja, está \"fora da curva\".\n\n"
        "Na prática, são geralmente capitais e grandes polos industriais que compram "
        "volumes de asfalto muito superiores ao restante dos municípios do estado."
    )

    o1, o2, o3 = st.columns(3)
    o1.metric("Total de registros", f"{total_registros:,}")
    o2.metric("Outliers detectados", f"{total_outliers:,}")
    pct_outliers = (total_outliers / total_registros * 100) if total_registros else 0
    o3.metric("% Outliers", f"{pct_outliers:.1f}%")

    # ---------- 6.3 Impacto dos outliers ----------
    st.markdown("### 6.3 Quanto do total de cada estado vem desses municípios \"fora da curva\"?")
    st.markdown(
        "O gráfico abaixo mostra, para cada UF, **qual percentual das vendas totais do estado "
        "é explicado apenas pelos municípios outliers**.\n\n"
        "- Barra **vermelha** (>50%): mais da metade das vendas vem de poucos municípios — altíssima concentração.\n"
        "- Barra **laranja** (30-50%): concentração moderada.\n"
        "- Barra **verde** (<30%): vendas mais bem distribuídas entre os municípios."
    )

    impacto = (
        df_outliers.groupby(["ANO", "UF"])
        .apply(
            lambda g: pd.Series({
                "TOTAL_UF": g["VENDAS_KG"].sum(),
                "QTD_MUNICIPIOS": len(g),
                "QTD_OUTLIERS": int(g["IS_OUTLIER"].sum()),
                "VENDAS_OUTLIERS": g.loc[g["IS_OUTLIER"], "VENDAS_KG"].sum(),
            })
        )
        .reset_index()
    )
    impacto["PCT_OUTLIERS_VENDAS"] = (
        (impacto["VENDAS_OUTLIERS"] / impacto["TOTAL_UF"] * 100).fillna(0)
    )

    resumo_outliers_uf = (
        impacto.groupby("UF")
        .agg(
            MEDIA_PCT_VENDAS_OUTLIERS=("PCT_OUTLIERS_VENDAS", "mean"),
            MEDIA_QTD_OUTLIERS=("QTD_OUTLIERS", "mean"),
            TOTAL_VENDAS_UF=("TOTAL_UF", "sum"),
        )
        .sort_values("MEDIA_PCT_VENDAS_OUTLIERS", ascending=False)
        .reset_index()
    )

    dados_plot = resumo_outliers_uf.sort_values("MEDIA_PCT_VENDAS_OUTLIERS", ascending=True)
    cores_bar = [
        "#e74c3c" if v > 50 else "#f39c12" if v > 30 else "#2ecc71"
        for v in dados_plot["MEDIA_PCT_VENDAS_OUTLIERS"]
    ]

    fig10, ax10 = plt.subplots(figsize=(12, 7))
    ax10.barh(dados_plot["UF"], dados_plot["MEDIA_PCT_VENDAS_OUTLIERS"], color=cores_bar)
    ax10.set_xlabel("% Médio das Vendas da UF Explicado por Outliers")
    ax10.set_title(
        "Concentração: Quanto os Municípios Outliers Representam por UF",
        fontsize=12, fontweight="bold",
    )
    ax10.axvline(x=50, color="red", linestyle="--", alpha=0.5, label=">50% (alta)")
    ax10.axvline(x=30, color="orange", linestyle="--", alpha=0.5, label=">30% (média)")
    ax10.legend(fontsize=8)
    for i, (uf, v) in enumerate(
        zip(dados_plot["UF"], dados_plot["MEDIA_PCT_VENDAS_OUTLIERS"])
    ):
        ax10.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig10)
    plt.close(fig10)

    # ---------- 6.4 Evolução temporal ----------
    st.markdown("### 6.4 Essa concentração mudou ao longo dos anos?")
    st.markdown(
        "Cada gráfico mostra como o percentual de vendas concentrado nos municípios outliers "
        "variou de 1992 a 2024. Mostramos o Brasil como um todo e os 5 estados que mais vendem asfalto.\n\n"
        "A linha vermelha tracejada marca 2007, quando houve uma mudança na metodologia de coleta dos dados "
        "(antes incluía consumo próprio, depois só vendas)."
    )

    top5_ufs = df.groupby("UF")["VENDAS_KG"].sum().nlargest(5).index.tolist()

    fig11, axes11 = plt.subplots(2, 3, figsize=(16, 8))
    axes_flat = axes11.flatten()

    # Brasil
    brasil_imp = (
        impacto.groupby("ANO")
        .agg(VENDAS_OUTLIERS=("VENDAS_OUTLIERS", "sum"), TOTAL=("TOTAL_UF", "sum"))
        .reset_index()
    )
    brasil_imp["PCT"] = brasil_imp["VENDAS_OUTLIERS"] / brasil_imp["TOTAL"] * 100

    axes_flat[0].plot(
        brasil_imp["ANO"], brasil_imp["PCT"], marker="o", color="black", markersize=4
    )
    axes_flat[0].set_title("BRASIL", fontweight="bold")
    axes_flat[0].set_ylabel("% Vendas por Outliers")
    axes_flat[0].axvline(x=2006.5, color="red", linestyle="--", alpha=0.3)

    for idx, uf in enumerate(top5_ufs):
        dados_e = impacto[impacto["UF"] == uf]
        axes_flat[idx + 1].plot(
            dados_e["ANO"], dados_e["PCT_OUTLIERS_VENDAS"],
            marker="o", markersize=4, color="steelblue",
        )
        axes_flat[idx + 1].set_title(uf, fontweight="bold")
        axes_flat[idx + 1].axvline(x=2006.5, color="red", linestyle="--", alpha=0.3)

    plt.suptitle(
        "Evolução do % de Vendas Explicado por Outliers",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    st.pyplot(fig11)
    plt.close(fig11)

    # ---------- 6.5 Boxplot top 10 UFs ----------
    st.markdown("### 6.5 Visualizando a dispersão: Boxplot das Top 10 UFs")
    st.markdown(
        "O boxplot (\"gráfico de caixa\") é uma das melhores formas de visualizar a distribuição dos dados. "
        "Para cada estado:\n"
        "- A **caixa** representa onde estão os 50% centrais dos municípios (vendas \"normais\").\n"
        "- A **linha no meio** da caixa é a mediana.\n"
        "- Os **pontinhos** fora da caixa são os **outliers** — municípios que vendem muito acima do padrão.\n\n"
        "Usamos escala logarítmica (log10) porque a diferença entre o menor e o maior município "
        "é enorme — sem essa escala, não seria possível enxergar os menores."
    )

    top10_ufs_q16 = df.groupby("UF")["VENDAS_KG"].sum().nlargest(10).index.tolist()
    dados_box = df[df["UF"].isin(top10_ufs_q16)].copy()
    dados_box["VENDAS_LOG"] = np.log10(dados_box["VENDAS_KG"].clip(lower=1))
    order_box = (
        df[df["UF"].isin(top10_ufs_q16)]
        .groupby("UF")["VENDAS_KG"]
        .median()
        .sort_values(ascending=False)
        .index
    )

    fig12, ax12 = plt.subplots(figsize=(14, 6))
    sns.boxplot(
        data=dados_box, x="UF", y="VENDAS_LOG",
        order=order_box, ax=ax12, fliersize=2,
    )
    ax12.set_title(
        "Distribuição de Vendas Municipais por UF (escala log10)",
        fontsize=12, fontweight="bold",
    )
    ax12.set_ylabel("log10(Vendas em kg)")
    plt.tight_layout()
    st.pyplot(fig12)
    plt.close(fig12)

    # ---------- Tabela top 30 municípios outliers ----------
    st.markdown("### 6.6 Quem são os municípios \"fora da curva\"?")
    st.markdown(
        "A tabela abaixo lista os 30 municípios que **mais vezes foram identificados como outlier** "
        "ao longo dos anos. A coluna \"Vezes Outlier\" mostra em quantos anos aquele município "
        "apareceu fora da curva — quanto maior, mais consistentemente ele domina as vendas no seu estado."
    )

    freq_outlier = (
        df_outliers[df_outliers["IS_OUTLIER"]]
        .groupby(["UF", "MUNICIPIO"])
        .agg(
            VEZES_OUTLIER=("IS_OUTLIER", "sum"),
            VENDAS_TOTAL=("VENDAS_KG", "sum"),
            VENDAS_MEDIA=("VENDAS_KG", "mean"),
        )
        .reset_index()
        .sort_values("VEZES_OUTLIER", ascending=False)
    )

    top30 = freq_outlier.head(30).copy()
    top30["VENDAS_TOTAL"] = top30["VENDAS_TOTAL"].apply(lambda x: f"{x / 1e6:,.1f} M kg")
    top30["VENDAS_MEDIA"] = top30["VENDAS_MEDIA"].apply(lambda x: f"{x / 1e6:,.2f} M kg")
    st.dataframe(
        top30[["UF", "MUNICIPIO", "VEZES_OUTLIER", "VENDAS_TOTAL", "VENDAS_MEDIA"]],
        use_container_width=True,
        hide_index=True,
    )

    # ---------- Resumo final por UF ----------
    st.markdown("### 6.7 Resumo consolidado por estado")
    st.markdown(
        "Tabela com os 15 maiores estados em vendas de asfalto, mostrando:\n"
        "- **% Médio Vendas por Outliers**: em média, quanto das vendas do estado vem dos municípios fora da curva.\n"
        "- **Média Outliers/Ano**: quantos municípios são classificados como outlier por ano naquele estado.\n"
        "- **Total Vendas UF**: volume total de asfalto vendido no estado em todo o período."
    )

    resumo_final = resumo_outliers_uf.sort_values("TOTAL_VENDAS_UF", ascending=False).head(15)
    resumo_final_display = resumo_final.copy()
    resumo_final_display["MEDIA_PCT_VENDAS_OUTLIERS"] = resumo_final_display[
        "MEDIA_PCT_VENDAS_OUTLIERS"
    ].apply(lambda x: f"{x:.1f}%")
    resumo_final_display["MEDIA_QTD_OUTLIERS"] = resumo_final_display[
        "MEDIA_QTD_OUTLIERS"
    ].apply(lambda x: f"{x:.1f}")
    resumo_final_display["TOTAL_VENDAS_UF"] = resumo_final_display[
        "TOTAL_VENDAS_UF"
    ].apply(lambda x: f"{x / 1e9:.2f} bi kg")

    resumo_final_display.columns = [
        "UF", "% Médio Vendas por Outliers",
        "Média Outliers/Ano", "Total Vendas UF",
    ]
    st.dataframe(resumo_final_display, use_container_width=True, hide_index=True)

    st.markdown("### Conclusões — O que aprendemos com essa análise?")
    st.markdown(
        """
**1. O mercado de asfalto é extremamente concentrado em poucos municípios.**
Em quase todos os estados, um pequeno grupo de cidades (geralmente a capital e grandes polos
industriais) compra a maior parte do asfalto. Isso faz sentido: cidades maiores têm mais
ruas, estradas e obras de infraestrutura, então naturalmente demandam mais asfalto.

**2. As mesmas cidades dominam há mais de 30 anos.**
Municípios como São Paulo, Rio de Janeiro, Belo Horizonte, Salvador e Goiânia aparecem
como outliers em praticamente todos os anos analisados (1992 a 2024). Isso mostra que
a estrutura do mercado de asfalto no Brasil é estável — não houve grandes mudanças
de quem compra mais.

**3. Em alguns estados, mais da metade das vendas vem de 2 ou 3 cidades.**
Estados menores ou com economia concentrada na capital apresentam altíssima dependência
de poucos municípios. Já estados como SP e MG, que têm mais cidades grandes, possuem
vendas um pouco mais distribuídas.

**4. A mudança de metodologia em 2007 não alterou esse padrão.**
Em 2007, os dados passaram a registrar somente vendas (antes incluíam consumo próprio).
Isso mudou os volumes absolutos, mas a concentração em poucos municípios continuou igual.

**Como fizemos a análise?**
Utilizamos o método estatístico **IQR (Intervalo Interquartil)**, aplicado para cada
estado em cada ano. Esse método calcula a faixa normal de vendas e identifica automaticamente
quais municípios estão muito acima dessa faixa — os chamados outliers.
"""
    )
