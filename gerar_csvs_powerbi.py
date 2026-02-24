"""
Script para gerar os CSVs do Modelo Estrela para importar no Power BI.
Gera: DimTempo, DimRegiao, DimUF, DimLocalidade, FatoVendasAsfalto
"""
import pandas as pd
import numpy as np
import os

# Pasta de saída
pasta = os.path.dirname(os.path.abspath(__file__))
pasta_powerbi = os.path.join(pasta, 'powerbi_dados')
os.makedirs(pasta_powerbi, exist_ok=True)

print('Carregando dataset...')
df = pd.read_csv(
    os.path.join(pasta, 'vendas-anuais-de-asfalto-por-municipio.csv'),
    sep=';',
    encoding='utf-8'
)

# === ETL ===
df.columns = ['ANO', 'GRANDE_REGIAO', 'UF', 'PRODUTO', 'CODIGO_IBGE', 'MUNICIPIO', 'VENDAS_KG']
df['ANO'] = df['ANO'].astype(int)
df['CODIGO_IBGE'] = df['CODIGO_IBGE'].astype(str)
df['VENDAS_KG'] = pd.to_numeric(df['VENDAS_KG'], errors='coerce').fillna(0).astype(int)
df['MUNICIPIO'] = df['MUNICIPIO'].str.strip()
df['UF'] = df['UF'].str.strip()
df['GRANDE_REGIAO'] = df['GRANDE_REGIAO'].str.strip()
df.drop(columns=['PRODUTO'], inplace=True)
df.loc[df['VENDAS_KG'] < 0, 'VENDAS_KG'] = 0

print(f'Registros: {len(df)}')

# === DIMENSÃO TEMPO ===
dim_tempo = df[['ANO']].drop_duplicates().sort_values('ANO').reset_index(drop=True)
dim_tempo['ID_TEMPO'] = dim_tempo.index + 1
dim_tempo['DECADA'] = (dim_tempo['ANO'] // 10) * 10
dim_tempo['METODOLOGIA'] = dim_tempo['ANO'].apply(
    lambda x: 'VENDAS+CONSUMO' if x <= 2006 else 'SOMENTE VENDAS'
)
dim_tempo = dim_tempo[['ID_TEMPO', 'ANO', 'DECADA', 'METODOLOGIA']]

# === DIMENSÃO REGIÃO ===
dim_regiao = df[['GRANDE_REGIAO']].drop_duplicates().sort_values('GRANDE_REGIAO').reset_index(drop=True)
dim_regiao['ID_REGIAO'] = dim_regiao.index + 1
dim_regiao = dim_regiao[['ID_REGIAO', 'GRANDE_REGIAO']]

# === DIMENSÃO UF ===
dim_uf = df[['UF', 'GRANDE_REGIAO']].drop_duplicates().sort_values('UF').reset_index(drop=True)
dim_uf['ID_UF'] = dim_uf.index + 1
dim_uf = dim_uf.merge(dim_regiao[['GRANDE_REGIAO', 'ID_REGIAO']], on='GRANDE_REGIAO')
dim_uf = dim_uf[['ID_UF', 'UF', 'ID_REGIAO', 'GRANDE_REGIAO']]

# === DIMENSÃO LOCALIDADE ===
dim_localidade = df[['CODIGO_IBGE', 'MUNICIPIO', 'UF']].drop_duplicates(
    subset=['CODIGO_IBGE']
).sort_values('CODIGO_IBGE').reset_index(drop=True)
dim_localidade['ID_LOCALIDADE'] = dim_localidade.index + 1
dim_localidade = dim_localidade.merge(dim_uf[['UF', 'ID_UF']], on='UF')
dim_localidade = dim_localidade[['ID_LOCALIDADE', 'CODIGO_IBGE', 'MUNICIPIO', 'UF', 'ID_UF']]

# === TABELA FATO ===
fato = df.merge(dim_tempo[['ANO', 'ID_TEMPO']], on='ANO')
fato = fato.merge(dim_localidade[['CODIGO_IBGE', 'ID_LOCALIDADE']], on='CODIGO_IBGE')
fato = fato.merge(dim_uf[['UF', 'ID_UF']], on='UF')
fato = fato.merge(dim_regiao[['GRANDE_REGIAO', 'ID_REGIAO']], on='GRANDE_REGIAO')
fato_vendas = fato[['ID_TEMPO', 'ID_LOCALIDADE', 'ID_UF', 'ID_REGIAO', 'VENDAS_KG']].copy()

# === EXPORTAR CSVs ===
dim_tempo.to_csv(os.path.join(pasta_powerbi, 'DimTempo.csv'), index=False, sep=';', encoding='utf-8-sig')
dim_regiao.to_csv(os.path.join(pasta_powerbi, 'DimRegiao.csv'), index=False, sep=';', encoding='utf-8-sig')
dim_uf.to_csv(os.path.join(pasta_powerbi, 'DimUF.csv'), index=False, sep=';', encoding='utf-8-sig')
dim_localidade.to_csv(os.path.join(pasta_powerbi, 'DimLocalidade.csv'), index=False, sep=';', encoding='utf-8-sig')
fato_vendas.to_csv(os.path.join(pasta_powerbi, 'FatoVendasAsfalto.csv'), index=False, sep=';', encoding='utf-8-sig')

print('\n=== CSVs gerados na pasta powerbi_dados/ ===')
print(f'  DimTempo.csv         ({len(dim_tempo)} registros)')
print(f'  DimRegiao.csv        ({len(dim_regiao)} registros)')
print(f'  DimUF.csv            ({len(dim_uf)} registros)')
print(f'  DimLocalidade.csv    ({len(dim_localidade)} registros)')
print(f'  FatoVendasAsfalto.csv ({len(fato_vendas)} registros)')
print('\nPronto! Agora importe esses CSVs no Power BI Desktop.')
