import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
tabela = pd.read_csv("advertising.csv")
print(tabela)
print(tabela.info())

# 1 Análise Exploratória (correlações)
print(tabela.corr()) 
# 1.1 Criando um grafico de calor
sns.heatmap(tabela.corr(), cmap='Blues', annot=True)
plt.show()
# 1.2 Separando em dados de treino e dados de teste
y = tabela["Vendas"]
x = tabela[["TV", "Radio", "Jornal"]]

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)


# 2 Modelagem + algoritimos (IA: Regrssão linear e Árvore de Decisão)
# Importando a IA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Criando a IA
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# Treina IA
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

# Previsão
from sklearn import metrics
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
from sklearn.metrics import r2_score
print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvoredecisao))

# Visualização Gráfica das previsões
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsão Árvore Decisão"] = previsao_arvoredecisao
tabela_auxiliar["Previsão Regressão Linear"] = previsao_regressaolinear
print(tabela_auxiliar)

sns.lineplot(data=tabela_auxiliar)
plt.show()

# Fazendo uma nova previsão
nova_tabela = pd.read_csv("novos.csv")
print(nova_tabela)
previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)

# Verifia se estamos investindo no produto correto
plt.figure(figsize=(15, 5))
sns.barplot(x=x_treino.columns, y=modelo_arvoredecisao.feature_importances_)
plt.show()
print(tabela[["Radio", "Jornal"]].sum())
