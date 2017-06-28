# Exercício 6


Os dados ex6-train.csv representam um problema de regressão. A ultima coluna coluna é o valor a ser aproximado.

Use varias técnicas para fazer a regressão ( SVM regressão, gbm, rf, redes neurais, knn, gaussian regression, e outras mesmo que não tenhamos visto em aula). A metrica sera MAE - erro absoluto médio (nao erro quadrado!).

O relatório deve conter a sua exploracao de pelo menos 2 das tecnicas de regressao (mais o pre-processamento, se for o caso), os hiperparametros tentados e o erro (MAE). O relatorio valerá 60% da nota.

Rode o seu melhor regressor nestes dados, e submita também o resultado do valor predito, um por linha na mesma ordem dos dados. Note que é um resultado por linha.

Eu avaliarei o MAE do seu regressor nos resultados corretos. 40% restante da nota sera competitiva: as submissoes no topo 10% com menos MAE receberão o 10 nessa parte e as submissões nos últimos 10% (maiores MAE) receberão 0.

Note que o dataset é grande (50.000 linhas e 77 atributos). Talvez nao valha a pena fazer a exploracao dos algoritimos e a busca de hiperparametros no dataset inteiro.

ATENCAO: é preciso submeter 2 arquivos, o pdf com o relatorio (60% da nota) e um csv com os valores computados para os dados em ex6-test.csv 
