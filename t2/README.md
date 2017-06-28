# Exercício 2

Gere um pdf que inclui seu código e respostas para as questões, e comentários que voce achar pertinentes. Submeta o pdf via Moodle.

## Pre-processamento

* Leia o arquivo abalone.csv
* Converta a primeira coluna de dados categóricos para numéricos usando o one-hot encoding. A função get_dumies parece ser mais conveniente que o one-hot-encoded do scikit learn.
* A última coluna será transformada em um dado categórico. A classe de saída será 1 se a ultima coluna for maior que 13 e 0 se menor ou igual a 13.
* Uma vez que vc criou o atributo de saída reova a ultima coluna.
* Separe os primeiros 3133 dados como treino e os restantes como teste.

## Classificação

1. Regressão logística com C=1000000. Faca a regressão logística dos dados transformados, com um C alto (sem regularização). Imprima a acurácia do classificador nos dados de teste com 3 dígitos significativos. Rode o LogisticrRgression com random_state=1 para garantir que de o mesmo resultado toda vez que vc rodar (isso seta o valor da semente do gerador aleatório e portanto usará sempre o mesmo ponto inicial na otimização da regressão logística).

2. Regressão logística com regularização (C=1). Imprima com 3 dígitos a acurácia, e use random_state=1.

3. Regressão logística sem regularização e com estandardização dos dados. Use C=1000000 mas transforme os dados antes de aplicar a regressão logistica.

4. Aplique um PCA nos dados, de forma que pelo menos 90% da variância dos dados seja preservada.

5. Rode a regressão logística sem regularização nos dados do PCA

6. Rode a regressão logística com regularização (C=1) nos dados do PCA

7. Leia o arquivo abalone-missing.csv com dados faltantes na 2 a penúltima coluna. Faça o preprocessamento descrito em 1. e impute pela média os valores faltantes. Rode a regressão sem regularização, sem PCA e sem estandardização.
