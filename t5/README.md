# Exercício 5


O arquivo ex5-files.zip gera um diretorio de diretorios (com as classes) e 5000 textos divididos nos subdiretorios apropriados. Esse é o formato para a função sklearn.datasets.load_files

O arquivo category.tab contem a classe de cada documento.

Os textos sao parte de um data set de mineracao de documentos com textos de tamanho médio sobre tecnlogia. As classes sao as classes originais dos textos.

## Parte 1 - processamento de texto

Faça as tarefas usuais de processameno de textos:

* Conversao de caracteres maiusculos para minusculos
* Remoçao de pontuaçao
* Remoçao de stop words
* Steming dos termos
* Remocao dos termos que aparecem em um so documento ou que aparecem em todos.

Converta os textos processados acima em um bag of words no formato binario (0/1 se o termo aparece ou nao aparece no documento) e no formato de term frequency.

Em Python, o sklearn tem funçoes para gerar as diferentes formas da matriz termo-documento. Um tutorial sobre as funcoes no sklearn

O preprocessamento acima deve ser feito para todos os textos em conjunto. Não é preciso fazer a separação de fit no conjunto de treino e transform no conjunto de teste.

Divida o conjunto em 1000 documentos de teste e 4000 de treino aleatoriamente (pode ser estratificado ou nao).

## Parte 2 - Naive bayes

* Rode o naive bayes (BernoulliNB) na matriz binaria. Qual a acuracia?

* Rode o naive Bayes (MultinomialNB) na matriz de term-frequency. Qual a accuracia (compare com a anterior).

## Parte 3 - PCA e outros classificadores

* Rode o PCA e reduza o numero de dimensoes da matriz de term-frequency para 99% da variancia original. Voce nao consiguira usar o PCA tradicional do Sklearn. Voce terá que usar o TuncatedSVD que é menos conveniente.

* Rode SVM com RBF (modo one vs all), gradient boosting e random forest na matriz com o numero de dimensoes reduzidas. Não se esqueca de fazer a busca de hiperparametros. Quais as acurácias?

* Qual o melhor classificador dos testados? 
