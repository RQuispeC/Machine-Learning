# Exercício 3 
## Preprocessamento

* Leia o arquivo abalone do exercicio 2
* Faça o preprocessamento do atributo categorico e do atributo de saida como no exercicio 2
* Estandardize todos os atributos numéricos. Voce pode estardartizar todo o arquivo de uma vez. Como discutimos em aula esse não é a coisa 100% certa, mas é um erro menor.


## Logistic regression

* Faça o logistic regression com $C=10^{-1,0,1,2,3}$. O loop externo deve ser um 5-fold CV estratificado. O loop interno para a escolha do hiperparametro deve ser um 3-fold estratificado.
* Voce tem que fazer o loop interno explicitamente, usando StratifiedKFold e não funções como GridSearchCV
* Qual a acurácia do LR com a melhore escolha de parametros (para cada fold)?

## Linear SVM

* Faça o LinearSVM com $C=10^{-1,0,1,2,3}$. O loop externo deve ser um 5-fold estratificado. O loop interno um 3-fold estratificado. Neste caso voce nao precisa fazer o 3 fold explicitamente, voce pode usar o GridSearchCV.
* Qual a acurácia do LinearSVM com a melhor escolha de C?

## LDA
* Faça o LDA. Reporte a acuracia

## Classificador final

* Qual o melhor classificador para esse problema?
* Se não o LDA, calcule o hiperparametro C a ser usado
* Gere o classificador final.
