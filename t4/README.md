# Exercicio 4

* Use os dados do arquivo abalone faça os pre-processamentos do exercicio 3.

Usando um 5-fold externo para calcular a accuracia, e um 3-fold interno para a escolha dos hyperparametros, determine qual algoritimo entre kNN, SVM com kernel RBF, redes neurais, Random Forest, e Gradient Boosting Machine tem a maior acuracia. Imprima a acuracia com 3 digitos.

1. Para o kNN, faÃ§a um PCA que mantem 90% da variancia. Busque os valores do k entre os valores 1, 5, 11, 15, 21, 25.
2. Para o SVM RBF teste para $C=2^(-5), 2^(0), 2^(5), 2^(10)$ e gamma= $2^(-15) 2^(-10) 2^(-5) 2^(0) 2^(5)$.
3. Para a rede neural, teste com 3, 7, 10, e 20 neuronios na camada escondida.
4. Para o RF, teste max_features = 2, 3, 5, 7 e n_estimators = 100, 200, 400 e 800.
5. Para o GBM (ou XGB) teste para numero de arvores = 30, 70, e 100, com learning rate de 0.1 e 0.05, e profundidade da arvore=5.Voce pode tanto usar a versao do SKlearn ou o XGBoost.
6. Voce nao precisam fazer os loops da validacao cruzada explicitamente. Pode usar a funÃ§Ã£o GridSearchCV do SKlearn..
7. Reporte a acuracia (com 3 digitos) de cada algoritmo calculada pelo 5-fold CV externo.
8. Para o algoritmo com maior accuracia, reporte o valor dos hiperparamtertos obtidos para gerar o classificador final.
