# Exercicio 7

Os dados quakes.csv representam dodos reais de terremotos (nao sei de que periodo). A primeira coluna é a profundidado do terremoto, a segunda e terceira a latitude e longitude, e a quarta a scala Richer do terremoto. Nos queremos discobrir se ha clusters/grupos de terremotos.

* Estandarize cada coluna
* Rode o K-means para K=2..10, use random_state=1, e imprima (com 2 casas decimais) a silueta media e o indice de Calinski-Harabaz. Qual parece ser o melhor valor/valores para K?
* Talvez os dados de latitude e longitude estao estragando a clusterizacao. Remova essa colunas e repita a tarefa acima? Discuta os resultados
* Rode a clusterizacao hierarquica (metodo Ward) para 2..10 clusters para os dados de 4 dimensoes e calcule os 2 indices acima? Qual o melhor/melhores valores do numero de clusters?
* Rode o DBScan nos dados de 4 dimensoes. Use 5 como min_samples. Construa o grafico da distancia dos 5-NN e descubra o valor do eps. (Se voce nao conseguir gerar o grafico, use eps = 0.75 mas essa opcao perderá alguns pontos nesta questao). Qual o número de clusters? Calcule os indices acima para os clusteres.

Discuta quantos clusters voce acha apropriado para esse problema.
