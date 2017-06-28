# Exercício 1

Gere um pdf que inclui seu código e respostas para as questões, e comentários que voce achar pertinentes. Submeta o pdf via Moodle.

Assuma os valores:

    x=[-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

    y=[-53.9, -28.5, -20.7, -3.6, -9.8, 5.0, 4.2, 5.1, 11.4, 27.4, 44.0]

Vamos ajustar uma função cúbica aos dados, minimizando o erro quadrático. Ou seja:

    $f(x)=ax^3+bx^2+cx+d$

$\min a,b,c,d \sum [yi−f(xi)]^2$

* Implemente uma decida do gradiente (escreva explicitamente o código para a função que computa o gradiente). Use um learning rate the 1.0e-5, inicie do ponto a=0,b=0,c=0,d=0, e rode 50 iterações.

* Qual a solução encontrada (os valores de a,b,c,e d)? Qual o erro na solução? Plote a função com os valores solução em conjunto com os dados.

* Use a decida do gradiente com learning rate de 1.e-4 (para convergir mais rápido). O que aconteceu?

* Use o método de BFGS do scipy.optimize.minimize. Use o BFGS sem jacobiano (o método vai computar o Jacobiano usando diferenças finitas) De novo, imprima a solução, o erro e plote a função encontrada e os dados originais. Quantas interações foram precisas?

* Use o método de BFGS do scipy.optimize.minimize. Use o Jacobiano (gradiente da solução anterior. Houve diferença entre a solução anterior? Mesmo número de chamadas para a função?

* Use o método Nelder Mead do scipy.optimize.minimize. Imprima e plote. Quantas interações?

* Implemente uma solução usando decida do gradiente usando o Tensorflow. Use o otimizador AdamOptimizer com learning rate de 0.01. Rode 200 iterações. Plote a solução.Voce pode tanto usar uma otimização SGD (atualização dos pesos um dado por vez como é mais comum em redes neurais) ou uma solução batch (atualização acontece após processar todos os dados). Mas voce precisa dizer na resposta qual das duas voce implementou!

Se voce instalou o python pelo anaconda, ha uma explicação para instalar o Tensorflow aqui (funcionou para mim com algumas mensagens de warning quando inicio o Tensorflow)

Há alguns tutoriais sobre como fazer uma regressão linear em Tensorflow. São um bom começo para o nosso problema.
