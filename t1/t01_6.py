#autor: Rodolfo Quispe
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#define the original function
def f(x, w):
  return w[0]*(x**3) + w[1]*(x**2) + w[2]*(x) + w[3]

def plot_results(x, y, w):  
  fig = plt.figure(figsize=(12, 6))
  fig.add_subplot(111)
  
  # Plot the target t versus the input x
  plt.plot(x, y, 'o', label='data')
  # Plot the initial line
  x = list(range(-10, 10))
  plt.plot(x, [f(i,w) for i in x],'g-',label='f(x)')

  #set extra plot parameters
  plt.title('Question 6: Stochastic Gradient Descent on Tensorflow')
  plt.xlim(-6,6)
  plt.ylim(-60,60)
  plt.xlabel('x')
  plt.ylabel('f(x)', fontsize=15)
  plt.legend(loc=2)
  plt.grid()

  plt.show()
  fig.savefig("t01_6.png")

if __name__ == "__main__":

  #Model parameters
  a = tf.Variable([np.random.rand()], tf.float32)
  b = tf.Variable([np.random.rand()], tf.float32)
  c = tf.Variable([np.random.rand()], tf.float32)
  d = tf.Variable([np.random.rand()], tf.float32)

  # Our model of y = a*x^3 + b*x^2 + c*x + d
  x = tf.placeholder(tf.float32)
  cubic_model = a*x*x*x + b*x*x + c*x + d
  y = tf.placeholder(tf.float32)

  # Our error is defined as the sum of the squares
  loss = tf.reduce_sum(tf.square(cubic_model - y)) 

  # Defining AdamOptimizer to calulate gradient 
  optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
  train = optimizer.minimize(loss)

  #training data
  x_train = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
  y_train = np.array([-53.9, -28.5, -20.7, -3.6, -9.8, 5.0, 
                        4.2, 5.1, 11.4, 27.4, 44.0])

  # number of gradient descent iterations
  nb_of_iterations = 200

  # training loop
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init) # reset values to wrong

  errors = []

  #SGD optimization with random order
  order = list(range(len(x_train)))
  for i in range(nb_of_iterations):
    np.random.shuffle(order)
    for j in order:
      x_value = x_train[j]
      y_value = y_train[j]
      sess.run(train, {x:x_value, y:y_value})
    errors.append(sess.run(loss, {x:x_train, y:y_train}))

  #recover solution
  curr_a, curr_b, curr_c, curr_d, curr_loss  =  sess.run(
   [a, b, c, d, loss], {x:x_train, y:y_train})

  # show solution
  print ('iterations', nb_of_iterations)
  print ('cost', curr_loss)
  print ('best parameters', curr_a, curr_b, curr_c, curr_d)
  plot_results(x_train, y_train, np.array([curr_a, curr_b, 
                                          curr_c, curr_d]))



  '''
  Extra implementaions

  #batch optimizacion
  for i in range(nb_of_iterations):
    sess.run(train, {x:x_train, y:y_train})
    errors.append(sess.run(loss, {x:x_train, y:y_train}))

  #SGD optimization
  for i in range(nb_of_iterations):
    for x_value, y_value in zip(x_train, y_train):
      sess.run(train, {x:x_value, y:y_value})
    errors.append(sess.run(loss, {x:x_train, y:y_train}))
  '''