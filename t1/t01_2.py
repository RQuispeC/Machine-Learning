#autor: Rodolfo Quispe
import numpy as np
import matplotlib.pyplot as plt

#define the original function
def f(x, w):
  return w[0]*(x**3) + w[1]*(x**2) + w[2]*(x) + w[3]

#define  squared sum as cost function
def cost(calculated_y, expected_y):
  return ((expected_y - calculated_y)**2).sum()

#given the weights w (a, b, c, d), x 
#and y (training data), compute gradient 
def gradient(x, y, w):
  fact = np.vstack( (x*x*x, x*x, x , np.ones(len(x))) )
  return 2 * (f(x, w) - y) * fact

# define the update function delta w
def delta_w(w_k, x, t, learning_rate):
  return learning_rate * gradient(x, t, w_k).sum(axis = 1)

def plot_results(x, y, w):  
  fig = plt.figure(figsize=(12, 6))
  fig.add_subplot(111)
  
  # Plot the target t versus the input x
  plt.plot(x, y, 'o', label='data')
  # Plot the initial line
  x = list(range(-10, 10))
  plt.plot(x, [f(i,w) for i in x],'g-',label='f(x)')

  #set extra plot parameters
  plt.title('Question 2: Gradient Descent with learning rate 1e-4')
  plt.xlim(-6,6)
  plt.ylim(-60,60)
  plt.xlabel('x')
  plt.ylabel('f(x)', fontsize=15)
  plt.legend(loc=2)
  plt.grid()

  plt.show()
  fig.savefig("t01_2.png")

if __name__ == "__main__":

  x_train = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
  y_train = np.array([-53.9, -28.5, -20.7, -3.6, -9.8, 5.0, 
                        4.2, 5.1, 11.4, 27.4, 44.0])


  # Set the initial weight parameter
  w = np.array([0, 0, 0, 0])
  # Set the learning rate
  learning_rate = 1.0e-4

  #number of gradient descent iterations
  nb_of_iterations = 50 
  
  # Start performing the gradient descent updates, 
  # and print the weights and cost:
  
  # Lists to store the weight,costs values
  w_cost = [ cost(f(x_train, w), y_train)]
  w_status = [w]
  for i in range(nb_of_iterations):
      #Get the delta w update
      dw = delta_w(w, x_train, y_train, learning_rate)
      #Update the current weight parameter
      w = w - dw
      #Add weight,cost to lists
      w_cost.append(cost(f(x_train, w), y_train))
      w_status.append(w)

  print ('iterations', nb_of_iterations)
  print ('cost', w_cost[-1])
  print ('best parameters', w_status[-1])
  plot_results(x_train, y_train, w_status[-1])
