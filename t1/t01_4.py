#autor: Rodolfo Quispe
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

#define the original function
def f(x, w):
  return w[0]*(x**3) + w[1]*(x**2) + w[2]*(x) + w[3]
 
#objetive fuction
def cost(w, x, y):
  return ((f(x, w) - y)**2).sum()

#return the jacobian (gradient)
def jacobian(w, x, y):
  fact = np.vstack( (x*x*x, x*x, x , np.ones(len(x))) )
  return (2 * (f(x, w) - y) * fact).sum(axis = 1)

def plot_results(x, y, w):  
  fig = plt.figure(figsize=(12, 6))
  fig.add_subplot(111)
  
  # Plot the target t versus the input x
  plt.plot(x, y, 'o', label='data')
  # Plot the initial line
  x = list(range(-10, 10))
  plt.plot(x, [f(i,w) for i in x],'g-',label='f(x)')

  #set extra plot parameters
  plt.title('Question 4: BFGS with jacobian')
  plt.xlim(-6,6)
  plt.ylim(-60,60)
  plt.xlabel('x')
  plt.ylabel('f(x)', fontsize=15)
  plt.legend(loc=2)
  plt.grid()

  plt.show()
  fig.savefig("t01_4.png")

if __name__ == "__main__":

  x_train = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
  y_train = np.array([-53.9, -28.5, -20.7, -3.6, -9.8, 5.0, 
                        4.2, 5.1, 11.4, 27.4, 44.0])

  # initial guess
  w = np.array([0, 0, 0 , 0])

  # minimize fuction
  res = minimize(cost, w, args=(x_train, y_train, ), 
        method='BFGS', jac = jacobian, options={'disp': True})

  # show results
  print (res)
  plot_results(x_train, y_train, res.x)



