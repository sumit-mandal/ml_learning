import numpy as np
import matplotlib.pyplot as plt

def draw(x1,x2):
    ln = plt.plot(x1,x2)

def sigmoid(score):
    return 1 /( 1 + np.exp(-score))

def calculate_error(line_parameters,points,y):
    m=points.shape[0]
    p=sigmoid(points*line_parameters)
    cross_entropy = -(1 / m) * (np.log(p).T * y + np.log(1 - p).T * (1 - y))
    return cross_entropy

n_pts=10
np.random.seed(0)
bias = np.ones(n_pts)
top_region = np.array([np.random.normal(10,2,n_pts), np.random.normal(12,2,n_pts),bias]).T #T means transpose
bottom_region = np.array([np.random.normal(5,2,n_pts),np.random.normal(6,2,n_pts),bias]).T
all_points = np.vstack((top_region,bottom_region))
w1 = -0.2
w2 = -0.35
b = 3.5
line_parameters = np.matrix([w1,w2,b]).T
x1 = np.array([bottom_region[:,0].min(),top_region[:,0].max()])

x2 = -b / w2 + x1 * (- w1/w2)

y = np.array([np.zeros(n_pts),np.ones(n_pts)]).reshape(n_pts*2,1)
print(y)

linear_combination = all_points*line_parameters
linear_combination
probabilities = sigmoid(linear_combination)


_, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0],top_region[:,1],color = 'r')
#The Axes.scatter(self,x,y) function in axes module of matplotlib
# library is used to plot a scatter of y vs. x with varying
# marker size and/or color.
#x, y: These parameter are the horizontal and vertical
#coordinates of the data points.
ax.scatter(bottom_region[:,0],bottom_region[:,1],color = 'b')
draw(x1,x2)
plt.show()

print(f'error is {calculate_error(line_parameters,all_points,y)}')
