import numpy as np
import matplotlib.pyplot as plt

n_pts=100
top_region = np.array([np.random.normal(10,2,n_pts),np.random.normal(12,2,n_pts)]).T
bottom_region = np.array([np.random.normal(5,2,n_pts),np.random.normal(6,2,n_pts)]).T

_, ax = plt.subplots(figsize=(4,4))
ax.scatter(top_region[:,0],top_region[:,1],color = 'r')
#The Axes.scatter() function in axes module of matplotlib
# library is used to plot a scatter of y vs. x with varying
# marker size and/or color.
ax.scatter(bottom_region[:,0],top_region[:,1],color = 'b')
plt.show()
