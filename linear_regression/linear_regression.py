# see basic example here:
#    http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
# full documentation of the linear_model module here:
#    http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

import numpy.random # for generating a noisy data set
from sklearn import linear_model # model fitting/training
import matplotlib.pyplot # for plotting in general
from mpl_toolkits.mplot3d import Axes3D # for 3D plotting

MIN_X = -10
MAX_X = 10
NUM_INPUTS = 50

################################################################################
#  GENERATED DATA
################################################################################

# Generate some normally distributed noise
noise = numpy.random.normal(size=NUM_INPUTS)

### 1 feature (2D)

# randomly pick 50 numbers
x1 = numpy.random.uniform(low=MIN_X, high=MAX_X, size=(NUM_INPUTS, 1))

# x needs to be an array of arrays for the model fit function, but sometimes we also need it as a list.
# The [:,0] slicing pulls out the values into a one-dimensional list
x1_1d = x1[:,0]

# y = 0.3x + 1
#noise is the error
y1_1 = 0.3 * x1_1d + 1 + noise
#y1_1 = 0.3 * x1_1d + 1

# y = 0.7x^2 - 0.4x + 1.5
y1_2 = 0.7 * x1_1d * x1_1d - 0.4 * x1_1d + 1.5 + noise


### 2 features (3D)

# randomly pick 50 pairs of numbers
x2 = numpy.random.uniform(low=MIN_X, high=MAX_X, size=(NUM_INPUTS, 2))

# x needs to be an array of arrays for the model fit function, but sometimes we also need each x as a list.
# The [:,n] slicing pulls out the values into a one-dimensional list
x2_1_1d = x2[:,0]
x2_2_1d = x2[:,1]

# y = 0.5x_1 - 0.2x_2 - 2
y2 = 0.5 * x2_1_1d - 0.2 * x2_2_1d - 2 + noise
#y2 = 0.5 * x2_1_1d - 0.2 * x2_2_1d - 2


################################################################################
# MODEL TRAINING
################################################################################

# use scikit-learn's linear regression model and fit to our 2D data
model2d = linear_model.LinearRegression()
model2d.fit(x1, y1_1)

# Repeat for the 3D data
model3d = linear_model.LinearRegression()
model3d.fit(x2, y2)

# Print out the parameters for the best fit line/plane
print()
print()
print('\t##### Output #####')
print('\t2D Data: Intercept: {0}  Coefficients: {1}'.format(model2d.intercept_, model2d.coef_))
print('\t3D Data: Intercept: {0}  Coefficients: {1}'.format(model3d.intercept_, model3d.coef_))
print()
print()


################################################################################
# PLOT
################################################################################

# 2D Plot

# create the first figure
fig = matplotlib.pyplot.figure(1)
fig.suptitle('2D Data and Best-Fit Line')
matplotlib.pyplot.xlabel('x')
matplotlib.pyplot.ylabel('y')

# put the generated points on the graph
matplotlib.pyplot.scatter(x1_1d, y1_1)

# predict for inputs along the graph to find the best-fit line
X = numpy.linspace(MIN_X, MAX_X)
Y = model2d.predict(list(zip(X)))
matplotlib.pyplot.plot(X, Y)

# 3D Plot

# create the second figure
fig = matplotlib.pyplot.figure(2)
fig.suptitle('3D Data and Best-Fit Plane')

# get the current axes, and tell them to do a 3D projection
axes = fig.gca(projection='3d')
axes.set_xlabel('x1')
axes.set_ylabel('x2')
axes.set_zlabel('y')

# put the generated points on the graph
axes.scatter(x2_1_1d, x2_2_1d, y2)

# predict for input points across the graph to find the best-fit plane
# and arrange them into the gride that matplotlib appears to require
X1 = X2 = numpy.arange(MIN_X, MAX_X, 0.05)
X1, X2 = numpy.meshgrid(X1, X2)
Y = numpy.array(model3d.predict(list(zip(X1.flatten(), X2.flatten())))).reshape(X1.shape)

# put the predicted plane on the graph
axes.plot_surface(X1, X2, Y, alpha=0.1)

# show the plots
matplotlib.pyplot.show()
