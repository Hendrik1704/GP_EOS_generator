from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, CompoundKernel,RationalQuadratic, Matern, DotProduct,ConstantKernel
import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt("EoS_hotQCD.dat", dtype=float, skiprows=0)
x_data=data[:,0].reshape(-1, 1) #the x value here must be reshaped so we only input a flat array

y_data=data[:,1]

data=np.loadtxt("EoS_hotQCD_full.dat", dtype=float, skiprows=0)
x_test=data[:,0].reshape(-1, 1) #the x value here must be reshaped so we only input a flat array

y_test=data[:,1]

#must change so that data equals the number got from standard deviation
noise=0.01

kernel =WhiteKernel(noise_level=noise) + RBF(length_scale=0.1001483459) #this determains how far away we are from the actual curve (the uncertainty)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
gpr.fit(x_data, y_data)

points=1000 #changing number points here in linspace
x_simulate = np.linspace(0, 0.5, points).reshape(-1, 1)
y_new=gpr.sample_y(x_simulate, 1, random_state=1000).flatten() #unlike ravel becuase ravel will NOT creat a copy---anyn changes made afefcts original;

#take the numerical derivitive
#create can index to relate numbers to one another
slope=5
def filter(x,y):
    dydx=np.gradient(y, x.ravel()) #creates a two dimensional array
    indices = np.where(dydx > slope)[0] #the [0] creates a new array of valid derivitives
    x_f = x[indices] #the valid elements will create Values_x
    y_f = y[indices]
    return (x_f,y_f)
Values_x,Values_y=filter(x_simulate,y_new) #these are filtering the 1000 randomly generated points
y_mean, y_std = gpr.predict(Values_x, return_std=True) #this will return y_mean that is the size of the filtered 1000 point in that of Values_x
x_finaldata,y_finaldata=filter(x_data,y_data)
x_testdata,y_testdata=filter(x_test,y_test)
#.flatten()
p=50 #percent away from the y_mean curve
shortened_y=np.interp(x_testdata,Values_y,y_mean) #instead of Values_x, we want to interpolate about Values_y; the values that are in the 1000 random point curve. Values_x will interpolate mean for y_test values
def percent_filter(x,y): #we are multiplying 64 with a 657 value
    indices = np.where(((1 - p/100) * shortened_y < y) & (y< (1 + p/100) * shortened_y))[0] #function creates a dictionary for bijections
    x_tested = x[indices]
    y_tested=y[indices]
    if np.size(x_tested)<np.size(x):
        return (x_tested,y_tested)
    else:#this is so that the interp. function doesn't start giving values greater than the number of total test points
        return(x_test,y_test)
#print(shortened_y)
#print("Lower bound example:", (1 - p/100) * shortened_y[:])
#print("Upper bound example:", (1 + p/100) * shortened_y[:])
x_passed,y_passed=percent_filter(x_testdata,y_testdata)


plt.figure(figsize=(10, 5))
plt.xlabel("x--[Temperature (GEV)]")
plt.ylabel("y--[P$T^{-4}$]")
#what is the points of having Values_y?
plt.plot(Values_x, y_mean, 'k', lw=1, ls='-', label=f'Mean Curve, Filter: dy/dx > {slope}')
plt.plot(Values_x, Values_y, 'b', lw=1, ls='-', label='Derivative Graph: dy/dx')


#plt.scatter(x_testdata, y_testdata, marker='x', color='orange', s=10, label=f'Test Data Received:{np.size(x_test)}, Test Data Remain: {np.size(x_testdata)}')#passes derivitive test
plt.scatter(x_passed, y_passed, marker='x', color='orange', s=10, label=f'Test Data Received:{np.size(x_test)}, Test Data Passing Slope and Percent ({p}%) Test: {np.size(x_passed)}')#passes percent test

plt.scatter(x_finaldata, y_finaldata, marker='x', color='r', s=10, label=f'Data Points Received:{np.size(x_data)},Data Points Remain: {np.size(x_finaldata)}')
plt.errorbar(x_finaldata, y_finaldata, yerr=noise*y_finaldata, fmt='none',alpha=0.5, color='blue', label=f'{noise*100}% Error')

#note that ALL 3 curves will have 498 points remaining by the restriction of Values_x. change the slope for different points remaining
i=np.random.seed(10) #this from the random library allows us to create ANYN set of n lines (we set n to be 3 for now)
for t in range (3): #we can change the range value here to incoporate curves 0 to n-1, in this case n=3
    plt.plot(Values_x,gpr.sample_y(Values_x, 1, random_state=i), lw=1, ls='--', label=f'Gaussian Curve {t+1}, Predicted Poins:{points}, Remaining: {np.size(Values_x)}')
plt.fill_between(Values_x.flatten(), (1-noise)*y_mean-1*y_std, (1+noise)*y_mean+1*y_std, color='red', label='68% confidence level',alpha=0.2)
plt.fill_between(Values_x.flatten(), (1-noise)*y_mean-2*y_std, (1+noise)*y_mean+2*y_std, color='orange', label='95% confidence level',alpha=0.2)
plt.fill_between(Values_x.flatten(), (1-noise)*y_mean-3*y_std, (1+noise)*y_mean+3*y_std, color='yellow', label='99.7% confidence level',alpha=0.2)

plt.legend(title="Legend", loc='lower right', fontsize='x-small')
plt.show()
