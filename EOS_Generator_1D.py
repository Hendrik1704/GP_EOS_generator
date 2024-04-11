from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, CompoundKernel,RationalQuadratic, Matern, DotProduct,ConstantKernel
import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt("EoS_hotQCD.dat", dtype=float, skiprows=0)
x_data=data[:,0].reshape(-1, 1) #the x value here must be reshaped so we only input a flat array

y_data=data[:,1]
#must change so that data equals the number got from standard deviation
number=np.size(x_data)
noise=0.01

kernel =WhiteKernel(noise_level=noise) + RBF(length_scale=0.1001483459) #this determains how far away we are from the actual curve (the uncertainty)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
gpr.fit(x_data, y_data)

x_simulate = np.linspace(0, 0.5, 1000).reshape(-1, 1)
y_mean, y_std = gpr.predict(x_simulate, return_std=True)

plt.figure(figsize=(10, 5))
plt.xlabel("x--[Temperature (GEV)]")
plt.ylabel("y--[P$T^{-4}$]")

plt.plot(x_simulate, y_mean, 'k', lw=1, ls='-', label='Mean Curve')
plt.scatter(x_data, y_data, marker='x', color='r', s=10, label='Data Points')
plt.errorbar(x_data, y_data, yerr=noise*y_data, fmt='none',alpha=0.5, color='blue', label=f'{noise*100}% Error')


i=np.random.seed(10) #this from the random library allows us to create ANYN set of n lines (we set n to be 3 for now)
for n in range (3): #we can change the range value here to incoporate curves 0 to n-1, in this case n=3
    plt.plot(x_simulate,gpr.sample_y(x_simulate, 1, random_state=i), lw=1, ls='--', label=f'Predictive Gaussian Curve {n+1}')
plt.fill_between(x_simulate.ravel(), (1-noise)*y_mean-1*y_std, (1+noise)*y_mean+1*y_std, color='red', label='68% confidence level',alpha=0.2)
plt.fill_between(x_simulate.ravel(), (1-noise)*y_mean-2*y_std, (1+noise)*y_mean+2*y_std, color='orange', label='95% confidence level',alpha=0.2)
plt.fill_between(x_simulate.ravel(), (1-noise)*y_mean-3*y_std, (1+noise)*y_mean+3*y_std, color='yellow', label='99.7% confidence level',alpha=0.2)

plt.legend(title="Legend", loc='lower right', fontsize='small')
plt.show()
