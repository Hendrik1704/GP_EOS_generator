from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt("EoS_hotQCD.dat", dtype=float, skiprows=0)
x_data=data[:,0].reshape(-1, 1)
y_data=data[:,1]
data=np.loadtxt("EoS_hotQCD_full.dat", dtype=float, skiprows=0)
x_test=data[:,0].reshape(-1, 1)
y_test=data[:,1]
noise=0.01
slope=0
p = 50
randomness=np.random.seed(2)
number=2
points=350
i=0
f=0
special_case=0
kernel =WhiteKernel(noise_level=noise) + RBF(length_scale=0.001818)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
gpr.fit(x_data, y_data)
x_set=[]
y_set=[]
size_list=[]
sample_number= []
working_count=[]
min_simulated=[]
max_simulated=[]
min = np.min(x_data)
max = np.max(x_data)
print(f"Minimum of the datapoints is: {min}")
print(f"Maximum of the datapoints is: {max}")
while i in range (0,number):
    x_simulate = np.linspace(min, max, points).reshape(-1, 1)
    y_simulate = gpr.sample_y(x_simulate, 1, random_state=randomness).flatten()
    dydx = np.gradient(y_simulate, x_simulate.flatten())
    def filter(x, y):
        indices = np.where(dydx > slope)[0]
        x_f = x[indices]
        size=np.size(x_f)
        size_list.append(len(x_f))
        sample_number.append(f+1)
        if len(x_f)==len(x):
            return True
        else:
            return False
    def filter_sound5(x,y):
        pass
    boolean=filter(x_simulate, y_simulate)
    boolean1=filter(x_simulate, dydx)
    if boolean==True and boolean1==True:
        x_set.append(x_simulate.flatten())
        y_set.append(y_simulate)
        print(f"\n\n\n\nWorking Pairs Set {i+1}:\n\n", end='\n')
        for k in range(len(x_simulate)):
            flattened_x=x_simulate.ravel()
            print(f"{x_simulate[k][0], y_simulate[k]}",end='\n')
        min_sim = np.min(x_simulate)
        max_sim = np.max(x_simulate)
        min_simulated.append(min_sim)
        max_simulated.append(max_sim)
        working_count.append(f+1)
        i=i+1
    elif boolean==False:
        i=i
    f=f+1
print(f"Filter repeats needed:{len(sample_number)}")
for t in range(0, len(working_count)):
    print(f"Trials needed for successful sample {t+1}: {working_count[t]}")
for t in range(0, len(min_simulated)): #because min_simulated and max_simulated have the same length
    print(f"Min Simulated Point for Case {t+1}: {min_simulated[t]}")
    print(f"Max Simulated Point for Case: {t + 1}: {max_simulated[t]}")
for k in range (0, len(working_count)):
    plt.axvline(x=working_count[k], color='r', linestyle='--')
for tick in working_count:
    plt.text(tick, points, f'{tick}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transData, color='k')
plt.scatter(sample_number,size_list)
plt.xlabel('Filter Number')
plt.ylabel('Equivalent Matching Points')
plt.title(f"Graph of Each Trial, Trials Needed: {len(sample_number)}")
plt.axhline(y=points, color='r', linestyle='--')
plt.yticks(ticks=np.append(plt.yticks()[0], points), labels=list(plt.yticks()[0]) + [f'{points}'])
for i in range (len(x_set)):
    plt.figure(figsize=(10, 5))
    x_i=x_set[i].reshape(-1,1)
    y_i=y_set[i]
    y_mean, y_std = gpr.predict(x_i, return_std=True)  # this will return y_mean that is the size of the filtered points
    plt.fill_between(x_i.flatten(), (1 - noise) * y_mean - 1 * y_std, (1 + noise) * y_mean + 1 * y_std, color='red', label='68% confidence level', alpha=0.2)
    plt.fill_between(x_i.flatten(), (1 - noise) * y_mean - 2 * y_std, (1 + noise) * y_mean + 2 * y_std, color='orange', label='95% confidence level', alpha=0.2)
    plt.fill_between(x_i.flatten(), (1 - noise) * y_mean - 3 * y_std, (1 + noise) * y_mean + 3 * y_std, color='yellow', label='99.7% confidence level', alpha=0.2)
    plt.scatter(x_data,y_data, marker='x', color='r', s=10, label=f"Number Original Data Points: {len(x_data)}")
    plt.errorbar(x_data, y_data, yerr=noise * y_data, fmt='none', alpha=0.5, color='blue', label=f'{noise * 100}% Error')
    plt.scatter(x_test,y_test,marker='x', color='orange', s=10, label=f"Number Test Data Points: {len(x_test)}")
    plt.plot(x_i,y_i, 'k', lw=1, ls='-', label=f'Filtered Curve {i+1}, Filter: dy/dx > {slope}')
    plt.plot(x_i,y_i, 'k', lw=1, ls='-', label=f'Filtered Curve {i+1}, Filter: dy/dx > {slope}, d^2y/dx^2 > 2nd derivitive, sound squared restraint=$C^{2}$)')
    plt.title(f'Filtered Curve {i+1}')
    plt.xlabel("x--[Temperature (GEV)]")
    plt.ylabel("y--[P$T^{-4}$]")
    plt.legend(title="Legend", loc='lower right', fontsize='x-small')
    plt.show()