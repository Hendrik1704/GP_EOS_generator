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

noise=0.01 #the uncertainty which is fed into both the white kernel and the errorbar
slope=0 #this is the slope for the 1st derivitive, and is also the restiction for the second derivitive

randomness=np.random.seed(2)
number_of_successful_trials=2 #this is the number of successfully-filtered graphs we want to create in the while loop
linspace_simulated_points=2000 #the fewer the points, the faster the computer runs the trials
i=0 #used in the while loop and is equal to the number of successful trials of the samples (number_of_successful_trials)
f=0 #used in the while loop and counts the total number of attempts the while loop repeats


kernel =RBF(length_scale=0.0001518)#+WhiteKernel(noise_level=noise)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
gpr.fit(x_data, y_data)
#gpr_derivitive,gpr_sound   # add these parts so that the graph can better predict
# gpr_derivitive.fit()


x_working_set=[] #appending the simulated x values that work after testing the filters
y_working_set=[] #appending the simulated y values that work after testing the filters


#Trial:, Number of working linspace values:
working_simulated_1_derivitive=[] # the number of working points from the total number of simulated points
working_simulated_2_derivitive=[]
working_simulated_sound=[]

number_iterations_while=[] # a list of 1 to f, where f is defined as the total number of times the while loop has run for a trial to work
# - will repeat numbers because the derivitive filter is outside while. corresponds to the different first and second derivitive values in working_simulated

working_trial_number=[] #elements show the number of iterations of the while loop until a successful trial is added, each subsequent element is the number of while loop repeats from the most recent term

min_simulated_pressure=[] #indicate the minimim y_value of the simulated points- could be equal to min_data
max_simulated_pressure=[] #indicate the minimim y_value of the simulated points- could be equal to min_data

#print out the minimum and maximum values of the training data x_values
min_data = np.min(x_data) # the min of actual data points
max_data = np.max(x_data) # the max of actual data points
print(f"Minimum of the datapoints is: {min_data}")
print(f"Maximum of the datapoints is: {max_data}")

def derivitive_1_filter(x, y):  # put outside the while loop , put the dydx into  the filter for local variable to prevent confusion
    dydx = np.gradient(y, x)
    indices = dydx > slope #don't use np.where, instead just have the requirment written
    after_derivitive_filter = x[indices]
    print(f"Achieved Value: {len(after_derivitive_filter)}")
    print(f"Required Derivitive 1:{len(x)}")

    working_simulated_1_derivitive.append(len(after_derivitive_filter))

    if len(after_derivitive_filter) == len(x):
        return True
    else:
        return False

def derivitive_2_filter(x, y):  # put outside the while loop , put the dydx into  the filter for local variable to prevent confusion
    dydx = np.gradient(y, x)
    indices = dydx > slope #don't use np.where, instead just have the requirment written
    after_derivitive_filter = x[indices]
    print(f"(Achieved Value: {len(after_derivitive_filter)}")
    print(f"Required Derivitive 2:{len(x)}")

    working_simulated_2_derivitive.append(len(after_derivitive_filter))

    if len(after_derivitive_filter) == len(x):
        return True
    else:
        #working_simulated_2_derivitive.append(-1) add this if we only want discrete true or false from graph
        return False


def speed_sound_squared_filter(T, P):
    dPdT = np.gradient(P,T)  # creates dPdT for the simulated points--- firstly define deritive for entropy like before; dp/dt

    e = T * dPdT - P  # +u_b * n_b+u_q * n_q+u_s * n_s  #these values can be ignored
    dPdE = np.gradient(P,e)
    index = (0 < dPdE) & (dPdE < 1)  # will change due to the temperature of dense conditions
    after_sound_filter = T[index]

    print(f"Required Sound Filter (numerical):{len(dPdE)}")
    print(f"(Achieved Value: {len(after_sound_filter)}")

    working_simulated_sound.append(len(after_sound_filter))
    if len(dPdE) == len(after_sound_filter):
        return True
    else:
        return False

while i in range (0,number_of_successful_trials):
    x_simulate = np.linspace(min_data, max_data , linspace_simulated_points).reshape(-1, 1)
    y_simulate = gpr.sample_y(x_simulate, 1, random_state=randomness).flatten()
    y_simulate=x_simulate.flatten()**4*y_simulate #y_simulate is now the pressure- we interplot about pressure, then divide out in the end
    dydx = np.gradient(y_simulate, x_simulate.flatten())

    boolean_1_derivitive=derivitive_1_filter(x_simulate.flatten(), y_simulate)

    boolean_2_derivitive=derivitive_2_filter(x_simulate.flatten(), dydx)

    boolean_speedsound=speed_sound_squared_filter(x_simulate.flatten(), y_simulate)

    print(boolean_1_derivitive,boolean_2_derivitive,boolean_speedsound)

    if boolean_1_derivitive==True and boolean_2_derivitive==True and boolean_speedsound==True:
        x_working_set.append(x_simulate.flatten())
        y_working_set.append(y_simulate)
        print(f"\n\n\n\nWorking Pairs Set {i+1}:\n\n", end='\n')
        for k in range(len(x_simulate)):
            flattened_x=x_simulate.ravel()
            print(f"{x_simulate[k][0], y_simulate[k]}",end='\n')

        min_sim = np.min(y_simulate)
        max_sim = np.max(y_simulate)

        min_simulated_pressure.append(min_sim)
        max_simulated_pressure.append(max_sim)
        working_trial_number.append(f+1)
        i=i+1

    elif boolean_1_derivitive==False:
        i=i
    number_iterations_while.append(f + 1)

    f=f+1



print(f"Filter repeats needed:{working_trial_number[-1]}")

for t in range(0, len(working_trial_number)):
    print(f"Trials needed for successful sample {t+1}: {working_trial_number[t]}")
for t in range(0, len(min_simulated_pressure)): #because min_simulated and max_simulated have the same length
    print(f"Min y Simulated Point [P] for Case {t+1}: {min_simulated_pressure[t]}")
    print(f"Max y Simulated Point [P] for Case: {t + 1}: {max_simulated_pressure[t]}")



#creates custom number on the dots that work that show the number of trials needed until successful trial is reached
plt.scatter(number_iterations_while,working_simulated_1_derivitive, label="Points at 0 are failed trials")
plt.xlabel('Filter Number')
plt.ylabel('Equivalent Matching Points')
plt.title(f"Graph of 1st Derivitive of Each Trial, Trials Needed: {working_trial_number[-1]}")
plt.axhline(y=linspace_simulated_points, color='r', linestyle='--')
plt.yticks(ticks=np.append(plt.yticks()[0], linspace_simulated_points), labels=list(plt.yticks()[0]) + [f'{linspace_simulated_points}'])
for k in range (0, len(working_trial_number)):#creates veticle lines to indicate where the working x values lie on the x-axis
    plt.axvline(x=working_trial_number[k], color='r', linestyle='--')
for tick in working_trial_number:
    plt.text(tick, linspace_simulated_points, f'{tick}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transData, color='k')
plt.legend(title="Legend", loc='upper left', fontsize='x-small')
plt.show()

plt.scatter(number_iterations_while,working_simulated_2_derivitive, label="Points at 0 are failed trials")
plt.xlabel('Filter Number')
plt.ylabel('Equivalent Matching Points')
plt.title(f"Graph of 2nd Derivitive of Each Trial, Trials Needed: {working_trial_number[-1]}")
plt.axhline(y=linspace_simulated_points, color='r', linestyle='--')
plt.yticks(ticks=np.append(plt.yticks()[0], linspace_simulated_points), labels=list(plt.yticks()[0]) + [f'{linspace_simulated_points}'])
for k in range (0, len(working_trial_number)): #creates veticle lines to indicate where the working x values lie on the x-axis
    plt.axvline(x=working_trial_number[k], color='r', linestyle='--')
for tick in working_trial_number:
    plt.text(tick, linspace_simulated_points, f'{tick}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transData, color='k')
plt.legend(title="Legend", loc='upper left', fontsize='x-small')
plt.show()

plt.scatter(number_iterations_while,working_simulated_sound, label="Points at 0 are failed trials")
plt.xlabel('Filter Number')
plt.ylabel('Equivalent Matching Points')
plt.title(f"Graph of Sound Value of Each Trial, Trials Needed: {working_trial_number[-1]}")
plt.axhline(y=linspace_simulated_points, color='r', linestyle='--')
plt.yticks(ticks=np.append(plt.yticks()[0], linspace_simulated_points), labels=list(plt.yticks()[0]) + [f'{linspace_simulated_points}'])
for k in range (0, len(working_trial_number)):#creates veticle lines to indicate where the working x values lie on the x-axis
    plt.axvline(x=working_trial_number[k], color='r', linestyle='--')
for tick in working_trial_number:
    plt.text(tick, linspace_simulated_points, f'{tick}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transData, color='k')
plt.legend(title="Legend", loc='upper left', fontsize='x-small')
plt.show()

for i in range (len(x_working_set)):
    plt.figure(figsize=(10, 5))
    x_i=x_working_set[i].reshape(-1,1)
    y_i=y_working_set[i]
    y_mean, y_std = gpr.predict(x_i, return_std=True)  # this will return y_mean that is the size of the filtered points
    plt.fill_between(x_i.flatten(), (1 - noise) * y_mean - 1 * y_std, (1 + noise) * y_mean + 1 * y_std, color='red', label='68% confidence level', alpha=0.2)
    plt.fill_between(x_i.flatten(), (1 - noise) * y_mean - 2 * y_std, (1 + noise) * y_mean + 2 * y_std, color='orange', label='95% confidence level', alpha=0.2)
    plt.fill_between(x_i.flatten(), (1 - noise) * y_mean - 3 * y_std, (1 + noise) * y_mean + 3 * y_std, color='yellow', label='99.7% confidence level', alpha=0.2)
    plt.scatter(x_data,y_data, marker='x', color='r', s=10, label=f"Number Original Data Points: {len(x_data)}")
    plt.errorbar(x_data, y_data, yerr=noise * y_data, fmt='none', alpha=0.5, color='blue', label=f'{noise * 100}% Error')
    plt.scatter(x_test,y_test,marker='x', color='orange', s=10, label=f"Number Test Data Points: {len(x_test)}")
    plt.plot(x_i.flatten(),y_i/(x_i.flatten()**4), 'k', lw=1, ls='-', label=f'Filtered Curve {i+1}, Filter: dy/dx > {slope}, d^2y/dx^2 > 2nd derivitive, sound squared restraint')
    plt.title(f'Filtered Curve {i+1}')
    plt.xlabel("x--[Temperature (GEV)]")
    plt.ylabel("y--[P$T^{-4}$]")
    plt.legend(title="Legend", loc='lower right', fontsize='x-small')
    plt.show()