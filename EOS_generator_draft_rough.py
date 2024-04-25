from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel#, CompoundKernel,RationalQuadratic, Matern, DotProduct,ConstantKernel
import numpy as np
import matplotlib.pyplot as plt
import math

data=np.loadtxt("EoS_hotQCD.dat", dtype=float, skiprows=0)
x_data=data[:,0].reshape(-1, 1) #the x value here must be reshaped so we only input a flat array
y_data=data[:,1]
#print(y_data)

# use test data only in the scatter plot: not in the actual filtering proccess
data=np.loadtxt("EoS_hotQCD_full.dat", dtype=float, skiprows=0)
x_test=data[:,0].reshape(-1, 1) #the x value here must be reshaped so we only input a flat array
y_test=data[:,1]

#print(y_test)
pressure_list=[]



for k in range(0, len(x_data)):
    k_value=x_data[k]
    power4=np.power(k_value, 4)
    final_pressure=power4*y_data[k]
    #print(final_pressure[k])
    pressure_list.append(final_pressure[0])
# array=np.array(pressure_list)
# print(array[1])

print(f"pressure_list:{pressure_list}")
#print(len(x_data))

data=np.loadtxt("EoS_hotQCD_full.dat", dtype=float, skiprows=0)
x_test=data[:,0].reshape(-1, 1)
y_test=data[:,1]
pressure_test=[]
for k in range(0, len(x_test)):
    k_value=x_test[k]
    power4=np.power(k_value, 4)
    final_pressure=power4*y_test[k]
    #print(final_pressure[k])
    pressure_test.append(final_pressure[0])
print(pressure_test)

#all customizability labeled here
randomness=np.random.seed(3)


u_b=np.random.randint(45,55)
u_s=np.random.randint(45,55)
u_q=np.random.randint(45,55)
n_b=np.random.randint(45,55)
n_q=np.random.randint(45,55)
n_s=np.random.randint(45,55)
print(n_s)

# noise/ uncertainity added throught the white kernel
noise=0.01
# slope requirements, strictly greater than
slope=0
two_slope=0
#percentage away from the mean
p = 50
#number of random samples that we want
number=2
#changing number points generated in linspace random genrating function
points=75
i=0
f=0
special_case=0

#customize the kernel used: here, we have noise(uncertainty) with the white kernel)

#you haven't done anything with the kernel here
kernel =WhiteKernel(noise_level=noise) + RBF(length_scale=1.001818) #0.0001818 is the cutoff value until the curve oscillates without any fillters
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
gpr.fit(x_data, pressure_list)
#y_mean, y_std = gpr.predict(Values_x, return_std=True)

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

#this for loop will try vales of randomly generated function until
while i in range (0,number):
    # create a while loop such that for each sample , we create a now set of data ponts, we must add numpy arrays to a pandas data frame
    x_simulate = np.linspace(min, max, points).reshape(-1, 1)
    y_simulate = gpr.sample_y(x_simulate, 1, random_state=randomness).flatten() #here, y-simulate gives pediction points for the pressure_list which is actual pressure, not P/T^4
    dydx = np.gradient(y_simulate, x_simulate.flatten())  # creates a two dimensional array
    #print(y_simulate)
    #print(f"LISSSSSSSSSSSSSS: {pressure_list}" )
    def dervitive(x, y, m):
        indices = np.where(dydx > m)[0]  # the [0] creates a new array of valid derivitives
        x_f = x[indices]  # the valid elements will create x_f
        # this is incorrect because we only return segments of the curve that work (we then only graph these segments, so a sin curve will have discrete display)
        size=np.size(x_f)

#this will print the values of desired (x) and x_f which is the trial
        # print(len(x_f))
        # print(len(x))

        size_list.append(len(x_f))
        sample_number.append(f+1)

        if len(x_f)==len(x):
            return True
        else:
            return False

    def two_dervitive(x, y, m):
        indices = np.where(dydx > m)[0]  # the [0] creates a new array of valid derivitives that pass the test
        x_f = x[indices]  # the valid elements will create x_f
        # this is incorrect because we only return segments of the curve that work (we then only graph these segments, so a sin curve will have discrete display)
        size = np.size(x_f)

        # this will print the values of desired (x) and x_f which is the trial
        # print(len(x_f))
        # print(len(x))

        # size_list.append(len(x_f))
        # sample_number.append(f+1)

        if len(x_f) == len(x):
            return True
        else:
            return False

    def filter_sound5(x,y): #  e=T*s-P+u(b)*n(b)+u(q)*n(q)+u(s)n(s)
        # we must take the derivitive of this equation in terms of temperature
        dydx = np.gradient(y, x.flatten())  # creates dydx for the simulated points--- firstly define deritive for entropy like before; dp/dt
        pressure_array = np.array(pressure_list)
        e_values = []
        e_derivtive = []

        for k in pressure_list: # this will alwats produce same number as in the pressure_list

            e=x * dydx-pressure_array[k]+u_b * n_b+u_q * n_q+u_s * n_s  #pressure is negligible here cocmpared to u_b
            e_values.append(e)
            dydx_e = np.gradient(e_values,x.flatten())  # creates dydx for the simulated points--- firstly define deritive for entropy like before; dp/dt
            e_derivtive.append(dydx_e)
        index=np.where(e_derivtive==pow(343,2))[0] #will change due to the temperature of dense conditions
        pressure_f=x[index]

        if len(e_derivtive)==len(pressure_f):
            # indices = np.where(dydx > m)[0]  # the [0] creates a new array of valid derivitives that pass the test
            # x_f = x[indices]
            return True

        else:
            return False

    boolean=dervitive(x_simulate, y_simulate, slope) # in the first pass, the

    boolean1=two_dervitive(x_simulate, dydx, two_slope)
    boolean2=filter_sound5(x_simulate, y_simulate)



    if boolean==True and boolean1==True and boolean2==True:
        x_set.append(x_simulate.flatten())
        y_set.append(y_simulate)
        print(f"\n\n\n\nWorking Pairs Set {i+1}:\n\n", end='\n')
        for k in range(len(x_simulate)):
            flattened_x=x_simulate.ravel()
            print(f"{x_simulate[k][0], y_simulate[k]}",end='\n')
            #the [0] takes the first element in the 1 element array([x_value])--a simple syntax change.

        min_sim = np.min(x_simulate)
        max_sim = np.max(x_simulate)

        min_simulated.append(min_sim)
        max_simulated.append(max_sim)
        working_count.append(f+1)
        print(f"size list:{size_list}") # this values shows a list of 1 to f of the total trials
        print(f"sample number: {sample_number}") #for each bijective trial, we have the total number of values of simulated values that match our criteria- up until we reach desired value
        print(f"Trial {f + 1}")
        i=i+1

    elif boolean==False:
        i=i
    #i is the number of samples that work

#these values will actually show each trial
    print(f"Trial {f + 1}")

    #f counts the total number of times the filter has run
    f=f+1


print(f"Filter repeats needed:{len(sample_number)}")
for t in range(0, len(working_count)):
    print(f"Trials needed for successful sample {t+1}: {working_count[t]}")
for t in range(0, len(min_simulated)): #because min_simulated and max_simulated have the same length
    print(f"Min Simulated Point for Case {t+1}: {min_simulated[t]}")
    print(f"Max Simulated Point for Case: {t + 1}: {max_simulated[t]}")




for k in range (0, len(working_count)):
    plt.axvline(x=working_count[k], color='r', linestyle='--')
# Force matplotlib to draw the plot to make sure all updates are processed

# Retrieve the current y-tick values directly as numbers
# tick_values = plt.gca().get_yticks()
# low_tick = min(tick_values)
# print(tick_values)


for tick in working_count:
    plt.text(tick, points, f'{tick}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transData, color='k')
    #plt.xticks(ticks=working_count, labels=[f"{int(tick)}" for tick in working_count])
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

    plt.scatter(x_data,pressure_list, marker='x', color='r', s=10, label=f"Number Original Data Points: {len(x_data)}")
    plt.errorbar(x_data, pressure_list, yerr=noise * y_data, fmt='none', alpha=0.5, color='blue', label=f'{noise * 100}% Error')
    plt.scatter(x_test,pressure_test,marker='x', color='orange', s=10, label=f"Number Test Data Points: {len(x_test)}")
    plt.plot(x_i,y_i, 'k', lw=1, ls='-', label=f'Filtered Curve {i+1}, Filter: dy/dx > {slope}, d^2y/dx^2 > {two_slope}, sound squared restraint=$C^{2}$)')
    plt.title(f'Filtered Curve {i+1}')
    plt.xlabel("x--[Temperature (GEV)]")
    plt.ylabel("y--[P]") #P$T^{-4}$ power notation in scatterplot
    plt.legend(title="Legend", loc='lower right', fontsize='x-small')
    plt.show()


