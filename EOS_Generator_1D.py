from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel#, CompoundKernel,RationalQuadratic, Matern, DotProduct,ConstantKernel
import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt("EoS_hotQCD.dat", dtype=float, skiprows=0)
x_data=data[:,0].reshape(-1, 1) #the x value here must be reshaped so we only input a flat array
y_data=data[:,1]


# use test data only in the scatter plot: not in the actual filtering proccess
data=np.loadtxt("EoS_hotQCD_full.dat", dtype=float, skiprows=0)
x_test=data[:,0].reshape(-1, 1) #the x value here must be reshaped so we only input a flat array
y_test=data[:,1]


#all customizability labeled here

# noise/ uncertainity added throught the white kernel
noise=0.01
# slope requirements
slope=0
#percentage away from the mean
p = 50
randomness=np.random.seed(2)
#number of random samples that we want
number=2
#changing number points generated in linspace random genrating function
points=255
i=0
f=0


#customize the kernel used: here, we have noise(uncertainty) with the white kernel)

#you haven't done anything with the kernel here
kernel =WhiteKernel(noise_level=noise) + RBF(length_scale=0.001818) #0.0001818 is the cutoff value until the curve oscillates without any fillters
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-2)
gpr.fit(x_data, y_data)
#y_mean, y_std = gpr.predict(Values_x, return_std=True)

x_set=[]
y_set=[]

size_list=[]
sample_number= []



#this for loop will try vales of randomly generated function until
while i in range (0,number):
    # create a while loop such that for each sample , we create a now set of data ponts, we must add numpy arrays to a pandas data frame
    x_simulate = np.linspace(0, 0.5, points).reshape(-1, 1)
    y_simulate = gpr.sample_y(x_simulate, 1, random_state=randomness).flatten()

    def filter(x, y):
        dydx = np.gradient(y, x.flatten())  # creates a two dimensional array
        indices = np.where(dydx >= slope)[0]  # the [0] creates a new array of valid derivitives
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
    boolean=filter(x_simulate, y_simulate)

    if boolean==True:
        x_set.append(x_simulate.flatten())
        y_set.append(y_simulate)
        print(f"\n\n\n\nWorking Pairs Set {i+1}:", end='\n')
        for k in range(len(x_simulate)):
            print(f"{x_simulate[k].flatten(), y_simulate[k]}",end='\n')

        i=i+1
    elif boolean==False:
        i=i
    #i is the number of samples that work

#these values will actually show each trial
    # print(size_list)
    # print(sample_number)


    #f counts the total number of times the filter has run
    f=f+1


print(f"Filter repeats needed:{len(sample_number)}")
plt.scatter(sample_number,size_list)
plt.title(f"Graph of Each Trial, Trials Needed: {len(sample_number)}")



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
    plt.plot(x_i,y_i, 'k', lw=1, ls='-', label=f'Filtered Curve {i+1}, Filter: dy/dx >= {slope}')
    plt.title(f'Filtered Curve {i+1}')
    plt.xlabel("x--[Temperature (GEV)]")
    plt.ylabel("y--[P$T^{-4}$]")
    plt.legend(title="Legend", loc='lower right', fontsize='x-small')
    plt.show()


