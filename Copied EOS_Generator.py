from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt

from scipy import interpolate
import pickle

#this compute_derivitive does the same thing as the gradient function
def compute_derivative(x, y):
    """This function compute dy/dx using finite difference"""
    slope=np.gradient(y, x, edge_order=2)
    return slope

def compute_energy_density(T, P):
    """This function computes energy density"""
    dPdT = compute_derivative(T, P)
    e = T * dPdT - P  # energy density
    return e

def compute_speed_of_sound_square (T,P):
    dPdT = compute_derivative(T, P)
    e = T * dPdT - P
    c_squared = np.gradient(P, e)
    return c_squared

def compute_value_of_et4 (T,P):
    e = compute_energy_density(T, P)
    desired_y_value=e/(T**4)
    return desired_y_value

def derivative_filter(x, y,slope):
    """
        This filter check whether the derivative is larger than 0
        for all array elements
    """
    dydx = compute_derivative(x, y)
    indices = dydx > slope
    negative_derivatives = dydx[indices]

    #print(f"ooooooooooooooooooo{negative_derivatives}")
    if len(negative_derivatives) == len(x):
        return True
    else:
        return False

def derivitive_1_append_list(x, y, slope,list):
    dydx = compute_derivative(x, y)
    indices = dydx > slope
    after_derivitive_filter = dydx[indices]
    # print(f"Required Derivitive 1:{len(x)}")
    # print(f"Achieved Value: {len(after_derivitive_filter)}")
    list.append(len(after_derivitive_filter))

def derivitive_2_filter(x, y, slope):
    dydx = compute_derivative(x, y)
    dy2dx = compute_derivative(x, dydx)
    indices = dy2dx > slope
    after_derivitive_filter = dy2dx[indices]
    # print(len(after_derivitive_filter))
    # print(len(x))
    if len(after_derivitive_filter) == len(x):
        return True
    else:
        return False

def derivitive_2_append_list(x, y, slope,list):
    dydx = compute_derivative(x, y)
    dy2dx = compute_derivative(x, dydx)
    indices = dy2dx > slope
    after_derivitive_filter = dy2dx[indices]
    # print(f"Required Derivitive 2:{len(x)}")
    # print(f"Achieved Value: {len(after_derivitive_filter)}")
    list.append(len(after_derivitive_filter))

def speed_sound_squared_filter(T, P):
    dPdT = compute_derivative(T, P)
    e = T * dPdT - P
    dPdE = compute_derivative(e, P)
    index = (0 < dPdE) & (dPdE < 1) #we must use <1 for the values in the endpoints
    after_sound_filter = dPdE[index]
    # print(len(dPdE))
    # print(len(after_sound_filter))

    if len(dPdE) == len(after_sound_filter):
        return True
    else:
        return False

def speed_sound_squared_append_list(T,P,list):
    dPdT =compute_derivative(T, P)
    e = T * dPdT - P
    dPdE = compute_derivative(e, P)
    index = (0 < dPdE) & (dPdE < 1)
    after_sound_filter = dPdE[index]
    # print(f"Required Sound Filter (numerical):{len(dPdE)}")
    # print(f"Achieved Value: {len(after_sound_filter)}\n")
    list.append(len(after_sound_filter))

def masking_func(min_bound,max_bound,data): #the point of the mask is so that the
    mask = ((data[:, 0] < min_bound) | (data[:, 0] > max_bound))
    new_masked_x = data[mask]  # this is the CHANGED training values after manipulating under 'main'
    return new_masked_x

#-----------------------------------------------------------------------------------------------
ACCURACY = 1e-6
MAXITER = 100
def binary_search_1d(y_local, f_y, x_min, x_max):
    """
        This function performs a binary search to find the x value
        that corresponds to the y value y_local
    """
    iteration = 0
    y_low = f_y(x_min)
    y_up = f_y(x_max)
    if y_local < y_low:
        return x_min
    elif y_local > y_up:
        return x_max
    else:
        x_mid = (x_max + x_min) / 2.
        y_mid = f_y(x_mid)
        abs_err = abs(y_mid - y_local)
        rel_err = abs_err / abs(y_mid + y_local + 1e-15)
        while (rel_err > ACCURACY and abs_err > ACCURACY * 1e-2
               and iteration < MAXITER):
            if y_local < y_mid:
                x_max = x_mid
            else:
                x_min = x_mid
            x_mid = (x_max + x_min) / 2.
            y_mid = f_y(x_mid)
            abs_err = abs(y_mid - y_local)
            rel_err = abs_err / abs(y_mid + y_local + 1e-15)
            iteration += 1 #we are doing this until we have enough iterations or the accuracy is comprimised----we are essentially dividing 2 each time for MANY MANY times and seeing if the desired x values is above or below the average.
        return x_mid


def invert_EoS_tables(T, P): #notice that we use P, so we must scale down by factor 1/4 power so that our return e value is usable .
    """
        This function inverts the EoS table to get e(T), it also computes the
        pressure P(T)
    """
    e = compute_energy_density(T, P)
    f_e = interpolate.interp1d(T, e, kind='cubic') #by interpolating, we create PREDICTED y values based on the FIXED DATA VALUS THAT WE ALREADY KNOW----it create a function that can predict an infinite number of x values and give a Y_value
    f_p = interpolate.interp1d(T, P, kind='cubic')

    e_bounds = [np.min(e), np.max(e)]
    e_list = np.linspace(e_bounds[0] ** 0.25, e_bounds[1] ** 0.25, 200) ** 4 #here, we schaled dhown by power 1/4 then back up to ensure the values are well seperated/more uniform in our linspace-[0] is the minimum value while [1] is the maximum

    T_from_e = []
    for e_local in e_list:
        T_local = binary_search_1d(e_local, f_e, T[0], T[-1]) # we are using the MIN and MAX t values here, along with the values of the RANDOMLY GENERATED e_list values, which we then will narrow down to
        T_from_e.append(T_local)
    T_from_e = np.array(T_from_e)
    return (e_list ** 0.25, f_p(T_from_e), T_from_e) #here, we get a value of T from the interpolation of e into a model given the values of T and p. We then predict a value of T from random values of e. These values are then used to predict P values



def EoS_file_writer(e, P, T, filename):
    """
        This function writes the EoS to a pickle file with a dictionary
        for each EoS. The different columns are: e, P, T
    """
    EoS_dict = {}
    for EoS in range(len(e)):
        data = np.column_stack((e[EoS], P[EoS], T[EoS]))  #each time this iterates, we will create NEW data--- as e,p,t should be LISTS OF ARRAYS. This means that the eos_dict function will assign a value
        # (EOS---the tot. number of arrays in the list) to the stacked data. This acts as a 'key' to the dictionary. The first data set could be accessed using EOS_dict[1]
        EoS_dict[f'{EoS:04}'] = data #we assign the stacked data as an element in the EOS_dict
    print(f"000000000000000000000000000{EoS_dict[1]}")
    with open(filename, 'wb') as f: #we want to write the data into the pickle file for later use----here, f is just the placeholder for the name
        pickle.dump(EoS_dict, f)
#-----------------------------------------------------------------------------------------------


def combine_all_filters(T, P,slope) -> bool:
    """
        This calls the different physics filters to check if the
        EoS is a physical one
    """
    if derivative_filter(T, P,slope)==False:
        return False

    if derivitive_2_filter(T, P,slope)==False:
        return False

    if speed_sound_squared_filter(T, P)==False:
        return False

    return True




def main(ranSeed, slope,linspace_simulated_points,number_of_successful_trials,sliced_amount,min_mask_x_values, max_mask_x_values, blogflag):
    data=np.loadtxt("EoS_hotQCD.dat") #here, we set the x_train to a fixed variable. We use T_plot as the variable that can be bound to change. we will also use that as the plotting variable.
    #y_data=data[:,1]

    data_test=np.loadtxt("EoS_hotQCD_full.dat")
    #y_test=data_test[:,1]



#this issue is we need to take the x values that WORK and then have a set of y values that correpsond to those.
    train_masked_x=masking_func(min_mask_x_values,max_mask_x_values,data)
    test_masked_x=masking_func(min_mask_x_values,max_mask_x_values,data_test)


#we create the corresponding values for new_masked_x's y values
    #print(f"{train_masked_x}000000000000000000000000000000000000")


    T_Min=np.min(train_masked_x[:,0])
    T_Max=np.max(train_masked_x[:,0])


    # set the random seed, check for incorrect negative seeds
    if ranSeed >= 0:
        randomness = np.random.seed(ranSeed)
    else:
        randomness = np.random.seed()


    kernel = RBF(length_scale=0.2, length_scale_bounds=(1e-3, 100))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-4)

    if blogflag: #define the x_train variable and establish the t_plot, which is what we wil use as desires x values
        x_train = np.log(train_masked_x[:, 0]).reshape(-1, 1)
        gpr.fit(x_train, np.log(train_masked_x[:, 1]))
        print(f"GP inputs score: {gpr.score(x_train, np.log(train_masked_x[:, 1]))}")
        T_GP = np.linspace(np.log(T_Min), np.log(T_Max), 1000).reshape(-1, 1)
        T_GP=T_GP[sliced_amount: -sliced_amount]
        T_plot = np.exp(T_GP.flatten())


    else:
        x_train = train_masked_x[:, 0].reshape(-1, 1)
        gpr.fit(x_train, train_masked_x[:, 1]) #x_train is needed to just train the gpr model
        print(f"GP inputs score: {gpr.score(x_train, train_masked_x[:, 1])}")
        T_GP = np.linspace(T_Min, T_Max, 1000).reshape(-1, 1)
        T_GP = T_GP[sliced_amount: -sliced_amount]
        T_plot = T_GP.flatten()


#now, we must convert the log values in blog flag BACK to original values (no negatives) and plot T_Plot

#first simulate the y values----we only need an x values to do so.


    x_working_set=[]
    y_working_set=[] #this is the equivalent of the Working_EOS_set=[] set

    number_iterations_while=[]


    working_simulated_1_derivitive=[]
    working_simulated_2_derivitive=[]
    working_simulated_sound=[]

    working_computed_speed_sound=[]
    working_computed_e=[]

    working_trial_number=[]

    min_simulated_pressure=[]
    max_simulated_pressure=[]

    min_data = np.min(x_train)
    max_data = np.max(x_train)
    print(f"Minimum of the datapoints is: {min_data}")
    print(f"Maximum of the datapoints is: {max_data}\n-------------------------")

    i=0
    f=0
    batch_size=100
    iter=0
    #T_GP ARE the simulated x points
    while i < number_of_successful_trials:
#this is because we draw from a data set with p/t^4
        P_divide_Tto4 = gpr.sample_y(T_GP, linspace_simulated_points, random_state=randomness).transpose() #this is randomly---we produce 1000 samples that are each 996 in array size after we slice
# ----the sample_y expects 2D array with x values and a fitted y value (through the GPR) so that we create MULTIPLE SAMPLES of possible y values to use

#notice that because we neither flattern nor reshape t_gp, the linspace function can now do what its made for: creating arrays inside a numpy array.


        P=P_divide_Tto4[0,:]*T_GP.flatten()**4 #here, p is an ARRAY of ARRAYS from the transpose function, they are ALL y values #we must use T_GP to make the linspace simulated values


        #print(P)
        for sample_i in P_divide_Tto4: #the loop will stop here when all the numpy arrays in the array of P has been iterated and tested---->all the generated arrays have been tested--->we will generate the first len(number_of_successful_trials) that match the combine_filter
        #here, sample_i is already sliced because of the T_GP slicing
            #Working_EOS_set = []
            if blogflag:
                P_plot = (np.exp(sample_i)*T_plot**4)
            else:
                P_plot = ((sample_i)*T_plot**4)

            # the derivitives that work for derivitive, 2nd derivitive, and sound values will work for the P_plot AND the Pover4 values----each value that doesn't work will lokely not work for the Pover4 value
            derivitive_1_append_list(T_plot.flatten(), P_plot, slope, working_simulated_1_derivitive)
            derivitive_2_append_list(T_plot.flatten(), P_plot, slope, working_simulated_2_derivitive)
            speed_sound_squared_append_list(T_plot.flatten(), P_plot, working_simulated_sound)


            if combine_all_filters(T_plot.flatten(),P_plot,slope)==True:
                #Working_EOS_set.append(P_plot)
                x_working_set.append(T_plot.flatten())
                y_working_set.append(P_plot)  # appending this is different than appending P_plot
                i = i + 1
                #print(f"\nWorking Pairs Set {i} Above:\n---------------------------------------", end='\n')

                min_sim = np.min(sample_i)
                max_sim = np.max(sample_i)

                min_simulated_pressure.append(min_sim)
                max_simulated_pressure.append(max_sim)
                working_trial_number.append(f + 1)

                c_squared = compute_speed_of_sound_square(T_plot.flatten(),P_plot)
                working_computed_speed_sound.append(c_squared[sliced_amount:-sliced_amount])
                computed_e_divide_T4 = compute_value_of_et4(T_plot.flatten(), P_plot)
                working_computed_e.append(computed_e_divide_T4[sliced_amount:-sliced_amount])


            number_iterations_while.append(f + 1)
            f = f + 1
            if number_of_successful_trials==i:
                break

        # print(f"value i{i}")
        # print(f"value i{f}")
        print(f"For loop success rate: {float(i) / f:.8f}\n----------------------------------------------") #this is the number of times the for loop runs successfully (i values) over the total runs by f

#here, we have to GRAPH the generated points with the y values corresponding so that we get line graph



#here, we HAVE to remove the data points that are invalid.
    pressure_original_data = train_masked_x[:, 0].flatten() ** 4 * train_masked_x[:, 1]
    sound_squared_orginial_dataset = compute_speed_of_sound_square(train_masked_x[:, 0].flatten(), pressure_original_data)
    pressure_test_data = test_masked_x[:, 0].flatten() ** 4 * test_masked_x[:, 1]
    sound_squared_test_dataset = compute_speed_of_sound_square(test_masked_x[:, 0].flatten(), pressure_test_data)
    train_energy_density_set = compute_value_of_et4(train_masked_x[:, 0].flatten(), pressure_original_data)
    test_energy_density_set = compute_value_of_et4(test_masked_x[:, 0].flatten(), pressure_test_data)




    # print(f"Filter repeats needed:{working_trial_number[-1]}")
    # print(f"Trials needed for successful sample 1: {working_trial_number[0]}")
    # for t in range(1, len(working_trial_number)):
    #     print(f"Trials needed for successful sample {t+1}: {working_trial_number[t]-working_trial_number[t-1]}")
    #
    #
    # for t in range(0, len(working_trial_number)):
    #     print(f"Min y Simulated Point [P/T^4] for Case {t+1}: {min_simulated_pressure[t]}")
    #     print(f"Max y Simulated Point [P/T^4] for Case: {t + 1}: {max_simulated_pressure[t]}")

# -------------------------------------------------
        # invert the EoS tables
    e_list_EoS = []
    P_list_EoS = []
    T_list_EoS = []



    for i in range(number_of_successful_trials):
        if (i + 1) % 100 == 0: #if our iteration is dividable by 0, (and have successfully been uploaded, we will show the progress using the print)
            print(f"Inverting EoS table {i + 1}/{number_of_successful_trials}") #we do not slice the number of successful trials- we only slice for the linspace generated values, as we want to rid of the outlier data values
        e_list, P_list, T_list = invert_EoS_tables(T_plot, y_working_set[i])


        e_list_EoS.extend([e_list])
        P_list_EoS.extend([P_list])
        T_list_EoS.extend([T_list])

    # write the EoS to a file
    EoS_file_writer(e_list_EoS, P_list_EoS, T_list_EoS, f"EoS.pkl")

#------------------------------------------------------------



    plt.scatter(number_iterations_while,working_simulated_1_derivitive, label='Number of Satisfied Trials per loop')
    plt.xlabel('Filter Number')
    plt.ylabel('Equivalent Matching Points')
    plt.title(f"Graph of 1st Derivitive of Each Trial, Trials Needed: {working_trial_number[-1]}")

    # plt.axhline(y=linspace_simulated_points, color='r', linestyle='--')
    # plt.yticks(ticks=np.append(plt.yticks()[0], linspace_simulated_points), labels=list(plt.yticks()[0]) + [f'{linspace_simulated_points}'])
    # for k in range (0, len(working_trial_number)):#creates veticle lines to indicate where the working x values lie on the x-axis
    #     plt.axvline(x=working_trial_number[k], color='r', linestyle='--')
    # for tick in working_trial_number:
    #     plt.text(tick, linspace_simulated_points, f'{tick}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transData, color='k')

    plt.legend(title="Legend", loc='upper left', fontsize='x-small')
    plt.show()
#--------------------------------------------
    plt.scatter(number_iterations_while,working_simulated_2_derivitive, label='Number of Satisfied Trials per loop')
    plt.xlabel('Filter Number')
    plt.ylabel('Equivalent Matching Points')
    plt.title(f"Graph of 2nd Derivitive of Each Trial, Trials Needed: {working_trial_number[-1]}")

    # plt.axhline(y=linspace_simulated_points, color='r', linestyle='--')
    # plt.yticks(ticks=np.append(plt.yticks()[0], linspace_simulated_points), labels=list(plt.yticks()[0]) + [f'{linspace_simulated_points}'])
    # for k in range (0, len(working_trial_number)): #creates veticle lines to indicate where the working x values lie on the x-axis
    #     plt.axvline(x=working_trial_number[k], color='r', linestyle='--')
    # for tick in working_trial_number:
    #     plt.text(tick, linspace_simulated_points, f'{tick}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transData, color='k')

    plt.legend(title="Legend", loc='upper left', fontsize='x-small')
    plt.show()
#--------------------------------------
    plt.scatter(number_iterations_while,working_simulated_sound, label='Number of Satisfied Trials per loop')
    plt.xlabel('Filter Number')
    plt.ylabel('Equivalent Matching Points')
    plt.title(f"Graph of Sound Value of Each Trial, Trials Needed: {working_trial_number[-1]}")


    # plt.axhline(y=linspace_simulated_points, color='r', linestyle='--')
    # plt.yticks(ticks=np.append(plt.yticks()[0], linspace_simulated_points), labels=list(plt.yticks()[0]) + [f'{linspace_simulated_points}'])
    # for k in range (0, len(working_trial_number)):#creates veticle lines to indicate where the working x values lie on the x-axis
    #     plt.axvline(x=working_trial_number[k], color='r', linestyle='--')
    # for tick in working_trial_number:
    #     plt.text(tick, linspace_simulated_points, f'{tick}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transData, color='k')


    plt.legend(title="Legend", loc='upper left', fontsize='x-small')
    plt.show()

    plt.figure(figsize=(10, 5))


    #we have to CHANGE our y_train to match the values of t_plot and simulate sample_y from here.
    for i in range (len(x_working_set)):
        x_i=x_working_set[i].reshape(-1,1)
        y_i=y_working_set[i]
        plt.plot(x_i.flatten(),y_i/(x_i.flatten()**4), lw=1, ls='-', label=f'Filtered Curve {i+1}') #this will give the value for p/t^4
    plt.scatter(train_masked_x[:, 0][sliced_amount:-sliced_amount], train_masked_x[:, 1][sliced_amount:-sliced_amount], marker='x', color='r', s=10, label=f"Number Original Data Points: {len(train_masked_x[sliced_amount:-sliced_amount])}")
    plt.scatter(test_masked_x[:, 0][sliced_amount:-sliced_amount], test_masked_x[:, 1][sliced_amount:-sliced_amount], marker='x', color='orange', s=10, label=f"Number Test Data Points: {len(test_masked_x[:, 0][sliced_amount:-sliced_amount])}")
    plt.title(f'Filtered Gaussian Predict Curve, Filter: dy/dx > {slope}, d^2y/dx^2 > 2nd derivitive, sound squared restraint')
    plt.xlabel("x--[Temperature (GEV)]")
    plt.ylabel("y--[P$T^{-4}$]")
    #plt.legend(title="Legend", loc='lower right', fontsize='x-small')
    plt.show()

    for i in range(len(x_working_set)):
        x_i = x_working_set[i].reshape(-1, 1)
        plt.plot(x_i.flatten(), working_computed_speed_sound[i], lw=1, ls='-',label=f'Working Speed Number {i+1}')
    plt.title(f'Speed of Sound Curve')
    plt.xlabel("x--[Temperature (GEV)]")
    plt.ylabel("y--[Speed of Sound Squared c^2]")
    plt.scatter(train_masked_x[:, 0][sliced_amount:-sliced_amount], sound_squared_orginial_dataset[sliced_amount:-sliced_amount], marker='x', color='r', s=10, label=f"Original Dataset Sound Squared, Number Data: {len(train_masked_x[:, 0][sliced_amount:-sliced_amount])}")
    plt.scatter(test_masked_x[:, 0][sliced_amount:-sliced_amount], sound_squared_test_dataset[sliced_amount:-sliced_amount], marker='x', color='orange', s=10, label=f"Test Dataset Sound Squared, Number Data: {len(test_masked_x[:, 0][sliced_amount:-sliced_amount])}")
    #plt.legend(title="Legend", loc='lower right', fontsize='x-small')
    plt.show()

    for i in range(len(x_working_set)):
        x_i = x_working_set[i].reshape(-1, 1)
        plt.plot(x_i.flatten(), working_computed_e[i], lw=1, ls='-',label=f'Working Energy Density Graph Number: {i+1}')
    plt.title(f'Energy Density Curve')
    plt.xlabel("x--[Temperature (GEV)]")
    plt.ylabel("y--[e/T^4]")
    #here, we use slices to get rid of extraneous values/outliers that create the verticle lines at the end of the graph
    plt.scatter(train_masked_x[:, 0][sliced_amount:-sliced_amount], train_energy_density_set[sliced_amount:-sliced_amount], marker='x', color='r', s=10, label=f"Original Dataset Energy Density, Number Data: {len(train_masked_x[:, 0][sliced_amount:-sliced_amount])}")
    plt.scatter(test_masked_x[:, 0][sliced_amount:-sliced_amount], test_energy_density_set[sliced_amount:-sliced_amount], marker='x', color='orange', s=10, label=f"Test Dataset Energy Density, Number Data: {len(test_masked_x[:, 0][sliced_amount:-sliced_amount])}")
    #plt.legend(title="Legend", loc='lower right', fontsize='x-small')
    plt.show()






if __name__ == "__main__":
    #by setting a main function, all the contollable variables can be set here.
    ranSeed = 20
    slope=0
    linspace_simulated_points = 1000
    number_of_successful_trials = 1000
    sliced_amount=2 #this will also slice the number of linspace_generated points to match the shapes of all the appended lists.
    blogflag=True #tests if we want smaller intervals by scaling down to a log size to project data through linspace function
    min_mask_x_values=0.20 #these values will detmerain how restricted the curve is at certain x values----how much variation we want in this interval-- also determains the GP score along with the effectiveness of the for loop in training.
    max_mask_x_values=0.25
    main(ranSeed,slope, linspace_simulated_points, number_of_successful_trials,sliced_amount,min_mask_x_values, max_mask_x_values, blogflag)
