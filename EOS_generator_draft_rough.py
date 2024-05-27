from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt

#this compute_derivitive does the same thing as the gradient function
def compute_derivative(x, y):
    """This function compute dy/dx using finite difference"""
    slope=np.gradient(y, x, edge_order=1)
    return slope

def compute_speed_of_sound_square (T,P):
    dPdT = compute_derivative(T, P)
    e = T * dPdT - P
    c_squared = np.gradient(P, e)
    return c_squared

def compute_value_of_et4 (T,P):
    dPdT = compute_derivative(T, P)
    e = T * dPdT - P
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
    print(f"Required Derivitive 1:{len(x)}")
    print(f"Achieved Value: {len(after_derivitive_filter)}")
    list.append(len(after_derivitive_filter))

def derivitive_2_filter(x, y, slope):
    dydx = compute_derivative(x, y)
    dy2dx = compute_derivative(x, dydx)
    indices = dy2dx > slope
    after_derivitive_filter = dy2dx[indices]
    print(len(after_derivitive_filter))
    print(len(x))
    if len(after_derivitive_filter) == len(x):
        return True
    else:
        return False

def derivitive_2_append_list(x, y, slope,list):
    dydx = compute_derivative(x, y)
    dy2dx = compute_derivative(x, dydx)
    indices = dy2dx > slope
    after_derivitive_filter = dydx[indices]
    print(f"Required Derivitive 2:{len(x)}")
    print(f"Achieved Value: {len(after_derivitive_filter)}")
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
    print(f"Required Sound Filter (numerical):{len(dPdE)}")
    print(f"Achieved Value: {len(after_sound_filter)}")
    list.append(len(after_sound_filter))

def masking_func(min_bound,max_bound,data):
    mask = ((data[:, 0] > min_bound) | (data[:, 0] < max_bound))
    new_masked_x = data[mask]  # this is the CHANGED training values after manipulating under 'main'
    return new_masked_x

# def binary_search_1d(y_local, f_y, x_min, x_max):
#     """
#         This function performs a binary search to find the x value
#         that corresponds to the y value y_local
#     """
#     iteration = 0
#     y_low = f_y(x_min)
#     y_up = f_y(x_max)
#     if y_local < y_low:
#         return x_min
#     elif y_local > y_up:
#         return x_max
#     else:
#         x_mid = (x_max + x_min) / 2.
#         y_mid = f_y(x_mid)
#         abs_err = abs(y_mid - y_local)
#         rel_err = abs_err / abs(y_mid + y_local + 1e-15)
#         while (rel_err > ACCURACY and abs_err > ACCURACY * 1e-2
#                and iteration < MAXITER):
#             if y_local < y_mid:
#                 x_max = x_mid
#             else:
#                 x_min = x_mid
#             x_mid = (x_max + x_min) / 2.
#             y_mid = f_y(x_mid)
#             abs_err = abs(y_mid - y_local)
#             rel_err = abs_err / abs(y_mid + y_local + 1e-15)
#             iteration += 1
#         return x_mid
#
#
# def invert_EoS_tables(T, P):
#     """
#         This function inverts the EoS table to get e(T), it also computes the
#         pressure P(T)
#     """
#     e = compute_energy_density(T, P)
#     f_e = interpolate.interp1d(T, e, kind='cubic')
#     f_p = interpolate.interp1d(T, P, kind='cubic')
#
#     e_bounds = [np.min(e), np.max(e)]
#     e_list = np.linspace(e_bounds[0] ** 0.25, e_bounds[1] ** 0.25, 200) ** 4
#
#     T_from_e = []
#     for e_local in e_list:
#         T_local = binary_search_1d(e_local, f_e, T[0], T[-1])
#         T_from_e.append(T_local)
#     T_from_e = np.array(T_from_e)
#     return (e_list ** 0.25, f_p(T_from_e), T_from_e)
#
#
# def EoS_file_writer(e, P, T, filename):
#     """
#         This function writes the EoS to a pickle file with a dictionary
#         for each EoS. The different columns are: e, P, T
#     """
#     EoS_dict = {}
#     for EoS in range(len(e)):
#         data = np.column_stack((e[EoS], P[EoS], T[EoS]))
#         EoS_dict[f'{EoS:04}'] = data
#     with open(filename, 'wb') as f:
#         pickle.dump(EoS_dict, f)

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


def main(ranSeed, slope,linspace_simulated_points,number_of_successful_trials,sliced_amount):
    data=np.loadtxt("EoS_hotQCD.dat", dtype=float, skiprows=0)
    x_data=data[:,0].reshape(-1, 1)
    y_data=data[:,1]

    data=np.loadtxt("EoS_hotQCD_full.dat", dtype=float, skiprows=0)
    x_test=data[:,0].reshape(-1, 1)
    y_test=data[:,1]

    # set the random seed, check for incorrect negative seeds
    if ranSeed >= 0:
        randomness = np.random.seed(ranSeed)
    else:
        randomness = np.random.seed()
    f=0

    kernel = RBF(length_scale=0.2, length_scale_bounds=(1e-3, 100))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5)
    gpr.fit(x_data, y_data)

    x_working_set=[]
    y_working_set=[]

    number_iterations_while=[]


    working_simulated_1_derivitive=[]
    working_simulated_2_derivitive=[]
    working_simulated_sound=[]

    working_computed_speed_sound=[]
    working_computed_e=[]

    working_trial_number=[]

    min_simulated_pressure=[]
    max_simulated_pressure=[]

    min_data = np.min(x_data)
    max_data = np.max(x_data)
    print(f"Minimum of the datapoints is: {min_data}")
    print(f"Maximum of the datapoints is: {max_data}\n-------------------------")

    pressure_original_data=x_data.flatten()**4*y_data
    sound_squared_orginial_dataset=compute_speed_of_sound_square(x_data.flatten(),pressure_original_data)
    pressure_test_data=x_test.flatten()**4*y_test
    sound_squared_test_dataset=compute_speed_of_sound_square(x_test.flatten(),pressure_test_data)
    train_energy_density_set=compute_value_of_et4(x_data.flatten(),pressure_original_data)
    test_energy_density_set=compute_value_of_et4(x_test.flatten(),pressure_test_data)

    i=0
    while i < number_of_successful_trials:
        x_simulate = np.linspace(min_data, max_data , linspace_simulated_points).reshape(-1, 1)
        y_simulate = gpr.sample_y(x_simulate, 1, random_state=randomness).flatten()
        y_simulate=x_simulate.flatten()**4*y_simulate
        dydx = np.gradient(y_simulate, x_simulate.flatten())

        boolean_1_derivitive=derivative_filter(x_simulate.flatten(), y_simulate, slope)
        derivitive_1_append_list(x_simulate.flatten(), y_simulate, slope, working_simulated_1_derivitive)

        boolean_2_derivitive=derivitive_2_filter(x_simulate.flatten(), y_simulate, slope)
        derivitive_2_append_list(x_simulate.flatten(), y_simulate, slope, working_simulated_2_derivitive)

        boolean_speedsound=speed_sound_squared_filter(x_simulate.flatten(), y_simulate)
        speed_sound_squared_append_list(x_simulate.flatten(), y_simulate, working_simulated_sound)


        print(f"{boolean_1_derivitive,boolean_2_derivitive,boolean_speedsound}")

        if boolean_1_derivitive==True and boolean_2_derivitive==True and boolean_speedsound==True:
            x_working_set.append(x_simulate[sliced_amount:-sliced_amount].flatten())
            y_working_set.append(y_simulate[sliced_amount:-sliced_amount])
            print(f"\nWorking Pairs Set {i+1} Above:\n---------------------------------------", end='\n')
            # for k in range(len(x_simulate)):
            #     flattened_x=x_simulate.ravel()
            #     #print(f"{x_simulate[k][0], y_simulate[k]}",end='\n')

            min_sim = np.min(y_simulate)
            max_sim = np.max(y_simulate)

            min_simulated_pressure.append(min_sim)
            max_simulated_pressure.append(max_sim)
            working_trial_number.append(f+1)

            # please note that  working_computed_speed_sound and working_simulated_sound are differant-
            # one serves purpose of scatter plot to show ALL values of sound chart in each trial,the other for the simulated values where all other booleans are also true
            c_squared = compute_speed_of_sound_square(x_simulate.flatten(), y_simulate)
            working_computed_speed_sound.append(c_squared[sliced_amount:-sliced_amount])
            computed_e_divide_T4=compute_value_of_et4(x_simulate.flatten(), y_simulate)
            working_computed_e.append(computed_e_divide_T4[sliced_amount:-sliced_amount])

            i=i+1


        number_iterations_while.append(f + 1)

        f=f+1

    print(f"Filter repeats needed:{working_trial_number[-1]}")

    for t in range(0, len(working_trial_number)):
        print(f"Trials needed for successful sample {t+1}: {working_trial_number[t]}")
    for t in range(0, len(min_simulated_pressure)):
        print(f"Min y Simulated Point [P] for Case {t+1}: {min_simulated_pressure[t]}")
        print(f"Max y Simulated Point [P] for Case: {t + 1}: {max_simulated_pressure[t]}")

    plt.scatter(number_iterations_while,working_simulated_1_derivitive, label='Number of Satisfied Trials per loop')
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

    plt.scatter(number_iterations_while,working_simulated_2_derivitive, label='Number of Satisfied Trials per loop')
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

    plt.scatter(number_iterations_while,working_simulated_sound, label='Number of Satisfied Trials per loop')
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

    plt.figure(figsize=(10, 5))


    for i in range (len(x_working_set)):
        x_i=x_working_set[i].reshape(-1,1)
        y_i=y_working_set[i]
        plt.plot(x_i.flatten(),y_i/(x_i.flatten()**4), lw=1, ls='-', label=f'Filtered Curve {i+1}') #this will give the value for p/t^4
    plt.scatter(x_data[sliced_amount:-sliced_amount], y_data[sliced_amount:-sliced_amount], marker='x', color='r', s=10, label=f"Number Original Data Points: {len(x_data[sliced_amount:-sliced_amount])}")
    plt.scatter(x_test[sliced_amount:-sliced_amount], y_test[sliced_amount:-sliced_amount], marker='x', color='orange', s=10, label=f"Number Test Data Points: {len(x_test[sliced_amount:-sliced_amount])}")
    plt.title(f'Filtered Gaussian Predict Curve, Filter: dy/dx > {slope}, d^2y/dx^2 > 2nd derivitive, sound squared restraint')
    plt.xlabel("x--[Temperature (GEV)]")
    plt.ylabel("y--[P$T^{-4}$]")
    plt.legend(title="Legend", loc='lower right', fontsize='x-small')
    plt.show()

    for i in range(len(x_working_set)):
        x_i = x_working_set[i].reshape(-1, 1)
        plt.plot(x_i.flatten(), working_computed_speed_sound[i], lw=1, ls='-',label=f'Working Speed Number {i+1}')
    plt.title(f'Speed of Sound Curve')
    plt.xlabel("x--[Temperature (GEV)]")
    plt.ylabel("y--[Speed of Sound Squared c^2]")
    plt.scatter(x_data[sliced_amount:-sliced_amount], sound_squared_orginial_dataset[sliced_amount:-sliced_amount], marker='x', color='r', s=10, label=f"Original Dataset Sound Squared, Number Data: {len(x_data[sliced_amount:-sliced_amount])}")
    plt.scatter(x_test[sliced_amount:-sliced_amount], sound_squared_test_dataset[sliced_amount:-sliced_amount], marker='x', color='orange', s=10, label=f"Test Dataset Sound Squared, Number Data: {len(x_test[sliced_amount:-sliced_amount])}")
    plt.legend(title="Legend", loc='lower right', fontsize='x-small')
    plt.show()

    for i in range(len(x_working_set)):
        x_i = x_working_set[i].reshape(-1, 1)
        plt.plot(x_i.flatten(), working_computed_e[i], lw=1, ls='-',label=f'Working Energy Density Graph Number: {i+1}')
    plt.title(f'Energy Density Curve')
    plt.xlabel("x--[Temperature (GEV)]")
    plt.ylabel("y--[e/T^4]")
    #here, we use slices to get rid of extraneous values/outliers that create the verticle lines at the end of the graph
    plt.scatter(x_data[sliced_amount:-sliced_amount], train_energy_density_set[sliced_amount:-sliced_amount], marker='x', color='r', s=10, label=f"Original Dataset Energy Density, Number Data: {len(x_data[sliced_amount:-sliced_amount])}")
    plt.scatter(x_test[sliced_amount:-sliced_amount], test_energy_density_set[sliced_amount:-sliced_amount], marker='x', color='orange', s=10, label=f"Test Dataset Energy Density, Number Data: {len(x_test[sliced_amount:-sliced_amount])}")
    plt.legend(title="Legend", loc='lower right', fontsize='x-small')
    plt.show()

if __name__ == "__main__":
    #by setting a main function, all the contollable variables can be set here.
    ranSeed = 23
    slope=0
    linspace_simulated_points = 1000
    number_of_successful_trials = 10
    sliced_amount=5
    main(ranSeed,slope, linspace_simulated_points, number_of_successful_trials,sliced_amount)