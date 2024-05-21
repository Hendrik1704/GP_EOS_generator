from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt
##
def derivitive_1_filter(x, y, slope):
    dydx = np.gradient(y, x)
    indices = dydx > slope
    after_derivitive_filter = x[indices]
    if len(after_derivitive_filter) == len(x):
        return True
    else:
        return False

def derivitive_1_append_list(x, y, slope,working_simulated_1_derivitive):
    dydx = np.gradient(y, x)
    indices = dydx > slope
    after_derivitive_filter = x[indices]

    working_simulated_1_derivitive.append(len(after_derivitive_filter))

    print(f"Achieved Value: {len(after_derivitive_filter)}")
    print(f"Required Derivitive 1:{len(x)}")
    if len(after_derivitive_filter) == len(x):
        return True
    else:
        return False

def derivitive_2_filter(x, y, slope):
    dydx = np.gradient(y, x)
    indices = dydx > slope
    after_derivitive_filter = x[indices]
    if len(after_derivitive_filter) == len(x):
        return True
    else:
        return False

def derivitive_2_append_list(x, y, slope,working_simulated_2_derivitive):
    dydx = np.gradient(y, x)
    indices = dydx > slope
    after_derivitive_filter = x[indices]
    print(f"(Achieved Value: {len(after_derivitive_filter)}")
    print(f"Required Derivitive 2:{len(x)}")
    working_simulated_2_derivitive.append(len(after_derivitive_filter))
    if len(after_derivitive_filter) == len(x):
        return True
    else:
        return False

def speed_sound_squared_filter(T, P):
    dPdT = np.gradient(P,T)
    e = T * dPdT - P
    dPdE = np.gradient(P,e)
    index = (0 < dPdE) & (dPdE < 1)
    after_sound_filter = T[index]
    if len(dPdE) == len(after_sound_filter):
        return True
    else:
        return False

def speed_sound_squared_append_list(P,T,working_simulated_sound):
    dPdT = np.gradient(P, T)
    e = T * dPdT - P
    dPdE = np.gradient(P, e)
    index = (0 < dPdE) & (dPdE < 1)
    after_sound_filter = T[index]
    print(f"Required Sound Filter (numerical):{len(dPdE)}")
    print(f"Achieved Value: {len(after_sound_filter)}\n------------------------------------------------------")
    working_simulated_sound.append(len(after_sound_filter))


def compute_speed_of_sound_square (T,P):
    dPdT = np.gradient(P,T)
    e = T * dPdT - P
    c_squared = np.gradient(P, e)
    return c_squared

def compute_value_of_et4 (T,P):
    dPdT = np.gradient(P,T)  # creates dPdT for the simulated points--- firstly define deritive for entropy like before; dp/dt
    e = T * dPdT - P  # +u_b * n_b+u_q * n_q+u_s * n_s  #these values can be ignored
    desired_y_value=e/(T**4)
    return desired_y_value

def main(ranSeed, slope,linspace_simulated_points,number_of_successful_trials):
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
    print(f"Maximum of the datapoints is: {max_data}")

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

        boolean_1_derivitive=derivitive_1_filter(x_simulate.flatten(), y_simulate, slope)
        derivitive_1_append_list(x_simulate.flatten(), y_simulate, slope, working_simulated_1_derivitive)

        boolean_2_derivitive=derivitive_2_filter(x_simulate.flatten(), dydx, slope)
        derivitive_2_append_list(x_simulate.flatten(), y_simulate, slope, working_simulated_2_derivitive)

        boolean_speedsound=speed_sound_squared_filter(x_simulate.flatten(), y_simulate)
        speed_sound_squared_append_list(x_simulate.flatten(), y_simulate, working_simulated_sound)


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

            # please note that  working_computed_speed_sound and working_simulated_sound are differant-
            # one serves purpose of scatter plot to show ALL values of sound chart in each trial,the other for the simulated values where all other booleans are also true
            c_squared = compute_speed_of_sound_square(x_simulate.flatten(), y_simulate)
            working_computed_speed_sound.append(c_squared)
            computed_e_divide_T4=compute_value_of_et4(x_simulate.flatten(), y_simulate)
            working_computed_e.append(computed_e_divide_T4)

            i=i+1


        number_iterations_while.append(f + 1)

        f=f+1

    print(f"Filter repeats needed:{working_trial_number[-1]}")

    for t in range(0, len(working_trial_number)):
        print(f"Trials needed for successful sample {t+1}: {working_trial_number[t]}")
    for t in range(0, len(min_simulated_pressure)):
        print(f"Min y Simulated Point [P] for Case {t+1}: {min_simulated_pressure[t]}")
        print(f"Max y Simulated Point [P] for Case: {t + 1}: {max_simulated_pressure[t]}")

    plt.scatter(number_iterations_while,working_simulated_1_derivitive)
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

    plt.scatter(number_iterations_while,working_simulated_2_derivitive)
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

    plt.scatter(number_iterations_while,working_computed_speed_sound)
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
    plt.scatter(x_data, y_data, marker='x', color='r', s=10, label=f"Number Original Data Points: {len(x_data)}")
    plt.scatter(x_test, y_test, marker='x', color='orange', s=10, label=f"Number Test Data Points: {len(x_test)}")
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
    plt.scatter(x_data, pressure_original_data, marker='x', color='r', s=10, label=f"Original Dataset Sound Squared, Number Data: {len(x_data)}")
    plt.scatter(x_test, pressure_test_data, marker='x', color='orange', s=10, label=f"Test Dataset Sound Squared, Number Data: {len(x_test)}")
    plt.legend(title="Legend", loc='upper left', fontsize='x-small')
    plt.show()


    for i in range(len(x_working_set)):
        x_i = x_working_set[i].reshape(-1, 1)
        plt.plot(x_i.flatten(), working_computed_e[i], lw=1, ls='-',label=f'Working Energy Density Graph Number: {i+1}')
    plt.title(f'Energy Density Curve')
    plt.xlabel("x--[Temperature (GEV)]")
    plt.ylabel("y--[e/T^4]")
    plt.scatter(x_data, train_energy_density_set, marker='x', color='r', s=10, label=f"Original Dataset Energy Density, Number Data: {len(x_data)}")
    plt.scatter(x_test, test_energy_density_set, marker='x', color='orange', s=10, label=f"Test Dataset Energy Density, Number Data: {len(x_test)}")
    plt.legend(title="Legend", loc='upper left', fontsize='x-small')
    plt.show()
if __name__ == "__main__":
    ranSeed = 23
    slope=0
    linspace_simulated_points = 1000
    number_of_successful_trials = 10
    main(ranSeed,slope, linspace_simulated_points, number_of_successful_trials)
