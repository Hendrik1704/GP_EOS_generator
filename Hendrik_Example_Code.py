from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pickle

ACCURACY = 1e-6
MAXITER = 100


def compute_derivative(x, y):
    """This function compute dy/dx using finite difference"""
    return np.gradient(y, x, edge_order=2)


def compute_energy_density(T, P):
    """This function computes energy density"""
    dPdT = compute_derivative(T, P)
    e = T * dPdT - P  # energy density
    return e


def compute_entropy_density(T, P):
    """This function computes entropy density"""
    dPdT = compute_derivative(T, P)
    s = dPdT  # entropy density
    return s


def compute_speed_of_sound_square(T, P):
    """This function computes the speed of sound square"""
    e = compute_energy_density(T, P)
    dPde = compute_derivative(e, P)
    return dPde


def derivative_filter(x, y) -> bool:
    """
        This filter check whether the derivative is larger than 0
        for all array elements
    """
    dydx = compute_derivative(x, y)
    indices = dydx < 0.
    negative_derivatives = x[indices]

    if len(negative_derivatives) == 0:
        return True
    else:
        return False


def speed_sound_squared_filter(T, P) -> bool:
    """
    	This filter ensures the speed of sound square is between 0 and 0.5.
    	The upper bound is chosen such that the causality constraints in
    	the hydrodynamical simulations are satisfied.
    """
    cs2 = compute_speed_of_sound_square(T, P)
    index = (cs2 > 0.) & (cs2 < 0.5)
    physical_points = cs2[index]

    if len(cs2) == len(physical_points):
        return True
    else:
        return False


def is_a_physical_eos(T, P) -> bool:
    """
        This calls the different physics filters to check if the
        EoS is a physical one
    """
    if not derivative_filter(T, P):
        return False

    dPdT = compute_derivative(T, P)
    if not derivative_filter(T, dPdT):
        return False

    if not speed_sound_squared_filter(T, P):
        return False

    return True


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
            iteration += 1
        return x_mid


def invert_EoS_tables(T, P):
    """
        This function inverts the EoS table to get e(T), it also computes the
        pressure P(T)
    """
    e = compute_energy_density(T.flatten(), P)
    f_e = interpolate.interp1d(T.flatten(), e, kind='cubic')
    f_p = interpolate.interp1d(T.flatten(), P, kind='cubic')

    e_bounds = [np.min(e), np.max(e)]
    e_list = np.linspace(e_bounds[0] ** 0.25, e_bounds[1] ** 0.25, 200) ** 4

    T_from_e = []
    for e_local in e_list:
        T_local = binary_search_1d(e_local, f_e, T[0].flatten(), T[-1].flatten())
        T_from_e.append(T_local)
    T_from_e = np.array(T_from_e)
    print(e_list ** 0.25)
    print(f_p(T_from_e))
    print(T_from_e)

    return (e_list ** 0.25, f_p(T_from_e), T_from_e)


def EoS_file_writer(e, P, T, filename):
    """
        This function writes the EoS to a pickle file with a dictionary
        for each EoS. The different columns are: e, P, T
    """
    EoS_dict = {}
    for EoS in range(len(e)):
        data = np.column_stack((e[EoS], P[EoS], T[EoS]))
        EoS_dict[f'{EoS:04}'] = data
    with open(filename, 'wb') as f:
        pickle.dump(EoS_dict, f)


def main(ranSeed: int, number_of_EoS: int, min_T_mask_region: float,
         max_T_mask_region: float, bLogFlag: bool) -> None:
    # load the full EOS table for verification
    validation_data = np.loadtxt("EoS_hotQCD.dat")

    # mask for the training data and exclude the region where the T
    # (first column) is between min_T_mask_region and max_T_mask_region
    mask = ((validation_data[:, 0] < min_T_mask_region) |
            (validation_data[:, 0] > max_T_mask_region))
    training_data = validation_data[mask]

    # print out the minimum and maximum values of the training data x_values
    T_min = np.min(training_data[:, 0])  # the min of actual data points
    T_max = np.max(training_data[:, 0])  # the max of actual data points
    # print(f"Minimum of the datapoints is: {T_min}")
    # print(f"Maximum of the datapoints is: {T_max}")

    # set the random seed
    if ranSeed >= 0:
        randomness = np.random.seed(ranSeed)
    else:
        randomness = np.random.seed()

    # define GP kernel
    kernel = RBF(length_scale=0.2, length_scale_bounds=(1e-3, 100))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-4)

    if bLogFlag:
        # train GP with log(T) vs. log(P/T^4) because all the quantities
        # are positive
        x_train = np.log(training_data[:, 0]).reshape(-1, 1)
        gpr.fit(x_train, np.log(training_data[:, 1]))
        print(f"GP score: {gpr.score(x_train, np.log(training_data[:, 1]))}")
        T_GP = np.linspace(np.log(T_min), np.log(T_max), 1000).reshape(-1, 1)  #by reshaping, we take the last array of the linspace function thus only one set of T_GP

       # print(T_GP)
        T_plot = np.exp(T_GP.flatten())
    else:
        x_train = training_data[:, 0].reshape(-1, 1)
        gpr.fit(x_train, training_data[:, 1])
        print(f"GP score: {gpr.score(x_train, training_data[:, 1])}")
        T_GP = np.linspace(T_min, T_max, 1000).reshape(-1, 1)
        T_plot = T_GP.flatten()

    EOS_set = []

    iSuccess = 0
    iter = 0
    nsamples_per_batch = 100
    while iSuccess < number_of_EoS:
        PoverT4_GP = gpr.sample_y(T_GP, nsamples_per_batch,
                                  random_state=randomness).transpose()

        for sample_i in PoverT4_GP:
            if bLogFlag:
                P_GP = np.exp(sample_i) * (T_plot ** 4)  # convert to P

            else:
                P_GP = sample_i * (T_plot ** 4)  # convert to P

            #print(f"trail {}")
            if is_a_physical_eos(T_plot, P_GP):
                EOS_set.append(P_GP)
                iSuccess += 1
                if iSuccess == number_of_EoS:
                    break

        iter += nsamples_per_batch
        print(f"Sample success rate: {float(iSuccess) / iter:.3f}")

    # make verification plots
    plt.figure()
    plt.scatter(training_data[:, 0], training_data[:, 1],
                marker='x', color='r', s=20, label="training data")
    plt.scatter(validation_data[:, 0], validation_data[:, 1],
                marker='+', color='b', s=20, label="validation data")
    for i in range(number_of_EoS):
        plt.plot(T_plot, EOS_set[i] / T_plot ** 4, '-')

    plt.legend()
    plt.xlim([0, 1.])
    # plt.ylim([0, 4.5])
    plt.xlabel(r"T (GeV)")
    plt.ylabel(r"$P/T^{4}$")
    plt.show()

    # plot e vs T
    plt.figure()
    for i in range(number_of_EoS):
        e = compute_energy_density(T_plot, EOS_set[i])
        plt.plot(T_plot, e / T_plot ** 4, '-')

    plt.xlim([0, 1.])
    # plt.ylim([0, 15])
    plt.xlabel(r"T (GeV)")
    plt.ylabel(r"$e/T^4$")
    plt.show()

    # plot cs^2 vs T
    plt.figure()
    for i in range(number_of_EoS):
        cs2 = compute_speed_of_sound_square(T_plot, EOS_set[i])
        plt.plot(T_plot, cs2, '-')

    plt.xlim([0, 1.])
    plt.ylim([0, 0.6])
    plt.xlabel(r"T (GeV)")
    plt.ylabel(r"$c_s^2$")
    plt.show()

    # invert the EoS tables
    e_list_EoS = []
    P_list_EoS = []
    T_list_EoS = []
    for i in range(number_of_EoS):
        if (i + 1) % 100 == 0:
            print(f"Inverting EoS table {i + 1}/{number_of_EoS}")
        e_list, P_list, T_list = invert_EoS_tables(T_plot, EOS_set[i])



        e_list_EoS.extend([e_list])
        P_list_EoS.extend([P_list])
        T_list_EoS.extend([T_list])

    # write the EoS to a file
    EoS_file_writer(e_list_EoS, P_list_EoS, T_list_EoS, f"EoS.pkl")


if __name__ == "__main__":
    ranSeed = 23
    number_of_EoS = 1000
    bLogFlag = True
    min_T_mask_region = 0.1
    max_T_mask_region = 0.4
    main(ranSeed, number_of_EoS, min_T_mask_region, max_T_mask_region, bLogFlag)