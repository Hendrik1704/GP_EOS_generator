from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import numpy as np
import matplotlib.pyplot as plt


def compute_derivative(x, y):
    """This function compute dy/dx using finite difference"""
    dydx = np.zeros(len(y))
    dydx[1:-1] = (y[2:] - y[:-2])/(x[2:] - x[:-2])
    dydx[0] = (y[1] - y[0])/(x[1] - x[0])
    dydx[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])
    return(dydx)


def compute_energy_density(T, P):
    """This function computes energy density"""
    dPdT = compute_derivative(T, P)
    e = T * dPdT - P    # energy density
    return e


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
    """This filter ensures the speed of sound square is between 0 and 1"""
    cs2 = compute_speed_of_sound_square(T, P)
    cs2 = cs2[2:-2]
    index = (cs2 > 0.) & (cs2 < 1.)
    physical_points = cs2[index]

    if len(cs2) == len(physical_points):
        return True
    else:
        return False


def is_a_physical_eos(T, P) -> bool:
    if not derivative_filter(T, P):
        return False

    dPdT = compute_derivative(T, P)
    if not derivative_filter(T, dPdT):
        return False

    if not speed_sound_squared_filter(T, P):
        return False

    return True


def main(ranSeed: int, number_of_EoS: int, bLogFlag: bool) -> None:
    # load the training data
    training_data = np.loadtxt("EoS_hotQCD.dat")

    # print out the minimum and maximum values of the training data x_values
    T_min = np.min(training_data[:, 0])  # the min of actual data points
    T_max = np.max(training_data[:, 0])  # the max of actual data points
    #print(f"Minimum of the datapoints is: {T_min}")
    #print(f"Maximum of the datapoints is: {T_max}")

    # load the full EOS table for verification
    validation_data = np.loadtxt("EoS_hotQCD_full.dat")

    # set the random seed
    if ranSeed >= 0:
        randomness = np.random.seed(ranSeed)
    else:
        randomness = np.random.seed()

    # define GP kernel
    kernel = RBF(length_scale=0.2, length_scale_bounds=(1e-3, 100))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-5)

    if bLogFlag:
        # train GP with log(T) vs. log(P/T^4) because all the quantities
        # are positive
        x_train = np.log(training_data[:, 0]).reshape(-1, 1)
        gpr.fit(x_train, np.log(training_data[:, 1]))
        print(f"GP score: {gpr.score(x_train, np.log(training_data[:, 1]))}")
        T_GP = np.linspace(np.log(T_min), np.log(T_max), 100).reshape(-1, 1)
        T_plot = np.exp(T_GP.flatten())
    else:
        x_train = training_data[:, 0].reshape(-1, 1)
        gpr.fit(x_train, training_data[:, 1])
        print(f"GP score: {gpr.score(x_train, training_data[:, 1])}")
        T_GP = np.linspace(T_min, T_max, 100).reshape(-1, 1)
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
                P_GP = np.exp(sample_i)*(T_plot**4)       # convert to P
            else:
                P_GP = sample_i*(T_plot**4)       # convert to P
            if is_a_physical_eos(T_plot, P_GP):
                EOS_set.append(P_GP)
                iSuccess += 1
                if iSuccess == number_of_EoS:
                    break
        iter += nsamples_per_batch
        print(f"Sample success rate: {float(iSuccess)/iter:.3f}")

    # make verification plots
    plt.figure()
    plt.scatter(training_data[:, 0], training_data[:, 1],
                marker='x', color='r', s=20, label="training data")
    plt.scatter(validation_data[:, 0], validation_data[:, 1],
                marker='+', color='b', s=20, label="validation data")
    for i in range(number_of_EoS):
        plt.plot(T_plot, EOS_set[i]/T_plot**4, '-')

    plt.legend()
    plt.xlim([0, 0.5])
    plt.ylim([0, 4.5])
    plt.xlabel(r"T (GeV)")
    plt.ylabel(r"$P/T^{4}$")
    plt.show()

    # plot e vs T
    plt.figure()
    for i in range(number_of_EoS):
        e = compute_energy_density(T_plot, EOS_set[i])
        plt.plot(T_plot, e/T_plot**4, '-')

    plt.xlim([0, 0.5])
    plt.ylim([0, 15])
    plt.xlabel(r"T (GeV)")
    plt.ylabel(r"$e/T^4$")
    plt.show()

    # plot cs^2 vs T
    plt.figure()
    for i in range(number_of_EoS):
        cs2 = compute_speed_of_sound_square(T_plot, EOS_set[i])
        plt.plot(T_plot[2:-2], cs2[2:-2], '-')

    plt.xlim([0, 0.5])
    plt.ylim([0, 1.0])
    plt.xlabel(r"T (GeV)")
    plt.ylabel(r"$c_s^2$")
    plt.show()


if __name__ == "__main__":
    ranSeed = 23
    number_of_EoS = 1000
    bLogFlag = True
    main(ranSeed, number_of_EoS, bLogFlag)
