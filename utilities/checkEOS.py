#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys


def compute_cs2(T, P):
    dPdT = np.gradient(P, T, edge_order=2)
    e = T*dPdT - P
    cs2 = np.gradient(P, e, edge_order=2)
    return cs2


def main(databaseFile: str, args: list[int]) -> None:
    hotQCD = np.loadtxt("../EoS_hotQCD.dat")
    hotQCD[:, 1] = hotQCD[:, 1]*(hotQCD[:, 0]**4)
    Nskip = 7
    hotQCD_cs2 = compute_cs2(hotQCD[::Nskip, 0], hotQCD[::Nskip, 1])

    with open(databaseFile, "rb") as pf:
        eosData = pickle.load(pf)

    # plot cs^2 vs. T
    fig = plt.figure()
    plt.plot(hotQCD[::Nskip, 0], hotQCD_cs2, lw=3, label="hotQCD")

    for eosId in args:
        indexKey = f"{eosId:04d}"

        eos = eosData[indexKey]
        e = eos[:, 0]**4.       # GeV^4
        # compute the speed of sound squared
        cs2 = np.gradient(eos[:, 1], e, edge_order=2)

        plt.plot(eos[:, 2], cs2, label=indexKey)

    plt.legend(loc=4, ncol=4)
    plt.xlim([0.05, 0.5])
    plt.ylim([0, 0.5])
    plt.xlabel(r"$T$ (GeV)")
    plt.ylabel(r"$c_s^2$")
    plt.tight_layout()
    plt.savefig("checkEOS_cs2.pdf")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} database EoSIds")
        sys.exit(1)
    else:
        databaseFile = str(sys.argv[1])
        args = []
        for arg in sys.argv[2:]:
            if "-" in arg:
                l, h = arg.split("-")
                args += range(int(l), int(h))
            else:
                args += [int(arg)]

    main(databaseFile, args)
