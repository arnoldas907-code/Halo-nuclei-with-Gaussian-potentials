import numpy as np
#import sys

# =========================
# Module subr (Fortran)
# =========================

# Parameters and globals
dp = float
irel = 0

mn = 1.0
mnucl = 938.91897
m2inv = 0.5 / mnucl

alphar4 = np.zeros(3, dtype=float)
alphar18 = np.zeros(3, dtype=float)
alphar = np.zeros(3, dtype=float)
alpham = np.zeros((3, 3), dtype=float)


def initPar():
    global alpham

    alpham[0, :] = [1.11545,  0.969431, -0.765903]
    alpham[1, :] = [-1.36841, -0.999263,  1.07024]
    alpham[2, :] = [1.92987,  0.773169, -0.834312]


def Kappastar(mc, rnn, rnc):
    global alphar

    rnnc12 = np.array([
        1.0,
        np.sqrt(rnc / rnn),
        rnc / rnn
    ], dtype=float)

    mm1 = np.array([
        1.0,
        mn / mc,
        (mn / mc) ** 2
    ], dtype=float)

    alphar = alpham @ mm1

    kpx = np.sum(alphar * rnnc12)
    kpx = 1.0 / (rnn * kpx)
    return kpx

def kappalinear(mc, rnn, rnc, ab1, ann1):
    mu2 = mc / (mc + 1.0) * 2.0

    kpx = Kappastar(mc, rnn, rnc)

    beta = 1.604 * (0.5 * mu2 * (kpx * rnc) ** 2) ** (-0.25)

    kpx = kpx * (1.0 + 0.674793 * rnn * ann1 + 0.681179 * rnc * ann1)
    beta = beta * (1.0 - 0.689315 * rnn * ann1 - 1.25297 * rnc * ann1)

    rncab = rnc*ab1

    kappa = kpx * (1.0 + beta * rncab)
    Ener = 41.47 * kappa ** 2 / mu2 
    return kappa, Ener

#def main():
#    initPar()

