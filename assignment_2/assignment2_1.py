# 3035150828P AARON,Sheshan Ryan

import numpy as np
import sys, getopt
from math import *

# Black Sholes Function
def BlackSholes(S,K,tau,sigma,r):
    if tau > 0:
        d1 = float(log(float(S)/K) + (r + 0.5*sigma*sigma)*(tau))/(sigma*sqrt(tau))
        d2 = float(d1 - sigma*sqrt(tau))
        N1 = 0.5*(1+erf(float(d1)/sqrt(2)))
        N2 = 0.5*(1+erf(float(d2)/sqrt(2)))
        C = S*N1-K*exp(-r*(tau))*N2
        Cdelta = N1
        P = C + K*exp(-r*tau) - S
        Pdelta = Cdelta - 1
    else:
        C = max(S-K,0)
        Cdelta = 0.5*(np.sign(S-K) + 1)
        P = max(K-S,0)
        Pdelta = Cdelta - 1
    return [C, Cdelta, P, Pdelta]


# Black Sholes Function
def BlackSholesParam(params):
    if 6 == len(params):
        sigma = float(params[4])/100
        r = float(params[5])/100
        return BlackSholes(params[0], params[1], (params[3] - params[2]), sigma, r)

def TestCase():
    # S, K, t, T, sig, r
    params = [
        [100, 100, 0, 0.5, 20, 1],
        [100, 120, 0, 0.5, 20, 1],
        [100, 100, 0, 1.0, 20, 1],
        [100, 100, 0, 0.5, 30, 1],
        [100, 100, 0, 0.5, 20, 2]
              ]
    for param in params:
        print "S, K, t, T, sigma, r"
        print param
        C, Cdelta, P, Pdelta = BlackSholesParam(param)
        print "Call option value %f" %C
        print "Put option value %f\n" %P

def main():
    CallPutFlag = ''
    S = 0
    K = 0
    t = 0
    T = 0
    r = 0.0
    sigma = 0.0
    test = False

    opts, args = getopt.getopt(sys.argv[1:], "S:K:t:T:r:v:", ["test"])
    for opt, arg in opts:
        if opt == '-S':
            S = float(arg)
        elif opt == '-K':
            K = float(arg)
        elif opt == '-t':
            t = float(arg)
        elif opt == '-T':
            T = float(arg)
        elif opt == '-r':
            r = float(arg)
        elif opt == '-v':
            sigma = float(arg)
        elif opt == '--test':
            test = True

    if test:
        print "===Running Test Cases===\n"
        TestCase()
    else:
        print "S %f, K %f, t %f, T %f, r %f, volatility %f" %(S, K, t, T, r, sigma)
        sigma = float(sigma)/100
        r = float(r)/100
        C, Cdelta, P, Pdelta = BlackSholes(S,K, (T-t), sigma, r)
        print "Call value = %f, Put value %f at t = %f" %(C, P, t)

if __name__ == "__main__":
    main()
