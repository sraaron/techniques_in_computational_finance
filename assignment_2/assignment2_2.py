# 3035150828P AARON,Sheshan Ryan

import numpy as np
from math import *

def main():
    print "Assignment 2_2"
    for iter in range (0, 5):
        filename = 'q2_iteration_%d.csv' %iter
        with open(filename, 'wb') as outputfile:
            # Generate 100 samples of X and Y
            # mu = 0
            # sigma = 1
            X = np.random.randn(100) # np.random.normal(mu, sigma, 100)
            Y = np.random.randn(100) # np.random.normal(mu, sigma, 100)

            # Generate samples of Z
            Z = []
            rho = 0.5
            for i in range(0, 100):
                Z.append(rho * X[i] + sqrt(1 - rho * rho)*Y[i])

            # Calculate sample Correlation Coefficient rho(X,Z)
            # corr_coeffA = np.cov(X, Z) / sqrt(np.var(X) * np.var(Z))
            corr_coeffB = np.corrcoef(X, Z)
            print 'correlation coefficient rho(X,Z): %f' %corr_coeffB[0][1]
            diff = (corr_coeffB[0][1]-rho)
            print 'correlation coefficient rho(X,Z)-rho: %f' %diff

            # write header
            outputfile.write("X, Y, Z, correlation_coefficient_rho(X_Z), theoretical_rho, rho(X_Z) - rho\n")
            # write values to file
            for i in range(0, 100):
                if i==0:
                    outputfile.write('%f, %f, %f, %f, %f, %f \n'%(X[i], Y[i], Z[i], corr_coeffB[0][1], rho, diff) )
                else:
                     outputfile.write('%f, %f, %f, , ,\n' %(X[i], Y[i], Z[i]))

if __name__ == "__main__":
    main()