# 3035150828P AARON,Sheshan Ryan

import sys, getopt
import numpy as np
import csv
import datetime
from math import *

def ExtendedBlackSholes(S, K, r, q, sigma, tau):
    if tau > 0:
        d1 = float(log(float(S)/K) + ((r-q) + 0.5*sigma*sigma)*(tau))/(sigma*sqrt(tau))
        d2 = float(d1 - sigma*sqrt(tau))
        N1 = 0.5*(1+erf(float(d1)/sqrt(2)))
        N2 = 0.5*(1+erf(float(d2)/sqrt(2)))
        S = S*exp(-q*(tau))
        C = S*N1-K*exp(-r*(tau))*N2
        Cdelta = N1
        Cvega = float(S*sqrt(tau)*exp(-0.5*d1*d1))/sqrt(2*pi)
        P = C + K*exp(-r*tau) - S
        Pdelta = Cdelta - 1
        Pvega = Cvega
    else:
        C = max(S-K,0)
        Cdelta = 0.5*(np.sign(S-K) + 1)
        Cvega = 0
        P = max(K-S,0)
        Pdelta = Cdelta - 1
        Pvega = 0
    return  [C, Cdelta, Cvega, P, Pdelta, Pvega]

def NewtonMethod(S, K, t, T, r, q, optionValue, optionType='C'):
    C_true = 0
    P_true = 0
    tau = T-t
    if optionType == 'C':
        C_true = optionValue
        if (C_true < max((S*exp(-q*tau) - K*exp(-r*tau)), 0)) or (C_true > S*exp(-q*tau)):
            return None
    elif optionType == 'P':
        P_true = optionValue
        if (P_true < max((K*exp(-r * tau) - S*exp(-q*tau)), 0)) or (P_true > K*exp(-r * tau)):
            return None

    # starting value
    sigmahat = sqrt(2*abs(float(log(float(S)/K) + (r - q)*tau)/tau))

    tol = 1e-8
    sigma = sigmahat
    sigmadiff = 1
    n = 1
    nmax = 100

    while (sigmadiff >= tol and n < nmax):
        C, Cdelta, Cvega, P, Pdelta, Pvega = ExtendedBlackSholes(S,K,r,q, sigma, tau)

        if optionType == 'C':
            increment = (C-C_true)/Cvega
        else:
            increment = (P-P_true)/Pvega

        sigma = sigma - increment
        n = n+1
        sigmadiff = abs(increment)

    return [sigmahat, sigma]

def getSigma(S, K, t, T, r, q, optionValue, optionType):
    rv = NewtonMethod(S, K, t, T, r, q, optionValue, optionType)
    if rv is not None:
        return rv[1]
    return 'NaN'

def TestCase():
    # S, K, t, T, sig, r
    params = [
        [100, 100, 0, 0.5, 0.2, 0.01, 0],
        [100, 120, 0, 0.5, 0.2, 0.01, 0],
        [100, 100, 0, 1.0, 0.2, 0.01, 0],
        [100, 100, 0, 0.5, 0.3, 0.01, 0],
        [100, 100, 0, 0.5, 0.2, 0.02, 0]
              ]
    for param in params:
        print "S, K, t, T, sigma, r, q"
        print param
        S = param[0]
        K = param[1]
        t = param[2]
        T = param[3]
        sigma_true = param[4]
        r = param[5]
        q = param[6]
        C_true, Cdelta, Cvega, P_true, Pdelta, Pvega = ExtendedBlackSholes(S, K,r, q, sigma_true, T-t)
        sigmahat, sigma = NewtonMethod(S,K,t,T,r,q, C_true, 'C')
        print 'C_true %f sigma_true %f, sigmahat %f, sigma %f' %(C_true, sigma_true, sigmahat, sigma)
        sigmahat, sigma = NewtonMethod(S,K,t,T,r,q, P_true, 'P')
        print 'P_true %f sigma_true %f, sigmahat %f, sigma %f' %(P_true, sigma_true, sigmahat, sigma)

def addToOutput(outputDict, Strike, BidVol, AskVol, optionType):
    BidVolP, AskVolP, BidVolC, AskVolC = 'NaN', 'NaN', 'NaN', 'NaN'
    if Strike in outputDict:
        BidVolP, AskVolP, BidVolC, AskVolC = outputDict[Strike]
    if 'C' == optionType:
        BidVolC,AskVolC = BidVol, AskVol
    elif 'P' == optionType:
        BidVolP,AskVolP = BidVol, AskVol
    outputDict[Strike] = BidVolP, AskVolP, BidVolC, AskVolC
    return outputDict

def getSpotPrice(subBidAsk):
    bid, ask = subBidAsk['510050']['bid'], subBidAsk['510050']['ask']
    return float(bid + ask)/2

def addBidAskToDict(subMarketData, row):
    if row['Bid1'] is not None and row['Ask1'] is not None:
        subMarketData[row['Symbol']] = {"bid": float(row['Bid1']), "ask": float(row['Ask1'])}
    return subMarketData

def run():
    # read instruments csv
    instruments = list(csv.DictReader(open('instruments.csv', 'r')))

    bidAskDict = {31: dict(), 32: dict(), 33: dict()}

    # read market data csv
    marketdata = list(csv.DictReader(open('marketdata.csv', 'r')))
    for row in marketdata:
        print row['LocalTime']
        LocalTime = datetime.datetime.strptime(str(row['LocalTime']).split(' ')[1], "%H:%M:%S.%f").time()
        if LocalTime <= datetime.time(9,31,00,00):
            bidAskDict[31] = addBidAskToDict(bidAskDict[31], row)
            print '31'
        elif LocalTime <= datetime.time(9,32,00,00):
            bidAskDict[32] = addBidAskToDict(bidAskDict[32], row)
            print '32'
        elif LocalTime <= datetime.time(9,33,00,00):
            bidAskDict[33] = addBidAskToDict(bidAskDict[33], row)
            print '33'

    outputDict = {31: dict(), 32: dict(), 33: dict()}
    for time, subBidAsk in bidAskDict.iteritems():
        # get spot price
        S = getSpotPrice(subBidAsk)
        # calculate implied volatilies for all options
        for instrument in instruments:
            if instrument['Type'] == 'Option' and instrument['Symbol'] in subBidAsk:
                K = float(instrument['Strike'])
                optionType = instrument['OptionType']
                T = float(24 - 16)/365
                r = 0.04
                q = 0.2
                symbol = instrument['Symbol']
                bid = subBidAsk[symbol]['bid']
                bidVol = getSigma(S, K, 0, T, r, q, bid, optionType)
                ask = subBidAsk[symbol]['ask']
                askVol = getSigma(S, K, 0, T, r, q, ask, optionType)

                outputDict[time] = addToOutput(outputDict[time], K, bidVol, askVol, optionType)
        filename = '%d.csv' %time
        with open(filename, 'wb') as outputfile:
            outputfile.write('Strike,BidVolP, AskVolP, BidVolC, AskVolC\n')
            for Strike, vols in outputDict[time].iteritems():
                BidVolP, AskVolP, BidVolC, AskVolC = vols
                outputfile.write('%s, %s, %s, %s, %s\n' %(Strike, BidVolP, AskVolP, BidVolC, AskVolC))


    print "Done!"

def main():
    opts, args = getopt.getopt(sys.argv[1:], "", ["test", "run"])
    for opt, arg in opts:
        if opt == '--test':
            TestCase()
        elif opt == '--run':
            run()
    print "Done!"

if __name__ == "__main__":
    main()