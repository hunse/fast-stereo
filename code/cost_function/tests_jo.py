# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 11:22:36 2015

@author: jorchard
"""

import numpy as np
import math
import matplotlib.pyplot as plt

# Parameters for two Gaussian components
mu = np.array([0.30, 1.1])
sigma = np.array([0.2, 0.25])
w = np.array([0.25, 0.75])  # weights (should add to 1)

def gaussian(x, mu, sig):
    return np.exp( -(x - mu)**2 / (2 * sig**2) ) / np.sqrt(2.*math.pi) / sig

def pdf(delta, mu, sigma, weight):
    v = np.zeros(np.shape(delta))
    for m,s,w in zip(mu, sigma, weight):
        v += w*gaussian(delta, m, s)
    return v

delta = np.linspace(0, 1.5, 200)
p = pdf(delta, mu, sigma, w)

plt.figure(1)
plt.clf()
plt.plot(delta, p)

plt.title('PDF')

#%% Evaluate the symmetric quadratic cost (uncertainty)

# Find the min d-value using the analytical formula (global min)
dmin = sum(mu*w)


# For comparison, we also evaluate the cost function over a range of d-values
# and look for the minimum.

def quadratic_cost(d, p, delta):
    c = sum(p * (d-delta)**2)
    return c*(delta[1]-delta[0])

c = []
dmin = 0.0
for d in delta:
    c.append(quadratic_cost(d, p, delta))


# Given a d-value, we can evaluate the cost two different ways.
cmin = quadratic_cost(dmin, p, delta)
cmin_formula = sum( w * (sigma**2 + (dmin-mu)**2) )


c = np.array(c)
plt.figure(2)
plt.clf()
plt.plot(delta, c)
plt.title('Symmetric Uncertainty')


print('\nSYMMETRIC COST')
print('True min cost is '+str(cmin)+' at '+str(dmin))
print('Analytical min cost is '+str(cmin_formula)+' at '+str(dmin))









#%% Now the ASYMMETRIC cost

w_over = 2.0
w_under = 0.5

# This is the ideal version of the cost function, evaluating the integral
# using the Heaviside functions to choose which weight to use (hence the
# if-statement).
def asym_cost(d, p, delta, w_up, w_down):
    c = 0.
    for p_i, delta_i in zip(p, delta):
        if d<delta_i:
            c += w_down * p_i * (delta_i - d)**2
        else:
            c += w_up * p_i * (delta_i - d)**2
    return c*(delta[1]-delta[0])


# This is the version of the asymmetric cost function that avoids having
# to compute the integral (notice no sum over delta). Instead, it's based
# on a weighted sum of the means, etc.
def asym_cost_approx(d, mu, sigma, weight, w_up, w_down):
    c = 0.
    for m,s,w in zip(mu, sigma, weight):
        h = 1. / (1. + np.exp(-(d-m)/0.1))
        blah = w*(s**2 + (d-m)**2)
        p_down = w_down*(1.-h)*blah
        p_up   = w_up  *h*blah
        c += p_up + p_down
    return c

#%% Let's evaluate and plot the two ways of computing the cost.

casym = []
casym_formula= []
dmin = 0.0
for d in delta:
    casym.append(asym_cost(d, p, delta, w_over, w_under))
    casym_formula.append(asym_cost_approx(d, mu, sigma, w, w_over, w_under))

casym = np.array(casym)
casym_formula = np.array(casym_formula)

plt.figure(3)
plt.clf()
plt.plot(delta, casym, 'b', label='True')
plt.hold(True)
plt.plot(delta, casym_formula, 'r--', label='Formula')
plt.title('Asymmetric Uncertainty')
plt.plot(delta[argmin(casym)], min(casym), 'kx')
plt.hold(False)

print('\nASYMMETRIC COST')
print('True min cost is '+str(min(casym))+' at '+str(delta[argmin(casym)]))
print('Gloabal min of approx is '+str(min(casym_formula))+' at '+str(delta[argmin(casym_formula)]))


#%% Analytical approximation of the minimum.

# Fit the approximation with a cubic polynomial

import scipy.linalg as slin

# Choose a fixed set of samples
d_samples = np.array([0.0, 0.1, 0.2, 0.3])

D = np.vander(d_samples)  # Vandermonde matrix

Dinv = np.linalg.inv(D)  # Invert and store for later (repeated) use.
#perm, lower, upper = slin.lu(D)  # Another way of storing the solved system


#%% Solve for the approximate minimum

# Where do we want the sequence of samples to start?
# "offset" is where d_samples[0] is.
offset = 0.5

# Compute the corresponding cost values using the appoximate (non-integrating)
# version of the cost function.
U = []
for ds in d_samples:
    U.append(asym_cost_approx(ds+offset, mu, sigma, w, w_over, w_under))

# And solve the system to get the cubic coefficients.
# In this case, p3[0] is the coef for the cubic term, p3[3] the constant term.
p3 = np.dot(Dinv, np.array(U))


# You can also use polyfit, but it will waste computation by solving the
# same system every time.
#p3 = np.polyfit(delta[idxs], casym_formula[idxs], 3)

# Create a polynomial object to make plotting easier.
p = np.poly1d(p3)

# Plot the polynomial approximation, and indicate the min.
plt.hold(True)
plt.plot(delta, p(delta-offset), 'g:', label='Polynomial')

# For what it's worth, this is the derivative of the polynomial.
dp3 = np.array([p3[0]*3, p3[1]*2, p3[2]])
dp = np.poly1d(dp3)

#plt.plot(delta, dp(delta-offset), 'g:')

# Find the roots of the derivative
r1 = (-2.*p3[1] + np.sqrt(4.*p3[1]**2-4.*3.*p3[0]*p3[2])) / (2.*3.*p3[0])
r2 = (-2.*p3[1] - np.sqrt(4.*p3[1]**2-4.*3.*p3[0]*p3[2])) / (2.*3.*p3[0])

# Choose the root that is reasonable.
r = r1 if (r1>0 and r1<3.) else r2

r = r + offset # compensate for the offset

poly_min = asym_cost_approx(r, mu, sigma, w, w_over, w_under)
print('Analytical approx min of '+str(poly_min)+ ' at '+str(r))

plt.plot(d_samples+offset, U, 'gx', label='Samples')
plt.plot(r, poly_min, 'ko', label='_blah')

plt.legend(loc='best')

plt.hold(False)



#
