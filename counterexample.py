import math
import scipy
import numpy

# accuracy, the number of nodes for Gaussian quadrature
# also the number of terms in infinite sum
# the program runs in O(ACC^2)
ACC = 5000



alpha = 0.9 # a number in open interval (0, 1)
gamma = -1/(1 - alpha)
beta = -1 - gamma

m = 0.00000001 # a number in open interval (0, 1). Preferably as close to 0 as possible.
M = ((m + beta)/((1 + beta) * m))**(1/alpha) * (1 + beta) * m - beta
C1 = (m + beta)**(-gamma)
C2 = 1/beta * C1
C3 = -math.log(1 - m)


# in reality, wP(h) =
# 1						if h <= m
# the formula below		if m <= h <= M
# 0						if M <= h
def wP(h):
	global beta, gamma, C1
	return C1 * (h + beta)**gamma


# in reality, wQ(h) - h * wP(h) = 
# 1 - h					if h <= m
# the formula below		if m <= h <= M
# 0						if M <= h
def wQ_hwP(h):
	global beta, gamma, C2
	return C2 * ((h + beta)**(gamma + 1) - (M + beta)**(gamma + 1))


# argument in [m, M]
def sigma(h):
	global ACC, beta, gamma, m, C3
	result = []

	for el in h:
		result.append(C3 + scipy.integrate.fixed_quad(lambda x: 1/wQ_hwP(x), m, el, n=ACC)[0])
	
	return numpy.array(result)


# calculate the sum over (1 + k)^alpha / k! * z^k, for k from 0 to ACC - 1
def infinite_sum(z):
	global ACC, alpha
	result = []

	for el in z:
		SUM = 0.0

		for k in range(ACC, -1, -1):
			SUM = SUM * el / (k + 1)
			SUM = SUM + (k + 1)**alpha
		
		result.append(SUM)
	
	return numpy.array(result)


# computes the argument to infinite_sum. Argument x in [m, M]
def integrate_fraction(x):
	global ACC, beta, gamma, m, M
	result = []

	for el in x:
		result.append(scipy.integrate.fixed_quad(lambda y: beta*((m + beta)**gamma - (y + beta)**gamma)/((y + beta)**(gamma + 1) - (M + beta)**(gamma + 1)), m, el, n=ACC)[0])

	return numpy.array(result)



# this is the expectation E[N^alpha]
def LHS():
	global ACC, alpha, m, M
	return m + scipy.integrate.fixed_quad(lambda x: wP(x) * (1 + integrate_fraction(x))**alpha, m, M, n=ACC)[0]
	# return m + scipy.integrate.fixed_quad(lambda x: numpy.exp(-sigma(x)) * wP(x)/wQ_hwP(x) * infinite_sum(integrate_fraction(x)), m, M, n=ACC)[0]



# this is the Renyi alpha-divergence
def RHS():
	global ACC, alpha, beta, gamma, m, M, C1
	return (m**(beta + 1)/(beta + 1) + C1 * scipy.integrate.fixed_quad(lambda x: (x + beta)**gamma * x**beta, m, M, n=ACC)[0])


# print results
def results():
	rhs = RHS()
	lhs = LHS()
	print(rhs)
	print(lhs)
	print(lhs / rhs) # I want to check whether rhs / lhs tends to infinity for fixed alpha, and m tending to 0 (i.e. M tending to infinity)

results()
