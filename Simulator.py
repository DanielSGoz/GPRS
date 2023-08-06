import math
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
import collections
tfd = tfp.distributions
import matplotlib.pyplot as plt

#try:
#  tf.compat.v1.enable_eager_execution()
#except ValueError:
#  print("no eager execution")




print()
print("==========  BEGIN  ==========")
print()

error = 0.00000001

mode = None
maximum = None

# ==========================================================

# we try to calculate top-down. If one is not defined, we
# approximate it using its dependencies.
# the input to p, q, r we assume is a real number.

sigma = None

wP = None
wP_prim = None
wQ = None

K = None

r = None

p = None
q = None

simulator = None

# the stupidly easy integration technique
# Riemann integration, from a to b, in N steps
def integrate_Riemann(f, a, b, N):
	dt = (b - a)/N

	res = 0
	for k in range(0, N):
		res = res + f(a + (2*k + 1)*dt/2)

	return res* (b - a)/N

def increasing_binsearch(f, a, b, v):
	global error

	while a + error < b:
		h = (a + b)/2
		if(f(h) > v):
			b = h
		else:
			a = h

	return (a + b)/2

def decreasing_binsearch(f, a, b, v):
	global error

	while a + error < b:
		h = (a + b)/2
		if(f(h) < v):
			b = h
		else:
			a = h

	return (a + b)/2

def increasing_function_inverse(f, v):
	a = 1
	while f(2*a) < v:
		a = a*2
	while f(a) > v:
		a = a/2

	return increasing_binsearch(f, a, 2*a, v)

def decreasing_function_inverse(f, v):
	a = 1
	while f(2*a) > v:
		a = a*2
	while f(a) < v:
		a = a/2

	return decreasing_binsearch(f, a, 2*a, v)

	

# Here is the Normal-Normal diagonal covariance case

# muP, muQ are vectors, varP, varQ are scalars
def initialize_normal(muP, varP, muQ, varQ):
	global sigma, wP, wP_prim, wQ, r, p, q, simulator, mode, maximum, K
	muP = tf.convert_to_tensor(muP, dtype=jnp.float32)
	muQ = tf.convert_to_tensor(muQ, dtype=jnp.float32)
	varP = tf.convert_to_tensor(varP, dtype=jnp.float32)
	varQ = tf.convert_to_tensor(varQ, dtype=jnp.float32)
	assert(varP > varQ)

	d = muP.shape[0]

	muZ = (varP * muQ - varQ * muP)/(varP - varQ)
	mode = muZ
	varZ = varP * varQ / (varP - varQ)

	K = math.exp(tf.reduce_sum(tf.square(tf.math.subtract(muP, muQ)))/(2*(varP - varQ)))
	Z = (varP/varQ * 2*math.pi * varZ) ** (d / 2) * K
	maximum = (varP/varQ) ** (d / 2) * K
	C = varZ*(2*math.log(Z) - d*math.log(2*math.pi * varZ))

	# for now, works only for 1-D case.
	normalP = tfd.Normal(loc=muP, scale=math.sqrt(varP))
	normalQ = tfd.Normal(loc=muQ, scale=math.sqrt(varQ))
	normalZ = tfd.Normal(loc=muZ, scale=math.sqrt(varZ))

	non_central_chi2P = tfd.NoncentralChi2(d, tf.reduce_sum(tf.square(tf.math.subtract(muZ, muP))) / varP)
	non_central_chi2Q = tfd.NoncentralChi2(d, tf.reduce_sum(tf.square(tf.math.subtract(muZ, muQ))) / varQ)

	p = normalP.prob
	q = normalQ.prob

	r = lambda x: Z*normalZ.prob(x)

	wP = lambda h: non_central_chi2P.cdf((-2*varZ*math.log(h) + C)/varP)
	wP_prim = lambda h: (-2*varZ)/(varP*h) * non_central_chi2P.prob((-2*varZ*math.log(h) + C)/varP)
	wQ = lambda h: non_central_chi2Q.cdf((-2*varZ*math.log(h) + C)/varQ)

	sigma = lambda h: integrate_Riemann(lambda x: 1/(wQ(x) - x*wP(x)), 0, h, 10)

	simulator = normalP

# P is uniform distribution in interval [aP, bP], Q is Uniform in [aQ, bQ].
def initialize_uniform(aP, bP, aQ, bQ):
	global sigma, wP, wQ, r, p, q, simulator
	aP = tf.convert_to_tensor(aP, dtype=jnp.float32)
	bP = tf.convert_to_tensor(bP, dtype=jnp.float32)
	aQ = tf.convert_to_tensor(aQ, dtype=jnp.float32)
	bQ = tf.convert_to_tensor(bQ, dtype=jnp.float32)
	assert(aQ >= aP)
	assert(bQ <= bP)

	C = (bQ - aQ)/(bP - aP)

	uniformP = tfd.Uniform(low=aP, high=bP)
	uniformQ = tfd.Uniform(low=aQ, high=bQ)

	p = uniformP.prob
	q = uniformQ.prob

	r = lambda x: q(x) / p(x)

	wP = lambda h: C
	wQ = lambda h: 1

	sigma = lambda h: integrate_Riemann(lambda x: 1/(wQ(x) - x*wP(x)), 0, h, 10)

	simulator = uniformP

# P is uniform in [0, 1], Q is triangular
def initialize_triangular(a, c, b):
	global sigma, wP, wQ, r, p, q, simulator
	a = tf.convert_to_tensor(a, dtype=jnp.float32)
	c = tf.convert_to_tensor(c, dtype=jnp.float32)
	b = tf.convert_to_tensor(b, dtype=jnp.float32)
	assert(0 < a)
	assert(a < c)
	assert(c < b)
	assert(b < 1)

	L = b - a

	uniformP = tfd.Uniform(low=0, high=1)
	triangularQ = tfd.Triangular(low=a, high=b, peak=c)

	p = uniformP.prob
	q = triangularQ.prob

	r = lambda x: q(x)

	wP = lambda h: L - L*L/2 * h
	wQ = lambda h: 1 - L*L/4 * h*h

	sigma = lambda h: integrate_Riemann(lambda x: 1/(wQ(x) - x*wP(x)), 0, h, 100)

	simulator = uniformP


# ==========================================================

# the original GPRS algorithm
def get_sample_from_Q():
	global simulator

	ExponentialT = tfd.Exponential(rate=1)

	X = None
	T = 0.0
	while True:
		T = T + ExponentialT.sample().numpy()
		X = simulator.sample()

		if T < sigma(r(X)):
			break
		
	return X


def plot_samples(N):
	x = tf.map_fn(lambda _: get_sample_from_Q(), tf.convert_to_tensor(range(N), dtype=jnp.float32))
	y = tf.map_fn(lambda x: q(x), x)

	# plt.axis([0, 1, -1, 1])
	plt.scatter(x, y)
	plt.show()


def plot_sigma():
	global mode
	x = tf.convert_to_tensor(jnp.linspace(-10, 10, 20), dtype=jnp.float32)

	initialize_normal([0], 1, [0], 0.5)

	y1 = tf.map_fn(lambda t: sigma(r(t + mode)), x)

	initialize_normal([0], 1, [0.1], 0.5)

	y2 = tf.map_fn(lambda t: sigma(r(t + mode)), x)

	# plt.axis([-2, 2, 0, 3])
	plt.plot(x, y1)
	plt.plot(x, y2)
	plt.show()


def check_inequality():
	global mode, maximum
	# checking the inequality, only for 1-D case.

	initialize_normal([0], 1, [0], 0.96)

	t1 = tf.math.scalar_mul(maximum, tf.convert_to_tensor(jnp.linspace(0.001, 0.01, 100), dtype=jnp.float32))

	x1 = tf.map_fn(lambda t: sigma(t), t1)
	y1 = tf.map_fn(lambda t: wP(t), t1)

	initialize_normal([0], 1, [0.1], 0.96)
 
	t2 = tf.math.scalar_mul(maximum, tf.convert_to_tensor(jnp.linspace(0.001, 0.01, 100), dtype=jnp.float32))

	x2 = tf.map_fn(lambda t: sigma(t), t2)
	y2 = tf.map_fn(lambda t: wP(t), t2)

	x0 = tf.convert_to_tensor(jnp.linspace(0.001, 0.999, 100), dtype=jnp.float32)

	plt.plot(x1, y1, label='1')
	plt.plot(x2, y2, label='2')
	plt.legend()
	plt.show()

def extend_x(x1, y1, x2):
	x3 = []
	y3 = []

	i1 = -1
	i2 = -1

	while True:
		if (i1 == len(x1) - 1) or (i2 == len(x2) - 1):
			break
		elif x1[i1 + 1] < x2[i2 + 1]:
			i1 = i1 + 1

			if not (i2 == -1):
				x3.append(x1[i1])
				y3.append(y1[i1])
		else:
			i2 = i2 + 1

			if not (i1 == -1):
				x3.append(x2[i2])
				if x1[i1 + 1] == x1[i1]:
					x3.append(y1[i1])
				else:
					y3.append((y1[i1]*(x1[i1 + 1] - x2[i2]) + y1[i1 + 1]*(x2[i2] - x1[i1]))/(x1[i1 + 1] - x1[i1]))
		
	return (x3, y3)


def form_quotient(x1, y1, x2, y2):
	if x1[0] > x1[-1]:
		x1 = np.flip(x1)
		y1 = np.flip(y1)
	if x2[0] > x2[-1]:
		x2 = np.flip(x2)
		y2 = np.flip(y2)

	(x3, y3) = extend_x(x1, y1, x2)
	(x4, y4) = extend_x(x2, y2, x1)

	while len(x3) < len(x4):
		x4.pop()
		y4.pop()
	while len(x4) < len(x3):
		x3.pop()
		y3.pop()

	for i in range(len(x3)):
		y3[i] = y3[i]/y4[i]

	return (x3, y3)


def check_inequality2():
	global mode, maximum, K

	var = 0.5
	mu = 0.1
	d = 1
	center = [0] * d
	center_mu = [0] * d
	center_mu[0] = mu

	initialize_normal(center, 1, center, var)

	t = tf.math.scalar_mul(maximum, tf.convert_to_tensor(jnp.linspace(0.001, 0.999, 100), dtype=jnp.float32))
	x1 = tf.map_fn(lambda h: wP(h), t)
	# y1 = tf.map_fn(lambda h: (wQ(h) - h*wP(h))*(-wP_prim(h)), t)
	# y1 = tf.map_fn(lambda h: -wP_prim(h), t)

	y1 = tf.map_fn(lambda h: h, t)
	# y1 = tf.map_fn(lambda h: -wP_prim(h)*h, t)

	# y1 = tf.map_fn(lambda h: (wQ(h) - h*wP(h)), t)


	initialize_normal(center, 1, center_mu, var)
	x2 = tf.map_fn(lambda h: wP(h), t)
	# y2 = tf.map_fn(lambda h: (wQ(h) - h*wP(h))*(-wP_prim(h)), t)
	# y2 = tf.map_fn(lambda h: -wP_prim(h), t)

	y2 = tf.map_fn(lambda h: h, t)
	# y2 = tf.map_fn(lambda h: -wP_prim(h)*h, t)

	# y2 = tf.map_fn(lambda h: (wQ(h) - h*wP(h)), t)

	(x3, y3) = form_quotient(x1.numpy(), y1.numpy(), x2.numpy(), y2.numpy())

	# plt.plot(x1, y1, label='1')
	# plt.plot(x2, y2, label='2')
	# plt.plot(x3, y3, label='3')
	plt.plot(x1, x2, label='4')
	# I want label 1 to be under label 2.
	plt.legend()
	plt.show()

def find_inverse(x, y, v):
	# we assume that y is increasing.
	# if y[0] <= v <= y[-1], then we find an approximate x[i] s.t. y[i] = v.
	# the we print x[i].

	res = -100

	if v <= y[0]:
		res = x[0]
	elif v >= y[-1]:
		res = x[-1]
	else:
		for i in range(len(y) - 2, -1, -1):
			if y[i] <= v:
				c = 0

				if y[i] == y[i + 1]:
					c = 1/2
				else:
					c = (v - y[i])/(y[i + 1] - y[i])

				res = c*x[i + 1] + (1 - c)*x[i]
				break
			
	return res


def check_inequality3():
	coeff0 = 0
	coeff1 = 5
	d = 1

	non_central_chi2_0 = tfd.NoncentralChi2(d, coeff0)
	non_central_chi2_1 = tfd.NoncentralChi2(d, coeff1)

	t = tf.convert_to_tensor(jnp.linspace(0, 1, 1000), dtype=jnp.float32)
	y0 = tf.map_fn(lambda x: non_central_chi2_0.cdf(x), t)
	y1 = tf.map_fn(lambda x: non_central_chi2_1.cdf(x), t)

	t = t.numpy()
	y0 = y0.numpy()
	y1 = y1.numpy()


	b = min(y0[-1], y1[-1])
	a = 0
	s = jnp.linspace(a, b, 1000)
	l = []
	x0 = []
	x1 = []
	for y in s:
		# x0 = find_inverse(t, y0, y) # this is v0^{-1}(y)
		# x1 = find_inverse(t, y1, y) # this is v1^{-1}(y)
		# l.append(non_central_chi2_1.prob(x1)/non_central_chi2_0.prob(x0))
		x0.append(find_inverse(t, y0, y))
		x1.append(find_inverse(t, y1, y))

	plt.plot(x0, x1, label='1')
	# plt.plot(s, l, label='1')
	plt.legend()
	plt.show()


def check_inequality4():
	coeff = 0.1
	d = 2

	non_central_chi2_1 = tfd.NoncentralChi2(d, coeff)

	t = tf.convert_to_tensor(jnp.linspace(0, 30, 300), dtype=jnp.float32)
	y = tf.map_fn(lambda x: non_central_chi2_1.prob(x)/(1 - non_central_chi2_1.cdf(x)), t)

	t = t.numpy()
	y = y.numpy()

	plt.plot(t, y, label='1')
	plt.legend()
	plt.show()

	



	# a = 1
	# t = tf.convert_to_tensor(jnp.linspace(0.001, 2*a, 100), dtype=jnp.float32)
	# v1 = tf.map_fn(lambda x: non_central_chi2_2.cdf(x), t)
	# v1_a = non_central_chi2_2.cdf(a)
	# y = tf.map_fn(lambda x: max(0, 2*v1_a - x), v1)
	# v0 = tf.map_fn(lambda x: non_central_chi2_1.cdf(x), y)

	# plt.plot(t, v0, label='4')
	# plt.legend()
	# plt.show()


def check_inequality5():
	global mode, maximum, K

	var = 0.9
	d = 1
	center = [0] * d

	initialize_normal(center, 1, center, var)

	print("MAXIMUM:")
	print(maximum)
	print("")

	t = tf.math.scalar_mul(maximum, tf.convert_to_tensor(jnp.linspace(0.001, 0.999, 100), dtype=jnp.float32))
	x1 = tf.map_fn(lambda h: h, t)
	# y1 = tf.map_fn(lambda h: (wQ(h) - h*wP(h))*(-wP_prim(h)), t)
	# y1 = tf.map_fn(lambda h: -wP_prim(h), t)

	y1 = tf.map_fn(lambda h: sigma(h), t)
	# y1 = tf.map_fn(lambda h: -wP_prim(h)*h, t)

	# y1 = tf.map_fn(lambda h: (wQ(h) - h*wP(h)), t)
	x2 = tf.map_fn(lambda h: h, t)
	y2 = tf.map_fn(lambda h: maximum*math.log(maximum/(maximum - h)), t)


	plt.plot(x1, y1, label='1')
	plt.plot(x2, y2, label='2')
	# plt.plot(x3, y3, label='3')
	# plt.plot(x4, y4, label='4')
	# I want label 1 to be under label 2.
	plt.legend()
	plt.show()








# ==============================================================================

# initialize_triangular(0.1, 0.4, 0.5)

# initialize_normal([0], 1, [0], 0.5)

# plot_samples(50)

# check_inequality()

# plot_sigma()

# check_inequality2()

# check_inequality3()

# check_inequality4()

check_inequality5()
