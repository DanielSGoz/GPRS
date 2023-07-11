import math
import tensorflow as tf
import tensorflow_probability as tfp
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

# ==========================================================

# we try to calculate top-down. If one is not defined, we
# approximate it using its dependencies.
# the input to p, q, r we assume is a real number.

sigma = None

wP = None
wQ = None

r = None

p = None
q = None

# the stupidly easy integration technique
# Riemann integration, from a to b, in N steps
def integrate_Riemann(f, a, b, N):
	if N == 1:
		return f((a + b)/2)

	dt = (b - a)/(N - 1)

	res = 0
	for k in range(0, N):
		res = res + f(a + (2*k + 1)*dt/2)

	return res* (b - a)/N


# Here is the Normal-Normal diagonal covariance case

# muP, muQ are vectors, varP, varQ are scalars
def initialize_normal(muP, varP, muQ, varQ):
	global sigma, wP, wQ, r, p, q
	muP = tf.convert_to_tensor(muP, dtype=jnp.float32)
	muQ = tf.convert_to_tensor(muQ, dtype=jnp.float32)
	varP = tf.convert_to_tensor(varP, dtype=jnp.float32)
	varQ = tf.convert_to_tensor(varQ, dtype=jnp.float32)
	assert(varP > varQ)

	d = muP.shape[0]

	muZ = tf.math.subtract(tf.math.scalar_mul(varP, muQ), tf.math.scalar_mul(varQ, muP))
	varZ = varP * varQ / (varP - varQ)

	Z = (varP/varQ * 2*math.pi * varZ) ** (d / 2)
	Z = Z * math.exp(tf.reduce_sum(tf.square(tf.math.subtract(muP, muQ)))/(2*(varP - varQ)))

	C = varZ*(2*math.log(Z) - d*math.log(2*math.pi * varZ))

	# for now, works only for 1-D case.
	normalP = tfd.Normal(loc=muP, scale=math.sqrt(varP))
	normalQ = tfd.Normal(loc=muQ, scale=math.sqrt(varQ))
	normalZ = tfd.Normal(loc=muZ, scale=math.sqrt(varZ))

	non_central_chi2P = tfd.NoncentralChi2(d, tf.reduce_sum(tf.square(tf.math.subtract(tf.math.scalar_mul(1/math.sqrt(varP), muZ), muP))))
	non_central_chi2Q = tfd.NoncentralChi2(d, tf.reduce_sum(tf.square(tf.math.subtract(tf.math.scalar_mul(1/math.sqrt(varQ), muZ), muQ))))

	p = normalP.prob
	q = normalQ.prob

	r = lambda x: Z*normalZ.prob(x)

	wP = lambda h: non_central_chi2P.cdf((-2*varZ*math.log(h) + C)/varP)
	wQ = lambda h: non_central_chi2Q.cdf((-2*varZ*math.log(h) + C)/varQ)

	sigma = lambda h: integrate_Riemann(lambda x: 1/(wQ(x) - x*wP(x)), 0, h, 10)

# P is uniform distribution in interval [aP, bP], Q is Uniform in [aQ, bQ].
def initialize_uniform(aP, bP, aQ, bQ):
	global sigma, wP, wQ, r, p, q
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

# P is uniform in [0, 1], Q is triangular
def initialize_triangular(a, c, b):
	global sigma, wP, wQ, r, p, q
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





# ==========================================================

initialize_normal([0], 2, [1], 1)
# initialize_uniform(-3, 3, -2, 2)
# initialize_triangular(0.2, 0.5, 0.6)

# x = tf.math.scalar_mul(KLK, tf.convert_to_tensor(jnp.linspace(0.001, 0.999, 20), dtype=jnp.float32))
x = tf.convert_to_tensor(jnp.linspace(-5, 5, 200), dtype=jnp.float32)
y = tf.map_fn(lambda t: sigma(r(t)), x)
plt.axis([-5, 5, 0, 10])
plt.plot(x, y)
plt.show()


# plt.scatter(data[:, 0], data[:, 1], color='blue', alpha=0.4)
# plt.axis([-5, 5, 0, 10])
# plt.title("Data set")
# plt.show()
