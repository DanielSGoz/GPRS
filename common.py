# a program dedicated to calculating the
# probability of a common first arrival
import scipy
import random
import math
import itertools

def normalize(p):
	K = len(p)

	SUM = 0.0
	for i in range(K):
		SUM = SUM + p[i]
	for i in range(K):
		p[i] = p[i] / SUM
        
	return p


def generate_probs1(K):
	p = [2**(random.random()*30) for _ in range(K)]
        
	return normalize(p)

def generate_probs2(K):
	p = [random.random() for _ in range(K)]
        
	return normalize(p)


def uniform(K):
	return [1/K for _ in range(K)]

# ==========================================================

def process(q):
	res = []
	SUM = 1.0
	for i in range(len(q)):
		res.append(q[i]/SUM)
		SUM = SUM - q[i]
	return res

def calc_prob(q0, q1):
	a = process(q0)
	b = process(q1)

	PROB = 0.0
	PROD = 1
	for i in range(len(a)):
		PROB = PROB + min(a[i], b[i]) * PROD
		PROD = PROD * (1 - max(a[i], b[i]))
	return PROB

def calc_all(q0, q1):
	K = len(q0)
	combi = [(q0[i], q1[i]) for i in range(K)]

	BEST = 1.0
	for perm in itertools.permutations(combi):
		for i in range(K):
			q0[i] = perm[i][0]
			q1[i] = perm[i][1]
		PROB = calc_prob(q0, q1)

		if PROB <= BEST:
			BEST = PROB

	return BEST



# ==========================================================


def get_events(s, q, p):
	K = len(p)
	events = []

	list = [(q[i]/p[i], i) for i in range(K)]
	list.sort()

	wP = 1
	wQ = 1
	sigma = 0
	t0 = 0
	for l in range(K - 1):
		t1 = list[l][0]
		index = list[l][1]

		sigma = sigma + 1/wP * math.log((wQ - t0 * wP)/(wQ - t1 * wP))

		events.append((sigma, s, index))

		t0 = t1
		wP = wP - p[index]
		wQ = wQ - q[index]

	return events


def entropy(p):
	return scipy.stats.entropy(p)

def KL_divergence(p, q):
	return scipy.stats.entropy(p, q)

def infinity_divergence(p, q):
	MAX = 0.0
	for i in range(len(p)):
		MAX = max(MAX, p[i]/q[i])
	return MAX

def total_variation(p, q):
	SUM = 0
	for i in range(len(p)):
		SUM = SUM + abs(p[i] - q[i])
	return SUM/2

def hellinger_distance(p, q):
	SUM = 0.0
	for i in range(len(p)):
		SUM = SUM + (math.sqrt(p[i]) - math.sqrt(q[i]))**2
	return SUM/2.0

def calc_quotient(q0, q1):
	SUM_MIN = 0.0
	SUM_MAX = 0.0
	for i in range(len(q0)):
		SUM_MIN = SUM_MIN + min(q0[i], q1[i])
		SUM_MAX = SUM_MAX + max(q0[i], q1[i])
	return SUM_MIN/SUM_MAX

def calc_matching_hypothesis(q0, q1):
	SUM = 0.0
	for i in range(len(q0)):
		SUM = SUM + q0[i]*q1[i]/(q0[i] + q1[i])
	return SUM

# very neat algorithm. Works in O(K log(K))
def calculate_prob(q0, q1, p):
	K = len(p)

	events = []
	a = 1
	x = 1
	y = 1
	t0 = 0
	cancelled = [False for _ in range(K)]
	amt_cancelled = 0

	# firstly generate the events.

	events = events + get_events(0, q0, p)
	events = events + get_events(1, q1, p)

	events.sort()

	# secondly calculate the integral

	inner = 0
	PROB = 0
	for (t1, s, index) in events:
		if amt_cancelled < K:
			PROB = PROB + a * math.exp(inner) * math.expm1((t1 - t0)*(a - x - y))/(a - x - y)
		inner = inner + (t1 - t0)*(a - x - y)
		t0 = t1

		if s == 0:
			x = x - p[index]
		else:
			y = y - p[index]

		if not cancelled[index]:
			cancelled[index] = True
			a = a - p[index]
			amt_cancelled = amt_cancelled + 1

	if amt_cancelled < K:
		PROB = PROB + a * math.exp(inner)/(x + y - a)
	return PROB

def simulate(q0=[], q1=[]):
	K = 3
	if len(q0) == 0:
		q0 = generate_probs2(K)
	else:
		K = len(q0)
	if len(q1) == 0:
		q1 = generate_probs2(K)
	# print(q0)
	# print(q1)

	opt = [-1 for _ in range(K)]
	PROB = 1
	for _ in range(1000):
		# p = [0.0000000001, random.random(), random.random()]
		# p = normalize(p)
		p = generate_probs1(K)
		candidate = calculate_prob(q0, q1, p)
		# print(candidate)
		if candidate < PROB:
			PROB = candidate
			opt = p

	# print("=========================")
	print(q0)
	print(q1)
	print(PROB)
	# print(opt)

	# lower = calc_all(q0, q1)
	# print(lower)

	# print(calc_quotient(q0, q1))
	# print(calc_matching_hypothesis(q0, q1))
	print(1/infinity_divergence(q1, q0))
	
	# print("STATISTICS:")
	# print("KL divergence[q0||q1]:")
	# # print(KL_divergence(q0, q1))
	# print(math.exp(KL_divergence(q0, q1)))
	# print("KL divergence[q1||q0]:")
	# # print(KL_divergence(q1, q0)) 
	# print(math.exp(KL_divergence(q1, q0)))
	# print("inf divergence[q0||q1] (no log):")
	# x = infinity_divergence(q0, q1)
	# print(infinity_divergence(q0, q1))
	# print("inf divergence[q1||q0] (no log):")
	# y = infinity_divergence(q1, q0)
	# print(infinity_divergence(q1, q0))
	# print("entropy[q0]:")
	# # print(entropy(q0))
	# print(math.exp(entropy(q0)))
	# print("entropy[q1]:")
	# # print(entropy(q1))
	# print(math.exp(entropy(q1)))
	# print("total variation distance:")
	# print(total_variation(q0, q1))
	# print("1 - total variation distance:")
	# print(1 - total_variation(q0, q1))
	# print("1 - 2*total variation distance - true:")
	# print(1 - 2*total_variation(q0, q1) - PROB)
	# print("with Hellinger distance:")
	# print(1 - math.sqrt(2*hellinger_distance(q0, q1)))
	# return 1 - 2*total_variation(q0, q1) - PROB
	

def simulate_one(q0, q1):
	for k in range(3, 100):
		eps = 2**(-k)
		p = [eps, 0.5 - eps, 0.5]
		prob = calculate_prob(q0, q1, p)
		# print(eps)
		print(prob)

simulate()

# simulate([0.4, 0.3, 0.3], [0.3, 0.3, 0.4])
# simulate([0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.075, 0.075, 0.075, 0.075], [0.075, 0.075, 0.075, 0.075, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1])
# simulate([0.1, 0.1, 0.1, 0.1, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075], [0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.1, 0.1, 0.1, 0.1])
# simulate([0.05, 0.05, 0.1, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15], [0.075, 0.075, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1])
# simulate([2/4, 1/4, 1/4], [1/4, 1/4, 2/4])
# simulate([3/6, 1/6, 1/6, 1/6], [1/6, 1/6, 1.5/6, 2.5/6])

# q0 = [3/6, 1/6, 1/6, 1/6]
# q1 = [1/6, 1/6, 1.5/6, 2.5/6]
# f = 1000000
# a = f
# b = 1
# c = f
# d = f*f
# s = a + b + c + d
# p = [a/s, b/s, c/s, d/s]

# print(calculate_prob(q0, q1, p))


# simulate_one([0.2, 0.2, 0.6], [0.2, 0.2, 0.6])
# a = 0.524642
# b = 0.32253
# PROB = simulate([0, 1/2, 1/2], [1 - a - b, a, b])
# exp = min(1/2*b/(1 - a) + a, 1/2*a/(1 - b) + b)
# print(PROB - exp)
# print(PROB - (3/2 - x)*x/(1 - x))
# simulate([0.8, 0.2], [0.4, 0.6])
