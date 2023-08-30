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

		t0 = t1
		wP = wP - p[index]
		wQ = wQ - q[index]

		events.append((sigma, s, index))

	return events

def calculate_all(q0, q1, p):
	K = len(p)

	events = []
	a = 1.0
	x = 1.0
	y = 1.0
	t0 = 0.0
	minims = [-1.0 for _ in range(K)]
	cancelled = [False for _ in range(K)]

	# firstly generate the events.

	events = events + get_events(0, q0, p)
	events = events + get_events(1, q1, p)

	events.sort()

	# secondly calculate the integral

	inner = 0.0
	PROB0 = 0.0
	PROB1 = 0.0
	for (t1, s, index) in events:
		additional = math.exp(inner) * math.expm1((t1 - t0)*(a - x - y))/(a - x - y)
		PROB0 = PROB0 + a * additional
		PROB1 = PROB1 + additional
		inner = inner + (t1 - t0)*(a - x - y)
		t0 = t1

		if minims[index] < -0.5:
			minims[index] = p[index]/q0[index] * PROB1

		if s == 0:
			x = x - p[index]
		else:
			y = y - p[index]

		if not cancelled[index]:
			cancelled[index] = True
			a = a - p[index]

	additional = math.exp(inner)/(x + y - a)
	PROB0 = PROB0 + a * additional
	PROB1 = PROB1 + additional

	for index in range(K):
		if minims[index] < -0.5:
			minims[index] = p[index]/q0[index] * PROB1

	return (PROB0, minims)

def simulate(q0=[], q1=[]):
	K = 3
	if len(q0) == 0:
		q0 = generate_probs2(K)
	else:
		K = len(q0)
	if len(q1) == 0:
		q1 = generate_probs2(K)

	minims = [1 for i in range(K)]
	for _ in range(100000):
		p = generate_probs2(K)
		(_, candidates) = calculate_all(q0, q1, p)
		# print(candidates)
		for i in range(K):
			minims[i] = min(minims[i], candidates[i])

	print("=========================")
	print(q0)
	print(q1)
	print(minims)


def hiper_check(q0=[], q1=[], p=[]):
	K = 10
	if len(q0) == 0:
		q0 = generate_probs2(K)
	else:
		K = len(q0)
	if len(q1) == 0:
		q1 = generate_probs2(K)
	if len(p) == 0:
		p = generate_probs2(K)
	print(q0)
	print(q1)
	print(p)

	(integral, candidates) = calculate_all(q0, q1, p)
	print(candidates)

	against = 0.0
	for i in range(K):
		against = against + q0[i] * min(1, candidates[i])

	print(integral)
	print(against)


# hiper_check()
# simulate([2/4, 1/4, 1/4], [1/3, 1/3, 1/3])
# simulate([1/3, 2/3], [2/3, 1/3])
# simulate([1/6, 1/6, 2/6, 2/6], [2/6, 2/6, 1/6, 1/6])
# simulate([1/9, 1/9, 1/9, 2/9, 2/9, 2/9], [2/9, 2/9, 2/9, 1/9, 1/9, 1/9])
# simulate([1/12, 1/12, 1/12, 1/12, 2/12, 2/12, 2/12, 2/12], [2/12, 2/12, 2/12, 2/12, 1/12, 1/12, 1/12, 1/12])
simulate([1/15, 1/15, 1/15, 1/15, 1/15, 2/15, 2/15, 2/15, 2/15, 2/15], [2/15, 2/15, 2/15, 2/15, 2/15, 1/15, 1/15, 1/15, 1/15, 1/15])

