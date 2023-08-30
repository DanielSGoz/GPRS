# another file, weird stuff
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


def generate_probs(K):
	p = [random.random() for _ in range(K)]
	return normalize(p)


def calc_sum_min(q0, q1):
	SUM = 0.0
	for i in range(len(q0)):
		SUM = SUM + min(q0[i], q1[i])
	return SUM

def calc_quotient(q0, q1):
	SUM_MIN = 0.0
	SUM_MAX = 0.0
	for i in range(len(q0)):
		SUM_MIN = SUM_MIN + min(q0[i], q1[i])
		SUM_MAX = SUM_MAX + max(q0[i], q1[i])
	return SUM_MIN/SUM_MAX

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

def calc_all(q0=[], q1=[]):
	print("==========================================")
	K = 3
	if len(q0) == 0:
		q0 = generate_probs(K)
		print(q0)
	else:
		K = len(q0)
	if len(q1) == 0:
		q1 = generate_probs(K)
		print(q1)
	combi = [(q0[i], q1[i]) for i in range(K)]

	BEST = 1.0
	best_perm = combi[0]
	for perm in itertools.permutations(combi):
		for i in range(K):
			q0[i] = perm[i][0]
			q1[i] = perm[i][1]
		PROB = calc_prob(q0, q1)

		# print("==========================")
		# print(PROB)
		# print(perm)
		

		if PROB <= BEST:
			BEST = PROB
			best_perm = perm


	print(BEST)
	print(calc_quotient(q0, q1))
	# print("sum minim:")
	# print(calc_sum_min(q0, q1))
	# print(best_perm)
	# v = []
	# for i in range(K):
	# 	v.append(best_perm[i][0]/best_perm[i][1])
	# 	print(v[i])

	# for i in range(K - 2):
	# 	if (v[i] - v[i + 1])*(v[i + 1] - v[i + 2]) > 0:
	# 		return False
		
	# return True

calc_all([0.5, 0.1, 0.4], [0.5, 0.4, 0.1])

# calc_all([3/6, 1/6, 1/6, 1/6], [1/6, 1/6, 2/6, 2/6])
# calc_all([0.1, 0.1, 0.1, 0.1, 0.15, 0.15, 0.075, 0.075, 0.075, 0.075], [0.075, 0.075, 0.075, 0.075, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1])
# calc_all([1/5, 2/5, 2/5], [1/5, 3/5, 1/5])
# calc_all([1/10, 1/10, 2/10, 2/10, 2/10, 2/10], [1/10, 1/10, 3/10, 3/10, 1/10, 1/10])
# calc_all([1/15, 1/15, 1/15, 2/15, 2/15, 2/15, 2/15, 2/15, 2/15], [1/15, 1/15, 1/15, 3/15, 3/15, 3/15, 1/15, 1/15, 1/15])


# calc_all([0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.3, 0.2])
