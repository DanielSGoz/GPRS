# another file, weird stuff
import random
import math
import itertools
import numpy

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

def calc_all(p1, p2):
	print("==========================================")
	K = 20
	q0 = [p1/K for _ in range(K)] + [(1 - p1)/K for _ in range(K)]
	q1 = [p2/K for _ in range(K)] + [(1 - p2)/K for _ in range(K)]

	perm = []
	for i in range(2*K):
		if i%2 == 0:
			perm.append((q0[i//2], q1[i//2]))
		else:
			perm.append((q0[(i-1)//2 + K], q1[(i-1)//2 + K]))

	BEST = 1.0
	best_perm = perm
	for step in range(1000000):
		print(step)
		for i in range(2*K):
			q0[i] = perm[i][0]
			q1[i] = perm[i][1]
		PROB = calc_prob(q0, q1)

		# print("==========================")
		# print(PROB)
		# print(perm)

		if PROB <= BEST:
			BEST = PROB
			best_perm = perm

		perm = numpy.random.permutation(perm)

	print(BEST)
	print(best_perm)
	v = []
	for i in range(2*K):
		v.append(best_perm[i][0]/best_perm[i][1])
		print(v[i])

	# for i in range(K - 2):
	# 	if (v[i] - v[i + 1])*(v[i + 1] - v[i + 2]) > 0:
	# 		return False
		
	# return True

calc_all(1/3, 1/2)

# calc_all([0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.3, 0.2])
