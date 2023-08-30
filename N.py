import scipy
import random
import math


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

def generate_probs2(K):
	p = [2**(random.random()*10) for _ in range(K)]
        
	return normalize(p)


def uniform(K):
	return [1/K for _ in range(K)]


def exponential(a, K):
	return normalize([a**i for i in range(K)])



def order_two(p, q):
	K = len(p)

	unordered = [(p[i], q[i]) for i in range(K)]
	unordered.sort(key=lambda x: x[1]/x[0])

	for i in range(K):
		p[i] = unordered[i][0]
		q[i] = unordered[i][1]

	return (p, q)


# =================================================   THE MOST IMPORTANT FUNCTION   =================================================
def calculate_probN(p, q, S):
	K = len(p)

	A = [0.0 for _ in range(K)]
	B = [0.0 for _ in range(K)]
	C = [0.0 for _ in range(K)]


	A[K - 1] = q[K - 1]
	B[K - 1] = p[K - 1]
	C[0] = 0
	# C[K] = -infinity
	for i in range(K - 2, -1, -1):
		A[i] = q[i] + A[i + 1]
		B[i] = p[i] + B[i + 1]
		C[i + 1] = math.log(A[i] - q[i]/p[i]*B[i])

	probN = [[0.0 for _ in range(S)] for _ in range(K)]
	probN[K - 1][0] = p[K - 1]
	for s in range(1, S):
		probN[K - 1][s] = probN[K - 1][s - 1]*(1 - p[K - 1])

	for l in range(K - 2, -1, -1):
		yl = C[l] - C[l + 1]
		xl = yl * (1 - B[l])/B[l]
		coeff = math.exp(-(xl + yl))

		for s in range(S):
			SUM1 = 0.0
			for i in range(0, s + 1):
				SUM1 = SUM1*xl/(s + 1 - i)
				SUM1 = SUM1 + probN[l + 1][i]
			
			SUM2 = 0.0
			for i in range(S):
				SUM2 = SUM2*(xl + yl)/(S + s + 1 - i)
				SUM2 = SUM2 + 1
			for i in range(2, s + 2):
				SUM2 = SUM2 * xl/i

			probN[l][s] = coeff*(SUM1 + yl*SUM2)

	
	return probN[0]


def KL_divergence(p, q):
	return scipy.stats.entropy(p, q, base=2)


def entropy(p):
	return scipy.stats.entropy(p, base=2)


def expected_codelength(p):
	SUM = 0.0

	for i in range(len(p)):
		SUM = SUM + p[i] * math.log2(i + 1) 

	return SUM


def expected_power(p, a):
	SUM = 0.0

	for i in range(len(p)):
		SUM = SUM + p[i]*((i + 1)**a)

	return SUM


# Well, not exactly
def renyi_divergence(p, q, a):
	SUM = 0.0

	for i in range(len(p)):
		SUM = SUM + (p[i]**a)/(q[i]**(a - 1))

	return SUM


def infinity_divergence(p, q):
	MAX = 0.0

	for i in range(len(p)):
		MAX = max(MAX, p[i]/q[i])

	return math.log2(MAX)


def integral_a(p, q, a):
	K = len(p)
	SUM = 0.0
	PARTIAL = 1.0

	for i in range(K):
		diff = 0.0
		if i == 0:
			diff = q[0]/p[0]
		else:
			diff = q[i]/p[i] - q[i - 1]/p[i - 1]

		SUM = SUM + diff * (PARTIAL**(1 - a))
		PARTIAL = PARTIAL - p[i]

	return SUM


def generate_diffs(p, q, S, a):
	(p, q) = order_two(p, q)

	probN = calculate_probN(p, q, S)
	probN = normalize(probN)
	# print(probN)

	entropyN = entropy(probN)
	divergence = KL_divergence(q, p)
	divergencea = renyi_divergence(q, p, 1/(1 - a))
	divergenceinf = infinity_divergence(q, p)
	codelengthN = expected_codelength(probN)
	codelengthNa = expected_power(probN, a)
	integrala = integral_a(p, q, a)

	diff1 = (divergence + 2 + math.log2(divergence + 1)) - entropyN
	diff2 = entropyN - divergence - 1 - math.log2(divergence + 1)
	# diff3 = (1 + divergenceinf * math.log(2))*(divergencea**(1 - a)) - integrala
	# diff3 = codelengthNa - 1/(1 - a) * (1 + divergenceinf * math.log(2))*(divergencea**(1 - a))
	# diff3 = codelengthNa - 1/(1 - a) * (divergencea**(1 - a))
	# print(divergencea)
	diff3 = codelengthNa/(1/(1 - a) * divergencea)
	# diff3 = codelengthNa - (1/(1 - a) * divergencea)**(1 - a)
	# diff3 = codelengthNa - 1/(1 - a)*(divergencea)**(1 - a)
	# diff3 = codelengthNa - 1/(1 - a)*renyi_divergence(q, p, 1 + a)

	diff3 = codelengthNa/(renyi_divergence(q, p, a))


	# diff3 = codelengthNa - (1 + a)*renyi_divergence(q, p, 1 + a)
	# diff3 = integrala - divergencea**(1 - a)


	# print(diff1) # currently have an upper bound of 2 log2(e)
	# print(diff2) # currently have an upper bound of... 2log2(e) + 1 + log2(2log2(e) + 1)

	return (diff1, diff2, diff3)

def simulate_batch():
	min_diff1 = 1000
	# max_diff2 = -1000
	# max_diff3 = -1000
	max_p = None
	max_q = None


	for i in range(1000):
		K = 4
		S = 1000
		a = 0.9
		p = generate_probs(K)
		# p = [0.999, 0.001]
		# p = [0.49, 0.5, 0.01]
		# p = uniform(K)
		q = generate_probs(K)
		# q = [0.64042648, 1-0.64042648]
		# q = [1/3, 1/3, 1/3]
		# q = exponential(1 - (i + 1)/10000, K)
		# p = exponential(1 - (i + 1)/1000, K)
		

		(diff1, diff2, diff3) = generate_diffs(p, q, S, a)
		print(diff1)
		# print(diff2)
		# print(diff3)
		if not math.isnan(diff1) and diff1 < min_diff1:
			max_p = p
			max_q = q
			min_diff1 = diff1
		# if not math.isnan(diff2) and diff2 > max_diff2:
		# 	max_diff2 = diff2
		# if not math.isnan(diff3) and diff3 > max_diff3:
		# 	max_diff3 = diff3

	print("DONE!!")
	print(max_p)
	print(max_q)
	print(min_diff1)
	# print(max_diff2)
	# print(max_diff3)


def simulate_one():
	# e = 0.99
	# p = [e, 1 - e]
	# q = [0.620551507307884, 0.379448492692116]
	p = [0.95, 0.05]
	q = [0.5, 0.5]
	S = 1000
	a = 0.9
	(diff1, diff2, diff3) = generate_diffs(p, q, S, a)
	print(diff1)


simulate_batch()
# simulate_one()

##########   HALL OF FAME (i.e. best so far)  ############

# diff1: 0.6429 (the gamma one)
# diff2: 0.516464956776165 (something else whatever)
# diff3: -0.8678526198474881

# a = 1/2: 
# a = 1/4: -0.26210046805320597
# a = 1/5: -0.1928819855726842
# a = 1/10: -0.08009356740217943
