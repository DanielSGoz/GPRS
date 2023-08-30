import math
import numpy as np
import scipy
import matplotlib.pyplot as plt

from datetime import datetime
np.random.seed(int(datetime.now().timestamp() * 1000000) % (2**32))


def process1(mu, sigma2, N):
	d = mu.shape[0]
	samples = []
	lengths = []
	squares = 0.0

	index = -1
	while True:
		index = index + 1
		samples.append(np.random.normal(0, 1, d))
		lengths.append(np.sum(samples[index]**2))
		squares = squares + lengths[index]
		lengths[index] = math.sqrt(lengths[index])
		squareroot = math.sqrt(squares)

		for k in range(2**(index + 1), 2**(index + 2)):
			sum = 0.0
			for i in range(index + 1):
				if bin(k)[3 + i] == '0':
					# sum = sum + l[i]
					sum = sum + samples[i]*lengths[i]
				else:
					# sum = sum - l[i]
					sum = sum - samples[i]*lengths[i]
			# sum = sum / math.sqrt(d)
			sum = sum / squareroot

			if np.sum((sum - mu)**2) <= sigma2*(N**2):
				return index
			
# BAD SCHEME
def process2(mu, sigma2, N):
	d = mu.shape[0]
	samples = []
	lengths = []
	squares = 0.0
	sum = np.zeros(d)

	index = -1
	while True:
		index = index + 1
		samples.append(np.random.normal(0, 1, d))
		lengths.append(np.sum(samples[index]**2))
		squares = squares + lengths[index]
		lengths[index] = math.sqrt(lengths[index])
		squareroot = math.sqrt(squares)

		if np.dot(mu - (sum/squareroot), samples[index]) > 0:
			sum = sum + samples[index]*lengths[index]
		else:
			sum = sum - samples[index]*lengths[index]


		if np.sum(((sum/squareroot) - mu)**2) <= sigma2*(N**2):
			return index
		
# BAD SCHEME
def process3(mu, sigma2, N):
	d = mu.shape[0]
	samples = []
	squares = 0.0
	sum = np.zeros(d)

	index = -1
	while True:
		index = index + 1
		samples.append(np.random.normal(0, 1, d))
		squares = squares + 1
		squareroot = math.sqrt(squares)

		if np.dot(mu - (sum/squareroot), samples[index]) > 0:
			sum = sum + samples[index]
		else:
			sum = sum - samples[index]


		if np.sum(((sum/squareroot) - mu)**2) <= sigma2*(N**2):
			return index



def run_program(mu, sigma2, N, AMT):
	mu = np.array(mu)
	d = mu.shape[0]
	prob = scipy.stats.chi2.cdf(N*N, d)
	t = "mu: " + str(mu) + "   sigma2: " + str(sigma2) + "   N: " + str(N) + "   AMT: " + str(AMT)
	t = t + "   prob: " + str(prob)
	divergence = 1/2*(d*(sigma2 - 1 - math.log(sigma2)) + np.sum(mu**2))

	results = []
	for i in range(AMT):
		print(i)
		results.append(process2(mu, sigma2, N))

	maxim = 0
	for res in results:
		maxim = max(maxim, res)

	fig, axs = plt.subplots(2)

	x = range(maxim + 1)
	y = [0 for _ in x]
	for res in results:
		y[res] = y[res] + 1
	
	axs[0].bar(x, y)

	# brutally zero out the biggest (1 - wN) numbers
	rejects = math.ceil((1 - prob)*AMT)
	for _ in range(rejects):
		maxim = 0
		ind = 0
		for i in range(AMT):
			if results[i] > maxim:
				maxim = results[i]
				ind = i
		results[ind] = 0

	x2 = [i + 1 for i in range(len(results))]
	y2 = [0.0 for _ in range(len(results))]
	y2[0] = results[0]
	for i in range(1, len(results)):
		y2[i] = y2[i - 1] + results[i]

	t = t + "   expectation: " + str(y2[len(results) - 1]/len(results))
	t = t + "   divergence: " + str(divergence)

	for i in range(0, len(results)):
		y2[i] = y2[i]/(i + 1) - divergence

	t = t + "   cost: " + str(y2[len(results) - 1])

	axs[0].set_xlabel(t)

	axs[1].plot(x2, y2)
	plt.show()



# result = generate_samples(10, 10000, math.sqrt(10*2))

# plt.hist(result, bins=100, linewidth=0.5, edgecolor="white")
# plt.show()

run_program([10, 0, 0], 0.1, 3, 10000)
