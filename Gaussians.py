import math
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
np.random.seed(int(datetime.now().timestamp() * 1000000) % (2**32))


def generate_samples(d, n, mu):
	res = []

	for _ in range(n):
		l = np.random.normal(0, 1, d)

		candidate = 10000
		for k in range(2**d, 2**(d + 1)):
			sum = 0.0
			for i in range(d):
				if bin(k)[3 + i] == '0':
					sum = sum + l[i]
				else:
					sum = sum - l[i]
			sum = sum / math.sqrt(d)

			if abs(sum - mu) < abs(candidate - mu):
				candidate = sum
			# res.append(sum)

		res.append(candidate)

	return res


result = generate_samples(10, 10000, math.sqrt(10*2/math.pi))

plt.hist(result, bins=100, linewidth=0.5, edgecolor="white")
plt.show()
