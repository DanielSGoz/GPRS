from math import exp
from math import sqrt
from math import cos
from math import sin
from math import pi

e = 1

# needs e >= 0
def integrate_Riemann(c, s, N):
	global e

	dt = pi/N

	res = 0
	for k in range(0, N):
		t = (2*k + 1)*dt/2
		val = sin(t)**e
		val = val*(c*cos(t))**s
		val = val*exp(c*cos(t))

		res = res + val

	return res* pi/N

def single2(x, y, z, c, s):
	return exp(c*(x + y + z))*(((1 - x*x)*(1 - y*y)*(1 - z*z)) ** (s/2.0))

def single3(x, y, z, c, s):
	if -2 < x + y and x + y < 2 and -2 < y + z and y + z < 2 and -2 < z + x and z + x < 2:
		return exp(c*(x + y + z))*x*y*z* ((2 - x - y)*(2 + x + y)*(2 - y - z)*(2 + y + z)*(2 - z - x)*(2 + z + x)) ** (s/2.0)
	else:
		return 0
	
def ultra_symmetric3(x, y, z, c, s):
	return single3(x, y, z, c, s) + single3(-x, y, z, c, s) + single3(x, -y, z, c, s) + single3(-x, -y, z, c, s) + single3(x, y, -z, c, s) + single3(-x, y, -z, c, s) + single3(x, -y, -z, c, s) + single3(-x, -y, -z, c, s)

def integrate2(dzx, dyx, c, s, N):
	dt = (2 - dzx)/N

	res = 0
	for k in range(0, N):
		t = -1 + (2*k + 1)*dt/2

		x = t
		y = t + dyx
		z = t + dzx
		u = t + dzx - dyx
		res = res + single2(x, y, z, c, s)
		res = res - single2(x, u, z, c, s)

	return res * (2*dyx - dzx)


def single(x, y, z):
	global e    
	return exp(x + y + z) * (-e * z*z + e*z - 2*y*z*z + e*y*z + 2*x*y*z)
    
def ultra_symmetric2(x, y, z, c, s):
	if x > y or y > z:
		return 0
	else:
		u = x + z - y
		# return (y - u)*(single2(x, y, z, c) - single2(x, u, z, c) - single2(-x, -y, -z, c) + single2(-x, -u, -z, c))
		return (y - u)*(single2(x, y, z, c, s) - single2(x, u, z, c, s) - single2(-x, -y, -z, c, s) + single2(-x, -u, -z, c, s))

	

def symmetric(x, y, z):
	global e
	# return exp(x + y + z) * (-2*e*(x*x + y*y + z*z) + 2*e*(x + y + z) - 2*(x*y*y + x*x*y + y*z*z + y*y*z + z*z*x + z*x*x) + 2*e*(x*y + y*z + z*x) + 12*x*y*z)
	# return (2*(x*x*x + y*y*y + z*z*z) + 2*(x*x + y*y + z*z) - 3*(x*y*y + x*x*y + y*z*z + y*y*z + z*z*x + z*x*x) - 2*(x*y + y*z + z*x) - 2*(x + y + z) + 12*x*y*z)
	# return exp(x + y + z) * (2*(x*x*x + y*y*y + z*z*z) - 3*(x*y*y + x*x*y + y*z*z + y*y*z + z*z*x + z*x*x) + 12*x*y*z)
	return exp(x + y + z) * (x + y - 2*z)*(y + z - 2*x)*(z + x - 2*y)*(x + y + z)**4

def ultra_symmetric(x, y, z):
	return symmetric(x, y, z) + symmetric(-x, y, z) + symmetric(x, -y, z) + symmetric(-x, -y, z) + symmetric(x, y, -z) + symmetric(-x, y, -z) + symmetric(x, -y, -z) + symmetric(-x, -y, -z)


# table = [0.3, -0.3, 2, -2]
# table = [0.3, -0.3, 1, -1, 3, -3]
table1 = [-1, -0.9, -0.5, -0.1, 0, 0.1, 0.5, 0.9, 1]
table2 = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5]
table3 = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 1, 1.1, 1.5, 1.7, 1.8, 1.9, 1.95, 1.99]

# print(ultra_symmetric(2, 0.3, 0.3)/48.0)

# for x in table1:
# 	for y in table1:
# 		for z in table1:
# 			# print(ultra_symmetric(x, y, z))
# 			for c in table2:
# 				if x < y and y < z:
# 					# if ultra_symmetric2(x, y, z, c) < -0.000001:
# 				# print(ultra_symmetric3(x, y, z, c, 2))
# 					# print(integrate2(z - x, y - x, c, 1, 1000))
# 					if integrate2(z - x, y - x, c, 1, 1000) < -0.00001:
# 						print(integrate2(z - x, y - x, c, 3, 1000))

for dzx in table3:
	for dyx in table3:
		for c in table2:
			if dyx < dzx:
				if integrate2(dzx, dyx, c, 5, 1000) < -0.00001:
					print(integrate2(dzx, dyx, c, 5, 1000))



# print(integrate2(1.5, 0.2, 1, 1, 1000))

def evaluate_w(c):
	global e
	I1 = integrate_Riemann(c, 0, 1000)
	I2 = integrate_Riemann(c, 1, 1000)
	I3 = integrate_Riemann(c, 2, 1000)
	I4 = integrate_Riemann(c, 3, 1000)
	# I5 = integrate_Riemann(c, 4, 1000)
	# w = -e*I1*I1*I3 + e*I1*I1*I2 - 2*I1*I2*I3 + e*I1*I2*I2 + 2*I2*I2*I2

	# w = I1*I1*I4 + I1*I1*I3 - 3*I1*I2*I3 - I1*I2*I2 - I1*I1*I2 + 2*I2*I2*I2
	# w = I1*I1*I4 - 3*I1*I2*I3 + 2*I2*I2*I2
	# w = c*c*I3 - (e + 1)*(I4 - 2*I3 + 2*I2)
	# w = I1*I2 - I1*I3 + I2*I2
	w = I1*I4 - I2*I3

	# after applying inequality:
	# w = I1*I1*I4 - 3*I1*I2*I3 + 2*I2*I2*I2
	return w

# for c in range(1, 400):
# 	print(evaluate_w(c/100.0))

# for x in table:
# 	print(ultra_symmetric(x, x, x))

# def W(f, l):
#     fs = sqrt(f)
#     ls = sqrt(l)
#     A = exp(-1/2*(fs - ls)**2)
#     B = exp(-1/2*(fs + ls)**2)
#     a = A + B
#     b = fs*ls*(A - B)
#     c = f*l*A
#     return a*a*c - a*a*b - 2*a*b*c - a*a*b + 2*b*b*b

# print(W())
