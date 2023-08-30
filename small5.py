import random

EPS = 0.0000000001


def ineq1(s0, s1, u0, u1, w0, w1):
    global EPS
    LHS = min(u0/(1 - s0), u1/(1 - s1)) + min(w0/(1 - s0 - u0), w1/(1 - s1 - u1)) * (1 - max(u0/(1 - s0), u1/(1 - s1)))
    RHS = min((u0 + w0)/(1 - s0), (u1 + w1)/(1 - s1))
    print(LHS - RHS)
    return LHS - EPS <= RHS

def ineq2(s0, s1, u0, u1, w0, w1):
    global EPS
    LHS = (1 - max(u0/(1 - s0), u1/(1 - s1))) * (1 - max(w0/(1 - s0 - u0), w1/(1 - s1 - u1)))
    RHS = 1 - max((u0 + w0)/(1 - s0), (u1 + w1)/(1 - s1))
    print(LHS - RHS)
    return LHS - EPS <= RHS

def check_all():
	for _ in range(1000000):
		s0 = random.random()
		u0 = random.random()
		w0 = random.random()
		sum = random.random() + s0 + u0 + w0
		s0 = s0/sum
		u0 = u0/sum
		w0 = w0/sum
        
		s1 = random.random()
		u1 = random.random()
		w1 = random.random()
		sum = random.random() + s1 + u1 + w1
		s1 = s1/sum
		u1 = u1/sum
		w1 = w1/sum

		CHECK1 = ineq1(s0, s1, u0, u1, w0, w1)
		CHECK2 = ineq2(s0, s1, u0, u1, w0, w1)

		if not CHECK1:
			print("WRONG INEQ 1:")
			print(s0)
			print(u0)
			print(w0)
			print(s1)
			print(u1)
			print(w1)
			break

		if not CHECK2:
			print("WRONG INEQ 2:")
			print(s0)
			print(u0)
			print(w0)
			print(s1)
			print(u1)
			print(w1)
			break

check_all()
