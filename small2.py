import math

lambd = 3**2
f1 = 0.1**2
df = 0.001

f2 = f1 + df

a1 = (math.exp(-1/2*(math.sqrt(f1) - math.sqrt(lambd))**2) + math.exp(-1/2*(math.sqrt(f1) + math.sqrt(lambd))**2))
b1 = (math.sqrt(f1)*math.sqrt(lambd)) * (math.exp(-1/2*(math.sqrt(f1) - math.sqrt(lambd))**2) - math.exp(-1/2*(math.sqrt(f1) + math.sqrt(lambd))**2))
c1 = f1*lambd*a1

a2 = (math.exp(-1/2*(math.sqrt(f2) - math.sqrt(lambd))**2) + math.exp(-1/2*(math.sqrt(f2) + math.sqrt(lambd))**2))
b2 = (math.sqrt(f2)*math.sqrt(lambd)) * (math.exp(-1/2*(math.sqrt(f2) - math.sqrt(lambd))**2) - math.exp(-1/2*(math.sqrt(f2) + math.sqrt(lambd))**2))
c2 = f2*lambd*a2

val1 = 1/2 + b1*(a1 - b1)/(2*a1*c1)
val2 = 1/2 + b2*(a2 - b2)/(2*a2*c2)

dvaldf = (val2 - val1)/df

w = a1*a1*c1 - a1*a1*b1 - 2*a1*b1*c1 - a1*b1*b1 + 2*b1*b1*b1
primm = w/(4*lambd*f1*f1*a1*a1*a1)

# print(dvaldf)
# print(primm)


def W(f, l):
    fs = math.sqrt(f)
    ls = math.sqrt(l)
    A = math.exp(-1/2*(fs - ls)**2)
    B = math.exp(-1/2*(fs + ls)**2)
    a = A + B
    b = fs*ls*(A - B)
    c = f*l*A
    return a*a*c - a*a*b - 2*a*b*c - a*a*b + 2*b*b*b

for x in range(100):
    for y in range(100):
        f = (x/40.0 + 0.1)**2
        l = (y/40.0 + 0.1)**2
        print(W(f, l))


