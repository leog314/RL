import random

def calc_real_prob(n: int) -> float:
    prod = 1
    for k in range(366-n, 366):
        prod *= k

    return 1-prod/365**n

def birthday(n: int, samples: int) -> float:
    su = 0
    for sample in range(samples):
        x = [random.randint(1, 365) for _ in range(n)]
        if True in [True if x.count(y) != 1 else False for y in x]: su += 1

    return su / samples

for k in range(100):
    print(k, calc_real_prob(k), birthday(k, 1000))
