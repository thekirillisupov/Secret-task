from random import randint
from random import random

n = int(1e5)
with open('input.txt', 'w') as test_file:
    test_file.write(str(n)+'\n')
    for _ in range(n):
        test_file.write(f'{randint(0,100000)*random()} {randint(0,100000)*random()}'+'\n')