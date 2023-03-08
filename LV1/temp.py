import random
import math
for i in range(10):
    x = random.random()
    y = math.sin(x)
    print('Broj', x, 'Sin(broj):', y)

fhand = open('example.txt')
for line in fhand:
    line = line.rstrip()
    print(line)
    words = line.split()
fhand.close()


def print_hello():
    print(" Hello world ")


print_hello()
