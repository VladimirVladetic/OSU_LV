import math


def calculate(brojevi):
    print(len(brojevi))
    print(sum(brojevi)/len(brojevi))
    print(min(brojevi))
    print(max(brojevi))
    brojevi.sort()
    print(brojevi)
    
brojevi = []

while(1):
    unos = input()
    if(unos=="Done"):
        calculate(brojevi)
        break
    else:
        try:
            brojevi.append(float(unos))
        except:
            print("Unesi broj")
        