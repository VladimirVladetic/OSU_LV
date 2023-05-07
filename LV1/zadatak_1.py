import random
import math

def total_euro(sati, satnica):
    ukupno = sati*satnica
    print(ukupno)


while(1):
    try:
        sati = float(input())
        satnica = float(input())
        total_euro(sati, satnica)
        break
    except:
        print("Upisi broj")

