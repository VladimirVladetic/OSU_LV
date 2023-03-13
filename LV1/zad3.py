def racunaj(brojevi):
    print(len(brojevi))
    print(sum(brojevi)/len(brojevi))
    print(max(brojevi))
    print(min(brojevi))
    brojevi.sort()
    print(brojevi)


brojevi = []
while(1):
    temp = input()
    try:
        brojevi.append(float(temp))
    except:
        if(temp == "Done"):
            racunaj(brojevi)
            break
        else:
            print("Unosi samo brojeve")
