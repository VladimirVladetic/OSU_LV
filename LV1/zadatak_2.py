while(1):
    try:
        x = float(input())
        if(x >= 0.9 and x <= 1.0):
            print('A')
            break
        if(x >= 0.8 and x < 0.9):
            print('B')
            break
        if(x >= 0.7 and x < 0.8):
            print('C')
            break
        if(x >= 0.6 and x < 0.7):
            print('D')
            break
        if(x < 0.6):
            print('F')
            break
        else:
            print("Unesi broj izmedu 0.0 i 1.0")
    except:
        print("Unesi broj!")