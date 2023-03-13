rijeci = {}
fhand = open('song.txt')
for line in fhand:
    line = line.rstrip()
    line = line.lower()
    words = line.split()
    for word in words:
        if word in rijeci:
            rijeci[word] += 1
        else:
            rijeci[word] = 1

br = 0

for word in rijeci:
    if rijeci[word] == 1:
        print(word)
        br += 1

print(br)
fhand.close()
