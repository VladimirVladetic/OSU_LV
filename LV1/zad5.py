hamSum = 0
numberOfHam = 0
spamSum = 0
numberOfSpam = 0
exclamationPointSum = 0
fhand = open('SMSSpamCollection.txt')
for line in fhand:
    line = line.rstrip()
    words = line.split()
    if line.startswith("ham"):
        hamSum += len(words)-1
        numberOfHam += 1
    if line.startswith("spam"):
        spamSum += len(words)-1
        numberOfSpam += 1
        if line.endswith("!"):
            exclamationPointSum += 1

print(hamSum/numberOfHam)
print(spamSum/numberOfSpam)
print(exclamationPointSum)
