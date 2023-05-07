fhand=open('LV1\song.txt')
dictionary={}
br=0
for line in fhand:
    line=line.rstrip()
    line=line.lower()
    words=line.split()
    for word in words:
        if word in dictionary:
            dictionary[word]+=1
        else:
            dictionary[word]=1
    
print(dictionary)
for word in dictionary:
    if dictionary[word]==1:
        br+=1
print(br)