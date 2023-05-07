
ham_counter=0
ham_word_counter=0
spam_word_counter=0
spam_counter=0
exclamation_counter=0
fhand=open('LV1\SMSSpamCollection.txt',encoding="utf8")
for line in fhand:
    line=line.rstrip()
    words=line.split()
    if line.startswith("ham"):
        ham_counter+=1
        ham_word_counter+=len(words)-1
    else:
        spam_counter+=1
        spam_word_counter+=len(words)-1
        if line.endswith("!"):
            exclamation_counter+=1
        
print(ham_word_counter/ham_counter)
print(spam_word_counter/spam_counter)
print(exclamation_counter)


    