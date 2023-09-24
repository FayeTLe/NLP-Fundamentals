from spacy.lang.en import English
import math 

nlp = English()

trainSet = nlp("aaaa bbb aaa bbb ababab acacac cacacad ccca dcdcdccdddccc cbbcbccb acac bdbdbd dbdbdb dadaaddadadddaaa ddd ccc bbb cdcdcdcd ccddcd dcdcdcdc")

testSet = nlp("aabcacddbcbbdaadda")

# Function to train the model
def unigram (data):
    countAs = 0
    countBs = 0
    countCs = 0
    countDs = 0
    totalWord = 0
    
    # Find the total word in the dataset
    for token in data:
        for char in token.text:
            totalWord += 1
    
    # Calculate the probability   
    for token in data:
        for char in token.text:
            if char == "a":
                countAs += 1
            if char == "b":
                countBs += 1
            if char == "c":
                countCs += 1
            if char == "d":
                countDs += 1
                
    probA = countAs/totalWord
    probB = countBs/totalWord
    probC = countCs/totalWord
    probD = countDs/totalWord
    
    result = [probA, probB, probC, probD]
    
    return result

# Function to test the model
def unigram_test(data, uniProb):
    totalWord = 0
    for token in data:
        for char in token.text:
            totalWord += 1
            
    probability = 1
    for token in data:
        for char in token.text:
            if char == "a":
                probability *= uniProb[0]
            if char == "b":
                probability *= uniProb[1]
            if char == "c":
                probability *= uniProb[2]
            if char == "d":
                probability *= uniProb[3]
        
        
    return probability
            
probability = unigram(trainSet)
Ouput = unigram_test(testSet,probability)

print("Probability of each type is: ", probability)
print("Probability of the dataset is: ", Ouput)    

# Function to calculate the perplexity of the model    
def perplexity (data, avgLog):
    totalWord = 0
    for token in data:
        for char in token.text:
            totalWord += 1
                
    perplx = avgLog**(-1/totalWord)
    return perplx
            
UnigPerplx = perplexity(testSet, Ouput)
print("Perplexity of the unigram language model is: ", UnigPerplx)
            
            
    