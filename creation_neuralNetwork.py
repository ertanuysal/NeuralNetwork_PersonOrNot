from preprocess import read_data, create_samples, split_data
import numpy as np


# HYPERPARAMETERS
input_size = 50 #size of each word vector
output_size = 2 # our result should be between 1 and zero. if 0.5>result =PERSON 
hidden_layer_size = 2
learning_rate = 0.06
number_of_epochs = 2
path = "./data"  #use relative path like this

"""
Description of Work:
    
In our train.txt dataset,number of samples in class are unbalanced and , because of that reason
i changed dataset size for balancing and easy testing
(If I do not reduce the number of data, the working time can reach 1 hour.)

Because performance issue,i reduced dataset size to 256 data.

Since the weights is taken randomly when the program is first started,
the initial accuracy values can not change much .

Model calculates accuracy value using valid dataset 
in each 90 training data.Since the number 90 is very small, 
no change in accuracy values ​​was observed at the beginning.

(Weights are small different from each 90 samples,
prediction does not change,which affects the accuracy rate.)

Execution time of work is aproximately 5 minutes.
"""
    

def activation_function(layer):
    
    """
    This method returns between value between 0 and 1 to normalize.
    
    """
    
    return 1/(1+np.exp(-layer))
    
	
 

def derivation_of_activation_function(signal):
    
    """
    
    We track backward from output to sigmoid we need this formula.
    
    """
	
    return signal * (1 - signal)


def loss_function(true_labels, probabilities):
    
    """
    The Loss function basically calculates how the model's prediction differs from the ground truth.
    I used binary cross entropy loss function,according to that,
    if prediction result is 0,second part of equaction equal 0.
    On the contrary, the first part of the equation becomes 0.
    
    """
    
    m = true_labels.shape[0]   
    

    
    loss = (1/m) * np.sum(-true_labels*np.log(probabilities) - (1-true_labels)*np.log(1-probabilities)) 
    loss = np.squeeze(loss)   #function uses 

    return loss
            


# the derivation should be with respect to the output neurons
def derivation_of_loss_function(true_labels, probabilities):
    
    """
    Getting derivative from gradiant point.In order to optimize w1 and w2.
    
    """  
    m = true_labels.shape[0]
    #print(true_labels)
    #print(probabilities)
    #print(m)
    
    
    derivative = (1/m) * (-(true_labels/probabilities) + ((1-true_labels)/(1-probabilities)))
    return derivative
    
            
    
def index(filename, word):
    
   
    with open(filename, 'r') as infile:
        lines = [line.split() for line in infile]

   
    for linenumber, line in enumerate(lines, 1):
         
        line=line[0].lower()
        word=word.lower()
  
        if (line==word): #check word and line is equal 
            return linenumber #get linenumber in vocab.txt


def getVector(linenumber):
    
    vector=""
    with open("/Users/ertanuysal/Desktop/HW3 (2)/data/wordVectors.txt") as f:# i should change to relative path  
        for x, line in enumerate(f):
            
            if x == (linenumber-1): #find index in wordVectors list
                vector=line #get line as a word vector.
    return vector

def forward_pass(sample,W1,W2):
    
    
    """ 
    z = W1*x 
    sigmoid(z)
    p=z*w2
    output=sigmoid(p)
    
    
    In this function,Using skip-gram method
    the vector in middle gets as input,  -> [x1,x2,x3]  input x2 -> predict 
    trying to predict right word.
    
    """
    
    
    
    weightX=[]
    
        
    inputlist=[] #store 50 dimensional vector in list of list form
    
    StringVector=sample[0].split() # first vector(x2) on [x1,x2,x3]=x1
    intVector = [ float(x) for x in StringVector ] #convert from string to float 
    inputlist.append(intVector) # put all vector element into a list

    
    temp=[]
    inputlist = np.asarray(inputlist) #convert list to np.array to dot product.
    
    multip=(np.dot(inputlist,W1))  # calculate a1= (x1*w1)
    o1=activation_function(multip[0][0]) 
    o2=activation_function(multip[0][1]) 
    
    sigmoid1=activation_function(o1)
    sigmoid2=activation_function(o2) #sigmoid of hidden layer output
    
    temp.append(sigmoid1)
    temp.append(sigmoid2)
    
    temp= np.asarray(temp)
    
    dotproduct=(np.dot(temp,W2))  # output layer result 
    sigmoid3=activation_function(dotproduct)  #sigmoid of output layer
    
    weightX.append(sigmoid3)
        
    return (weightX),temp
        
           

def embedding_layer(samples):
    
    """
    This function returns word vectors of word.If word is not in "vocab.txt",
    gets fist vector as an input.
    
    """
     
    l=[]
    counter=0
   
    for i in samples:
        temp=[]
        
        index0=index("/Users/ertanuysal/Desktop/HW3 (2)/data/vocab.txt",i[0])
        index1=index("/Users/ertanuysal/Desktop/HW3 (2)/data/vocab.txt",i[1])
        index2=index("/Users/ertanuysal/Desktop/HW3 (2)/data/vocab.txt",i[2])
    
    
        if(index0 ==None):
            index0=1
        if(index1 ==None):
            index1=1
        if(index2 ==None):
            index2=1
        
        a=getVector(index0)
        temp.append(a)
        b=getVector(index1)
        temp.append(b)
        c=getVector(index2)
        temp.append(c)
        counter=counter+1   
        l.append(temp)
        
            
    return l
 

def backward_pass(data,loss_signals,predictions,learning_rate,W1,W2,hidden): #rather than use hidden layer result as a argument i go back from output layer(predictions)
    
    """
    In this function, w1 and  w2 values ​​are optimized according to prediction,loss and learningrate .
    I did not use hidden because i went back grom output result to input.
    
    """
    
    
    
    inputlist=[] #store 50 dimensional vector in list of list form
    
    StringVector=data[0].split() # first vector  on [x1,x2,x3] =x1
    intVector = [ float(x) for x in StringVector ] # convert string data to int array
    inputlist.append(intVector) #put all vector number in a array.
    
    
    
    data = np.asarray(inputlist) #convert list to numpy array in order to make dot product.
    derv=derivation_of_activation_function(predictions) # go back to sigmoid to input layer
       
    """ 
    print("loss_signal",loss_signals)
    print("learning_rate:",learning_rate)
    print("predictions:",predictions)
    
    print("derv:",derv)
    print("w1:",W1)
    #print("data:",data)
    """
    
    p=np.dot(derv,loss_signals.T)  
    k=np.dot(p,data)
   
    W2=W2 - learning_rate * np.dot(p,W2) # calculate new weight 2
    W1=W1 - learning_rate * np.dot(k,W1) # calculate new weight 1
    
    #print("weight1:",W1)
    #print("weight2",W2)
   

    
    return W1,W2


def train(train_data, train_labels, valid_data, valid_labels):
    
    W1 = np.random.randn(input_size, hidden_layer_size)  # i get the random weight in order to optimize it. 
    W2 = np.random.randn(hidden_layer_size, output_size)
    #print(valid_labels) 

    for epoch in range(number_of_epochs):
        index = 0
        for data, labels in zip(train_data, train_labels):
			
            predictions,hidden=forward_pass(data,W1,W2) #forward_pass function returns prediction outpus with  [x,y] format
            
            loss_signals = derivation_of_loss_function(labels, predictions[0]) #we used this in backward part to go back.
            W1,W2=backward_pass(data,loss_signals,predictions[0],learning_rate,W1,W2,hidden)#returns optimized W1 and W2
            loss = loss_function(labels, predictions[0])
            
           
            
            if index%90 == 0: # at each 30th sample, we run validation set to see our model's improvements using new W1 and W2.
                accuracy,loss = test(valid_data, valid_labels,W1,W2)
                print("Epoch= "+str(epoch)+", Coverage= %"+ str(100*(index/len(train_data))) + ", Accuracy= "+ str(accuracy) + ", Loss= " + str(loss))
            index += 1
            
    return W1,W2
            
            


def test(test_data, test_labels,W1,W2):
    
    """
    In this method,According to our weights(w1,w2),
    i tested the our model in each 20 repetitions.
    """
    
    avg_loss = 0
    predictions = []
    labels = []

	#for each batch
    for data, label in zip(test_data, test_labels):
        prediction,temp= forward_pass(data,W1,W2) #predict whether data is person or not
        predictions.append(prediction[0]) # put the result in array
        labels.append(label)
        avg_loss += np.sum(loss_function(label, prediction[0]))

    #turn predictions into one-hot encoded 
    one_hot_predictions = np.zeros(shape=(len(predictions), output_size))
    for i in range(len(predictions)):
        one_hot_predictions[i][np.argmax(predictions[i])] = 1

    predictions = one_hot_predictions
    #print("predictions:",predictions)
    accuracy_score = accuracy(labels, predictions)

    return accuracy_score,  avg_loss / len(test_data)




def accuracy(true_labels, predictions):
	true_pred = 0

	for i in range(len(predictions)):
		if np.argmax(predictions[i]) == np.argmax(true_labels[i]): # if 1 is in same index with ground truth
			true_pred += 1

	return true_pred / len(predictions)

 
if __name__ == "__main__":


	#PROCESS THE DATA
    words, labels = read_data(path)
    sentences = create_samples(words, labels)
    train_x, train_y, test_x, test_y = split_data(sentences)
    
	# creating one-hot vector notation of labels. (Labels are given numeric)
	# [0 1] is PERSON
	# [1 0] is not PERSON
    new_train_y = np.zeros(shape=(len(train_y), output_size))
    new_test_y = np.zeros(shape=(len(test_y), output_size))
    
    for i in range(len(train_y)):
        new_train_y[i][int(train_y[i])] = 1

    for i in range(len(test_y)):
        new_test_y[i][int(test_y[i])] = 1

    train_y = new_train_y
    test_y = new_test_y

    # Training and validation split. (%80-%20)
    valid_x = np.asarray(train_x[int(0.8*len(train_x)):-1])
    valid_y = np.asarray(train_y[int(0.8*len(train_y)):-1])
    train_x = np.asarray(train_x[0:int(0.8*len(train_x))])
    train_y = np.asarray(train_y[0:int(0.8*len(train_y))])
    
    train_x=embedding_layer(train_x)
    valid_x=embedding_layer(train_x)
    test_x=embedding_layer(test_x)
    
    W1,W2=train(train_x, train_y, valid_x, valid_y)
    
    print("Test Scores") 
    print(test(test_x, test_y,W1,W2))

