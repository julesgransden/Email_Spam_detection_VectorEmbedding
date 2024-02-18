import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import sklearn.svm 
import sklearn.neural_network 
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt


df = pd.read_csv("spam.csv")

client = OpenAI(api_key="API_KEY")

# Create an empty DataFrame
em = pd.DataFrame(columns=['labels'] +[f'Feature{i+1}' for i in range(1536)])

l=0
#lets rearange df with vectorized messages
for obj in df.index:
    #use API to generate vectorization of the message
    response = client.embeddings.create(
        input=df['Message'][obj],
        model="text-embedding-ada-002"
    )
    # Assuming response.data[0].embedding is the correct path to access the embeddings
    embeddings = response.data[0].embedding
    
    
    if df["Category"][obj] == "ham": l = 1 
    else: l= -1
        
    embeddings.insert(0,l)
    em.loc[obj] =  embeddings
    #insert the embeddings in df from row 0->
    em.loc[obj] =embeddings

#Save new file of embbeded vectors 
em.to_csv("vectorized_Data.csv")

#split into training, validation and test data ---> 75-15-10
train = em[:int(0.75*em.shape[0])]
vs = em[int(0.75*em.shape[0]):int(0.9*em.shape[0])] 
test = em[int(0.75*em.shape[0]):] 

X_train = train.iloc[:,1:]
Y_train = train.iloc[:,0]
X_vs = vs.iloc[:,1:]
Y_vs = vs.iloc[:,0]
X_test = test.iloc[:,1:]
Y_test = test.iloc[:,0]


##                                                             DECISION TREE
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, Y_train)

#Test set
y_pred = clf.predict(X_test)
print("decision tree Accuracy:", accuracy_score(Y_test, y_pred))

#show decision tree
fig = plt.figure(figsize=(15,10))
_ = tree.plot_tree(clf, 
                   filled=True)


##                                                             LOGISTIC REGRESSION
clf = LogisticRegression(C=1.0, solver='lbfgs', multi_class='ovr')
# Train the model on the training set
clf.fit(X_train, Y_train)
# Evaluate the model on the test set
score = clf.score(X_test, Y_test)
print("Logistic regression Accuracy:", score)

##                                                             ARTIFICIAL NEURAL NETWORK
# Create an instance of the MLPClassifier class
neural_network = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu')
 # Fit the model to the training data 
neural_network.fit(X_train, Y_train) 
# Predict the labels of new data 
y_pred = neural_network.predict(X_test)
print("ANN Accuracy:", accuracy_score(Y_test, y_pred))


##                                                            RANDOM FORREST
clf = RandomForestClassifier()
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print('Random forrest Accuracy:', accuracy)

##                                                            SUPPORT VECTOR MACHINES
# Create an instance of the SVM class 
svm = sklearn.svm.SVC(kernel= 'linear', C=1.0) 
# Fit the model to the training data 
svm.fit(X_train, Y_train) 
# Predict the labels of new data 
y_pred = svm.predict(X_test)
print("Support Vector Machines Accuracy:", accuracy_score(Y_test, y_pred))