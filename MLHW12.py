import numpy as np
import pandas as pd
data = pd.read_csv('data.csv')
data.describe(include='all')
datanum=data.iloc[:,2:]
corri=abs(datanum.corr())

def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = abs(X.corr())
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped
remove_correlated_features(datanum)

def labelmap(x):
    x = 1 if x == "M" else -1
    return x
label = np.array(data.Diagnosis.map(labelmap))
corri=abs((pd.concat([datanum, pd.DataFrame(label)], axis=1)).corr())

def remove_less_significant_features(X,Y):
    sl = 0.05
    corr=abs((pd.concat([X, pd.DataFrame(Y)], axis=1)).corr())
    drop_columns = np.full(corr.shape[0]-1, False, dtype=bool)
    for j in range(0, corr.shape[0]-1):
        if corr.iloc[-1, j] <= sl:
            drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped
remove_less_significant_features(datanum,label)

num_variables=datanum.shape[1]
variables=np.array(datanum.iloc[:,0:num_variables].astype(float))
columnmax = variables.max(axis=0)
columnmin = variables.min(axis=0)
variables = ((variables-columnmin)/(columnmax-columnmin))*2-1

num_samples=datanum.shape[0]
weight=np.zeros(num_variables+1)
variables2=np.concatenate((np.ones((num_samples,1)), variables),axis=1)
check=pd.DataFrame(variables2).describe(include='all')

def predict(variable):
    act = np.dot(weight , variable)
    return np.sign(act)

def precision(predictionarray, label):
    TP = sum(np.logical_and(predictionarray == 1, label == 1))
    FP = sum(np.logical_and(predictionarray == 1, label == -1))
    return TP / (TP + FP)

def recall(predictionarray, label):
    TP = sum(np.logical_and(predictionarray == 1, label == 1))
    FN = sum(np.logical_and(predictionarray == -1, label == 1))
    return TP / (TP + FN)

def F1score(precision, recall):
    return 2 * precision * recall / (precision + recall)

def Performance(variables,label):
    predictionarray=np.array(list(map(predict, variables)))
    pre=precision(predictionarray, label)
    rec=recall(predictionarray, label)
    F1s=F1score(pre,rec)
    acc=np.mean(predictionarray == label)
    print(acc,pre,rec,F1s)

maxacc=0
stopcount=0
bestweight =0
weight=np.zeros(num_variables+1)
for c in range(1,10000):
    for i in range(0,num_samples):
        if label[i] != predict(variables2[i,:]):
            weight = weight + variables2[i,:]*label[i]
    if np.mean(list(map(predict, variables2)) == label) > maxacc:
        maxacc = np.mean(list(map(predict,variables2)) == label)
        bestweight = weight
        stopcount = 0
    else:
        stopcount +=1
    if stopcount == 100:
        break
weight = bestweight
print("linear gradient descent")
print(weight)
print(sum(np.square(weight)))
Performance(variables2,label)

weight = np.linalg.inv(variables2.T @ variables2) @ (variables2.T @ label)
print("linear matrix formula")
print(weight)
print(sum(np.square(weight)))
Performance(variables2,label)


'''
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(variables, label)
w_0 = regressor.intercept_
w_1 = regressor.coef_
print('Interception : ', w_0)
print('Coeficient : ', w_1)
def predict3c(variable):
    act = np.dot(variable,w_1)+w_0
    return np.sign(act)
np.mean(list(map(predict3c, variables)) == label)


'''

'''
maxacc=0
stopcount=0
bestweight=0
rate=0.000001
weight=np.zeros(num_variables+1)
for c in range(0,1000000):
    loss = label - variables2.dot(weight)
    grad = (variables2.T).dot(loss)
    weight = weight + rate*grad
    if np.mean(list(map(predict, variables2)) == label) > maxacc:
        maxacc = np.mean(list(map(predict, variables2)) == label)
        bestweight = weight
        stopcount = 0
    else:
        stopcount += 1
    if stopcount == 10000:
        break
    #print(loss)
    #print(grad)
    #print(beta)
    #print(sum(beta))
weight=bestweight
print(weight)
Performance(variables2,label)
'''

k=0
c = np.zeros((1,))
v = np.zeros((1,num_variables+1))
maxacc=0
weight=np.zeros(num_variables+1)
for t in range(0,100):
    for i in range(0,num_samples):
        if label[i] == np.sign(np.dot(v[k],variables2[i,:])):
            c[k]=c[k]+1
        else:
            v=np.vstack([v, v[k]+variables2[i,:]*label[i]])
            c=np.vstack([c,1])
            k=k+1

sumofweight= np.zeros(num_samples)
for l in range(1,len(v)):
    weight=v[l]
    sumofweight+=np.array(list(map(predict, variables2)))*c[l]
predictionarrayvp=np.sign(sumofweight/sum(c))
print("voted perceptron")
def Performance2(predictionarray,label):
    pre=precision(predictionarray, label)
    rec=recall(predictionarray, label)
    F1s=F1score(pre,rec)
    acc=np.mean(predictionarray == label)
    print(acc,pre,rec,F1s)
Performance2(predictionarrayvp,label)


'''
maxacc=0
stopcount=0
bestbeta=0
rate=0.000001
beta=np.zeros(num_variables+1)
count=0
lambda_=0.0001
for c in range(0,1000000):
    prediction = np.sign(variables2.dot(beta))
    ifwrong = (prediction!=label)
    beta = beta - rate * (2*lambda_*beta- np.dot(variables2.T,label)*np.mean(ifwrong))
    if np.mean(list(map(predict3, variables2)) == label) > maxacc:
        maxacc = np.mean(list(map(predict3, variables2)) == label)
        bestbeta = beta
        stopcount = 0
    else:
        stopcount += 1
    if stopcount == 2000:
        break
    count+=1
    #print(loss)
    #print(grad)
    print(beta)
    print(sum(beta))
print(maxacc)
print(bestbeta)
print(count)
print(beta)



X=variables
y=label
lr=0.001
lambda_param=0.03
n_iters=1000

n_samples, n_features = X.shape

w = np.ones(n_features)
b = 0

for _ in range(n_iters):
    for idx, x_i in enumerate(X):
        condition = y[idx] * (np.dot(x_i, w) - b) >= 1
        if condition:
            w -= lr * (2 * lambda_param * w)
        else:
            w -= lr * (
                2 * lambda_param * w - np.dot(x_i, y[idx])
            )
            b -= lr * y[idx]

def predict(X):
    approx = np.dot(X, w) - b
    return np.sign(approx)
np.mean(predict(X)==label)

'''


'''
regularization_strength = 00000
learning_rate = 0.000001
max_epochs = 5000
weight = np.zeros(num_variables+1)
stopcount = 0
bestcost = float("inf")

for epoch in range(1, max_epochs):
    for i in range(0,variables2.shape[0]):
        grad = calculate_cost_gradient(weight, variables2[i,:], label[i])
        weight = weight - (learning_rate * grad)
        cost = compute_cost(weight, variables2, label)
        if bestcost > cost:
            bestcost = cost
            stopcount =0
        else:
            stopcount+=1
        if stopcount == 1000:
            break
print("minimize ||W||^2")
print(weight)
Performance(variables2,label)
'''

maxacc=0
stopcount=0
bestweight =0
weight=np.zeros(num_variables+1)
for c in range(1,10000):
    for i in range(0,num_samples):
        if label[i] != predict(variables2[i,:]):
            weight = weight + variables2[i,:]*label[i] + (-2 *(0.1/c)* weight)
        else:
            weight = weight + (-2 * (0.1 / c) * weight)
    if np.mean(list(map(predict, variables2)) == label) > maxacc:
        maxacc = np.mean(list(map(predict,variables2)) == label)
        bestweight = weight
        stopcount = 0
    else:
        stopcount +=1
    if stopcount == 100:
        break
weight = bestweight
print("linear gradient descent with minimize ||W||^2")
print(weight)
print(sum(np.square(weight)))
Performance(variables2,label)

def compute_cost(W, X, Y, regularization_strength):
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0
    hinge_loss = regularization_strength * np.sum(distances)
    cost = np.dot(W, W) + hinge_loss
    return cost

def calculate_cost_gradient(W, X, Y, regularization_strength):
    Y= np.array([Y])
    X= np.array([X])
    distance = 1 - (Y * np.dot(X, W))
    dw = np.zeros(len(W))
    if max(0, distance) == 0:
        di = W
    else:
        di = W - (regularization_strength * Y * X)
    return di.reshape(-1)

def Performance(variables,label):
    predictionarray=np.array(list(map(predict, variables)))
    pre=precision(predictionarray, label)
    rec=recall(predictionarray, label)
    F1s=F1score(pre,rec)
    acc=np.mean(predictionarray == label)
    print(acc,pre,rec,F1s)
    return F1s

learning_rate = 0.00001
weight = np.zeros(num_variables+1)
stopcount = 0
bestcost = float("inf")
count=0
F1sarray=[]
weight2array=[]
strengtharray=[200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
for regularization_strength in strengtharray:
    for c in range(1, 1000):
        for i in range(0,num_samples):
            grad = calculate_cost_gradient(weight, variables2[i,:], label[i], regularization_strength)
            weight = weight - (learning_rate * grad)
            cost = compute_cost(weight, variables2, label, regularization_strength)
            count+=1
        if bestcost > cost:
            bestcost = cost
            stopcount =0
        else:
            stopcount+=1
        if stopcount == 100:
            break

    print("minimize ||W||^2 with slack")
    print(regularization_strength)
    print(weight)
    print(sum(np.square(weight)))
    weight2array=np.append(weight2array,sum(np.square(weight)))
    F1sarray=np.append(F1sarray,Performance(variables2,label))

import matplotlib.pyplot as plt
fig,ax = plt.subplots()
ax.plot(strengtharray, F1sarray, color="red", marker="o")
ax.set_xlabel("Regularization Strength", fontsize = 14)
ax.set_ylabel("F1s", color="red", fontsize=14)
ax2=ax.twinx()
ax2.plot(strengtharray, weight2array,color="blue",marker="o")
ax2.set_ylabel("||W||^2",color="blue",fontsize=14)
plt.title("Fls & ||W||^2 vs Regularization Strength")
plt.show()

from sklearn.svm import SVC
svm=SVC(kernel="linear",probability=True)
svm.fit(variables,label)
print('w = ',svm.coef_)
print('b = ',svm.intercept_)
weight=np.append(svm.intercept_,svm.coef_)
print(sum(np.square(weight)))
Performance(variables2,label)