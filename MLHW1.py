import numpy as np
import pandas as pd
crx = pd.read_csv('crx.csv')
crx.describe(include='all')
dropindex = crx[(crx.att1=="?") |(crx.att2=="?") | (crx.att14=="?")| (crx.att6=="?")| (crx.att7=="?")].index
crx.drop(dropindex , inplace=True)
crx=crx.reset_index(drop=True)
twosides=["att1","att9","att10","att12"]
for i in twosides:
    a,b=np.unique(crx[i], return_counts=True)
    crx[i].replace(a,[1, -1], inplace=True)
crxnum = pd.get_dummies(crx.iloc[:,0:-1], columns = ['att4','att5','att6','att7','att13'])
crxnum = crxnum.astype(float)
crxnum.describe(include='all')
corri=abs(crxnum.corr())

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
remove_correlated_features(crxnum)

def labelmap(x):
    x = 1 if x == "+" else -1
    return x
label = np.array(crx.label.map(labelmap))
corri=abs((pd.concat([crxnum, pd.DataFrame(label)], axis=1)).corr())

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
remove_less_significant_features(crxnum,label)

num_variables=crxnum.shape[1]
variables=np.array(crxnum.iloc[:,0:num_variables].astype(float))
columnmax = variables.max(axis=0)
columnmin = variables.min(axis=0)
variables = ((variables-columnmin)/(columnmax-columnmin))*2-1

num_samples=crxnum.shape[0]
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
strengtharray=[0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4]
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
