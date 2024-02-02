import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

loc = (r"C:\Users\Srinidh Vudityala\Documents\housepricedata.csv")
lr = 0.0005
itr = 200000
total_loss = 0
hidden_neurons_1 = 100
hidden_neurons_2 = 50
hidden_neurons_3 = 20
hidden_neurons_4 = 10
av_loss =[]
t_loss=[]
house = np.zeros((1,10))

tp = 0
fp = 0
fn = 0
n_crct = 0
n_total = 0
tot_pred = 0
accuracy =[]
av_pred=[]

data = pd.read_csv(loc, nrows=1460)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

#creating a matrix of random weights
w1 = np.random.randn(hidden_neurons_1,10)
b1 = np.random.randn(1,hidden_neurons_1)

w2 = np.random.randn(hidden_neurons_2,hidden_neurons_1)
b2 = np.random.randn(1,hidden_neurons_2)

w3 = np.random.randn(hidden_neurons_3,hidden_neurons_2)
b3 = np.random.randn(1,hidden_neurons_3)

w4 = np.random.randn(hidden_neurons_4,hidden_neurons_3)
b4 = np.random.randn(1,hidden_neurons_4)

w5 = np.random.randn(1,hidden_neurons_4)
b5 = np.random.randn()

z1 = np.zeros((1 , hidden_neurons_1 ))
z2 = np.zeros((1 , hidden_neurons_2 ))
z3 = np.zeros((1 , hidden_neurons_3 ))
z4 = np.zeros((1 , hidden_neurons_4 ))

#the training loop
for i in range(itr):
    ri = np.random.randint(1160)
    house1 = data.loc[ri]
    house1 = np.append([1] , house1)
    house = np.delete(house1 , 11)
    house = np.delete(house , 0)
    house1= np.delete(house1 , 0)
    
    house = sigmoid(house)

    a1 = np.dot(house,w1.transpose()) + b1
    z1 = np.tanh(a1)
    
    a2 = np.dot(z1,w2.transpose()) + b2
    z2 = np.tanh(a2).reshape(1,hidden_neurons_2)
    
    a3 = np.dot(z2,w3.transpose()) + b3
    z3 = np.tanh(a3).reshape(1,hidden_neurons_3)

    a4 = np.dot(z3,w4.transpose()) + b4
    z4 = np.tanh(a4)

    h = np.dot(z4,w5.transpose()) + b5

    pred = sigmoid(h)
    target = house1[10]
    loss = -1 * target * np.log(pred) + (1 - target) * np.log(1 - pred)
    
    if pred > 0.5:
        pred1 = 1
    else :
        pred1 = 0
    if pred1 == target:
        n_crct += 1
    acc = n_crct / (i+1)
    accuracy.append(acc)

    if pred1 == 1:
        if target == 1:
            tp += 1
        else:
            fp += 1
    else:
        if target == 1:
            fn += 1

    tot_pred += pred1
    avg_pred = tot_pred/(i+1)
    av_pred.append(avg_pred)

    total_loss += loss
    avg_loss = total_loss / (i + 1)   
    if i % 5000 == 0:
        print(avg_loss,i)
    av_loss.append(avg_loss[0])
    t_loss.append(total_loss[0])

    #back-propagation
    dl_dpred = -1 *((target/pred) - (1-target)/(1-pred))
    dpred_dh = pred * (1 - pred)
    dl_dh = dl_dpred * dpred_dh

    dl_dz4 = dl_dh * w5  #1*hn3
    dz4_da4 = 1 - np.square(z4) #1*hn3
    da4_dw4 = z3
    da4_dz3 = w4

    dl_da4 = dl_dz4 * dz4_da4
    dl_dz3 = np.dot(dl_da4,w4)
    dz3_da3 = 1 - np.square(z3)

    dl_da3 = dl_dz3 * dz3_da3
    dl_dz2 = np.dot(dl_da3,w3)
    dz2_da2 = 1 - np.square(z2)
    
    dl_da2 = dz2_da2 * dl_dz2 # 1*hn2  
    dl_dz1 = np.dot(dl_da2,w2)   #hn1*1
    dz1_da1 = 1 - np.square(z1) #1*hn1
    da1_dw1 = house   #1x10

    dl_da1 = dz1_da1 * (dl_dz1.reshape(1,hidden_neurons_1))   #1 x hn1
    
    w5 -= lr * dl_dh * z4
    b5 -= lr * dl_dh * 1
    w4 -= lr * np.dot(dl_da4.transpose(),z3.reshape(1,hidden_neurons_3))
    b4 -= lr * dl_da4.reshape(1 , hidden_neurons_4)
    w3 -= lr * np.dot(dl_da3.transpose(),z2.reshape(1,hidden_neurons_2))
    b3 -= lr * dl_da3.reshape(1 , hidden_neurons_3)
    w2 -= lr * np.dot(dl_da2.transpose(),z1.reshape(1,hidden_neurons_1))  #hn2*hn1
    b2 -= lr * dl_da2.reshape(1, hidden_neurons_2)
    w1 -= lr * np.dot(dl_da1.reshape(hidden_neurons_1,1),house.reshape(1,10)) #hn1*10
    b1 -= lr * dl_da1

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = (2 * tp) / (2 * tp + fp + fn)

print("train f1 score =", f1)
print("training acc",acc)
plt.plot(av_pred)
plt.title('Average Prediction')
plt.show()
plt.plot(accuracy)
plt.title('Training accuracy of the model over time')
plt.show()

plt.plot(av_loss)
plt.title('Average loss over time')
plt.show()

tp = 0
fp = 0
fn = 0
pred = 0
n_crct = 0
n_total = 0
tot_pred = 0
accuracy =[]
av_pred=[]

#the testing loop
for i in range(1160,1460):
    n_total +=1
    house1 = data.loc[i]
    house1 = np.append([1] , house1)
    house = np.delete(house1 ,11)
    house = np.delete(house , 0)
    house1= np.delete(house1 , 0)
    house = sigmoid(house)

    a1 = np.dot(house,w1.transpose()) + b1
    z1 = np.tanh(a1)
    
    a2 = np.dot(z1,w2.transpose()) + b2
    z2 = np.tanh(a2).reshape(1,hidden_neurons_2)
    
    a3 = np.dot(z2,w3.transpose()) + b3
    z3 = np.tanh(a3).reshape(1,hidden_neurons_3)

    a4 = np.dot(z3,w4.transpose()) + b4
    z4 = np.tanh(a4)

    h = np.dot(z4,w5.transpose()) + b5

    pred = sigmoid(h)
    
    target = house1[10]
    if pred > 0.5:
        pred1 = 1
    else :
        pred1 = 0
    if pred1 == target:
        n_crct += 1

    if pred1 == 1:
        if target == 1:
            tp += 1
        else:
            fp += 1
    else:
        if target == 1:
            fn += 1
    
    acc = n_crct / n_total
    accuracy.append(acc)

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = (2 * tp) / (2 * tp + fp + fn)

print("test f1 score =", f1)
plt.plot(accuracy)
plt.title('Test Accuracy over time')
plt.show()
print("final acc",acc)