import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

loc = (r"C:\Users\Srinidh Vudityala\Documents\housepricedata.csv")
lr = 0.0005
itr = 100000
total_loss = 0
hidden_neurons_1 = 100
hidden_neurons_2 = 50
hidden_neurons_3 = 30
hidden_neurons_4 = 20
av_losst =[]
t_losst=[]
house = np.zeros((1,10))

n_crct = 0
n_total = 0
tot_predt = 0
accuracyt =[]
av_predt=[]

data = pd.read_csv(loc, nrows=1460)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

#creating 1st neural network with 4 tanh hidden layers

#creating a matrix of random weights
wt1 = np.random.randn(hidden_neurons_1,10)
bt1 = np.random.randn(1,hidden_neurons_1)

wt2 = np.random.randn(hidden_neurons_2,hidden_neurons_1)
bt2 = np.random.randn(1,hidden_neurons_2)

wt3 = np.random.randn(hidden_neurons_3,hidden_neurons_2)
bt3 = np.random.randn(1,hidden_neurons_3)

wt4 = np.random.randn(hidden_neurons_4,hidden_neurons_3)
bt4 = np.random.randn(1,hidden_neurons_4)

wt5 = np.random.randn(1,hidden_neurons_4)
bt5 = np.random.randn()

zt1 = np.zeros((1 , hidden_neurons_1 ))
zt2 = np.zeros((1 , hidden_neurons_2 ))
zt3 = np.zeros((1 , hidden_neurons_3 ))
zt4 = np.zeros((1 , hidden_neurons_4 ))

for i in range(itr):
    ri = np.random.randint(1160)
    house1 = data.loc[ri]
    house1 = np.append([1] , house1)
    house = np.delete(house1 ,11)
    house = np.delete(house , 0)
    house1= np.delete(house1 , 0)
    
    house = sigmoid(house)

    at1 = np.dot(house,wt1.transpose()) + bt1
    zt1 = np.tanh(at1)
       
    at2 = np.dot(zt1,wt2.transpose()) + bt2
    zt2 = np.tanh(at2).reshape(1,hidden_neurons_2)
    
    at3 = np.dot(zt2,wt3.transpose()) + bt3
    zt3 = np.tanh(at3).reshape(1,hidden_neurons_3)

    at4 = np.dot(zt3,wt4.transpose()) + bt4
    zt4 = np.tanh(at4)

    ht = np.dot(zt4,wt5.transpose()) + bt5

    predt = sigmoid(ht)
    target = house1[10]
    losst = -1 * target * np.log(predt) + (1 - target) * np.log(1 - predt)
    
    if predt > 0.5:
        pred1 = 1
    else :
        pred1 = 0
    if pred1 == target:
        n_crct += 1
    acc = n_crct / (i+1)
    accuracyt.append(acc)

    tot_predt += pred1
    avg_predt = tot_predt/(i+1)
    av_predt.append(avg_predt)

    total_loss += losst
    avg_loss = total_loss / (i + 1)   
    if i % 5000 == 0:
        print(avg_loss,i)
    av_losst.append(avg_loss[0])
    t_losst.append(total_loss[0])

    dl_dpred = -1 *((target/predt) - (1-target)/(1-predt))
    dpred_dh = predt * (1 - predt)
    dl_dh = dl_dpred * dpred_dh

    dl_dz4 = dl_dh * wt5  #1*hn3
    dz4_da4 = 1 - np.square(zt4) #1*hn3
    da4_dw4 = zt3
    da4_dz3 = wt4

    dl_da4 = dl_dz4 * dz4_da4
    dl_dz3 = np.dot(dl_da4,wt4)
    dz3_da3 = 1 - np.square(zt3)

    dl_da3 = dl_dz3 * dz3_da3
    dl_dz2 = np.dot(dl_da3,wt3)
    dz2_da2 = 1 - np.square(zt2)
    
    dl_da2 = dz2_da2 * dl_dz2 # 1*hn2  
    dl_dz1 = np.dot(dl_da2,wt2)   #hn1*1
    dz1_da1 = 1 - np.square(zt1) #1*hn1
    da1_dw1 = house   #1x10

    dl_da1 = dz1_da1 * (dl_dz1.reshape(1,hidden_neurons_1))   #1 x hn1
    
    wt5 -= lr * dl_dh * zt4
    bt5 -= lr * dl_dh * 1
    wt4 -= lr * np.dot(dl_da4.transpose(),zt3.reshape(1,hidden_neurons_3))
    bt4 -= lr * dl_da4.reshape(1 , hidden_neurons_4)
    wt3 -= lr * np.dot(dl_da3.transpose(),zt2.reshape(1,hidden_neurons_2))
    bt3 -= lr * dl_da3.reshape(1 , hidden_neurons_3)
    wt2 -= lr * np.dot(dl_da2.transpose(),zt1.reshape(1,hidden_neurons_1))  #hn2*hn1
    bt2 -= lr * dl_da2.reshape(1, hidden_neurons_2)
    wt1 -= lr * np.dot(dl_da1.reshape(hidden_neurons_1,1),house.reshape(1,10)) #hn1*10
    bt1 -= lr * dl_da1
    
print("accuracy of tanh" ,acc)
plt.plot(av_predt)
plt.show()
plt.plot(accuracyt)
plt.show()
plt.plot(av_losst)
plt.show()

#creating 2nd network with 2 sigmoid hidden layers
n_crct = 0
av_loss =[]
t_loss=[]
tot_pred = 0
accuracy =[]
av_pred=[]

#creating a matrix of random weights
w1 = np.random.randn(hidden_neurons_1,10)
b1 = np.random.randn(1,hidden_neurons_1)

w2 = np.random.randn(hidden_neurons_2,hidden_neurons_1)
b2 = np.random.randn(1,hidden_neurons_2)

w3 = np.random.randn(1,hidden_neurons_2)
b3 = np.random.randn()

z1 = np.zeros((1 , hidden_neurons_1 ))
z2 = np.zeros((1 , hidden_neurons_2 ))

for i in range(100000):
    ri = np.random.randint(1160)
    house1 = data.loc[ri]
    house1 = np.append([1] , house1)
    house = np.delete(house1 ,11)
    house = np.delete(house , 0)
    house1= np.delete(house1 , 0)
    
    house = sigmoid(house)

    a1 = np.dot(house,w1.transpose()) + b1
    z1 = sigmoid(a1)
    
    a2 = np.dot(z1,w2.transpose()) + b2
    z2 = sigmoid(a2).reshape(1,hidden_neurons_2)
    
    h = np.dot(z2,w3.transpose()) + b3

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

    tot_pred += pred1
    avg_pred = tot_pred/(i+1)
    av_pred.append(avg_pred)

    total_loss += loss
    avg_loss = total_loss / (i + 1)   
    if i % 5000 == 0:
        print(avg_loss,i)
    av_loss.append(avg_loss[0])

    dl_dpred = -1 *((target/pred) - (1-target)/(1-pred))
    dpred_dh = pred * (1 - pred)
    dl_dh = dl_dpred * dpred_dh

    dl_dz2 = dl_dh * w3  #1*hn2
    dz2_da2 = np.dot(z2 ,(1 - z2).reshape(hidden_neurons_2,1)) #1*1
    da2_dw2 = z1
    da2_dz1 = w2

    dl_da2 = dz2_da2 * dl_dz2 # 1*hn2  
    dl_dz1 = np.dot(dl_da2,w2)   #hn1*1
    dz1_da1 = np.dot(z1,(1 - z1).transpose()) #1*1
    da1_dw1 = house   #1x10

    dl_da1 = dz1_da1 * (dl_dz1.reshape(1,hidden_neurons_1))   #1 x hn1
    
    w3 -= lr * dl_dh * z2
    b3 -= lr * dl_dh * 1
    w2 -= lr * np.dot(dl_da2.transpose(),z1.reshape(1,hidden_neurons_1))  #hn2*hn1
    b2 -= lr * dl_da2.reshape(1, hidden_neurons_2)
    w1 -= lr * np.dot(dl_da1.reshape(hidden_neurons_1,1),house.reshape(1,10)) #hn1*10
    b1 -= lr * dl_da1

print("acc of sig",acc)
plt.plot(av_pred)
plt.show()
plt.plot(accuracy)
plt.show()
plt.plot(av_loss)
plt.show()

#creating 3rd neural network with 1 tanh hidden layer
n_crct = 0
av_lossk =[]
t_lossk =0
tot_predk = 0
accuracyk =[]
av_predk =[]
#creating a matrix of random weights
w1k = np.random.randn(hidden_neurons_1,10)
b1k = np.random.randn(1,hidden_neurons_1)

w2k = np.random.randn(1,hidden_neurons_1)
b2k = np.random.randn()

z1k = np.zeros((1 , hidden_neurons_1 ))

for i in range(itr):
    ri = np.random.randint(1160)
    house1 = data.loc[ri]
    house1 = np.append([1] , house1)
    house = np.delete(house1 ,11)
    house = np.delete(house , 0)
    house1= np.delete(house1 , 0)   
    house = sigmoid(house)

    a1k = np.dot(house,w1k.transpose()) + b1k
    z1k = np.tanh(a1k)
    hk = np.dot(z1k,w2k.transpose()) + b2k

    predk = sigmoid(hk)
    target = house1[10]
    lossk = -1 * target * np.log(predk) + (1 - target) * np.log(1 - predk)
    
    if predk > 0.5:
        pred1 = 1
    else :
        pred1 = 0
    if pred1 == target:
        n_crct += 1
    acc = n_crct / (i+1)
    accuracyk.append(acc)

    tot_predk += pred1
    avg_predk = tot_pred/(i+1)
    av_pred.append(avg_predk)

    t_lossk += lossk
    avg_lossk = t_lossk / (i + 1)   
    if i % 5000 == 0:
        print(avg_lossk,i)
    av_lossk.append(avg_lossk)

    dl_dpred = -1 *((target/predk) - (1-target)/(1-predk))
    dpred_dh = predk * (1 - predk)
    dl_dh = dl_dpred * dpred_dh

    dl_dz1 = dl_dh * w2k  #1*hn1     
    dz1_da1 = 1 - np.square(z1k) #1*hn1
    da1_dw1 = house   #1x10

    dl_da1 = dz1_da1 * (dl_dz1.reshape(1,hidden_neurons_1))   #1 x hn1
    
    w2k -= lr * dl_dh * z1k
    b2k -= lr * dl_dh
    w1k -= lr * np.dot(dl_da1.reshape(hidden_neurons_1,1),house.reshape(1,10)) #hn1*10
    b1k -= lr * dl_da1

print("acc of 2nd tanh ", acc)
#plt.plot(av_predk)
#plt.show()
plt.plot(accuracyk)
plt.show()
#plt.plot(av_lossk)
#plt.show()

#testing the model

tp = 0
fp = 0
fn = 0
pred = 0
n_crct = 0
n_total = 0
tot_pred = 0
accuracy =[]
av_pred=[]
for i in range(1160,1460):
    n_total +=1
    house1 = data.loc[i]
    house1 = np.append([1] , house1)
    house = np.delete(house1 ,11)
    house = np.delete(house , 0)
    house1= np.delete(house1 , 0)
    house = sigmoid(house)

    at1 = np.dot(house,wt1.transpose()) + bt1
    zt1 = np.tanh(at1)    
    at2 = np.dot(zt1,wt2.transpose()) + bt2
    zt2 = np.tanh(at2).reshape(1,hidden_neurons_2)   
    at3 = np.dot(zt2,wt3.transpose()) + bt3
    zt3 = np.tanh(at3).reshape(1,hidden_neurons_3)
    at4 = np.dot(zt3,wt4.transpose()) + bt4
    zt4 = np.tanh(at4)
    ht = np.dot(zt4,wt5.transpose()) + bt5
    predt = sigmoid(ht)
    if predt > 0.5:
        predt1 = 1
    else:
        predt1 = 0


    a1 = np.dot(house,w1.transpose()) + b1
    z1 = sigmoid(a1)   
    a2 = np.dot(z1,w2.transpose()) + b2
    z2 = sigmoid(a2).reshape(1,hidden_neurons_2)    
    h = np.dot(z2,w3.transpose()) + b3
    pred = sigmoid(h)
    if pred > 0.5:
        pred1 = 1
    else:
        pred1 = 0

    a1k = np.dot(house,w1k.transpose()) + b1k
    z1k = np.tanh(a1k)
    hk = np.dot(z1k,w2k.transpose()) + b2k
    predk = sigmoid(hk)
    if predk > 0.5:
        predk1 = 1
    else:
        predk1 = 0 

    final_pred1 = predt1 + pred1 + predk1 
    if final_pred1 > 1.5:
        final_pred = 1
    else:
        final_pred = 0

    target = house1[10]
    if final_pred == target:
        n_crct +=1

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
plt.show()
print("final acc",acc)