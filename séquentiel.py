import numpy as np
import matplotlib.pyplot as plt
T = 100

F = np.mat('[1 1 0 0; 0 1 0 0; 0 0 1 1; 0 0 0 1.]')
H = np.mat('[1 0 0 0; 0 0 1 0.]')
F = np.array(F)
H = np.array(H)
sumU = np.mat('[0.333 0.5 0 0; 0.5 1 0 0; 0 0 0.33 0.5; 0 0 0.5 1.]')
sumU = np.array(sumU)
sumW = np.mat('[900 0; 0 900]')
print(sumW)
print(sumU)

x0 = np.array([3,40,-4,20])
x0 = np.transpose(x0)
print(x0)

Xk = np.zeros((100,4))
Xk[0] = x0
print(Xk)

Wk = np.random.multivariate_normal(np.array([0,0]), sumW)

for i in range(1,100): 
    u1,u2,u3,u4 = np.random.multivariate_normal(np.array([0,0,0,0]), sumU)
    Xk[i][0]+=x0[0]+ x0[1]+ u1
    Xk[i][1]+= x0[1] + u2
    Xk[i][2]+=x0[2]+ x0[3]+ u3
    Xk[i][3]+=x0[3] + u4
    x0 = Xk[i]
    i+=1
    
Yk =np.zeros((100,2))

for i in range(1,100):
    w1,w2 = np.random.multivariate_normal(np.array([0,0]), sumW)
    Yk[i][0]+= Xk[i][0] + w1
    Yk[i][1]+= Xk[i][2] + w2
    

print(Xk)
plt.plot(Xk[:,0],Xk[:,2])
plt.plot(Yk[:,0],Yk[:,1])
plt.show()


x00= x0
P00=np.identity(4)
print(P00)

#Prediction step

Xpred = np.zeros((100,4))
Xpred[0]=x0
Xest = np.zeros((100,4))
Ppred = P00
KKalman = np.dot(P00, H.transpose)
inverse = np.dot(H,P00)
inverse = np.dot(inverse,H.transpose)
inverse+= sumW
inverse = np.linalg.inv(inverse)
Kkalman = np.dot(KKalman,inverse)

for i in range(1,100):
    Xpred[i]= np.dot(F,Xk[i-1])
    Ppred = np.dot(F,Ppred)
    Pred = np.dot(Ppred,F.transpose) + sumU
    
    Kkalman = np.dot(Ppred,H)
    inv = np.dot(H, Ppred)
    inv = np.dot(inv, H.transpose)
    inv += sumW
    inv = np.linalg.inv(inv)
    Kkalman = np.dot(Kkalman,inv)
    
    Xest[i] = Xpred[i] + Kkalman*(Yk[i]- np.dot(H, Xpred[i]))
    

print(Xpred)



    