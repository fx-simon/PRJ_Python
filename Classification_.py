# 1. Regression 2. Classification 3.clustering ( UNSUPERVISED LEARNING )
# Logistic Regression  / classification
# W = 1.15221034
# b = -14.85416169

# 
import numpy as np

x_data = np.array([2,4,6,8,10,12,14,16,18,20]).reshape(10,1)
t_data = np.array([0,0,0,0,0,0,1,1,1,1]).reshape(10,1)
# loaded_data = np.loadtxt('./data/sample.csv', delimiter=',', dytpe=np.float32)
# x_data = loaded_data[:,0:-1]
# t_data = loaded_data[:,[-1]]

W = np.random.rand(1,1)
b = np.random.rand(1)

learning_rate = 1e-3

#------------------------------------ [ A ]
#  Lambda 함수는 손실함수(loos_func)를 재호출
#  List Comprehension
#------------------------------------

f = lambda x : loss_func(x_data, t_data)
# f(x)=loss_func(x_data, t_data)

#------------------------------------ [ A ]
#  예측 실행
# f : 시그모이드 함수
#------------------------------------ #1
def predict(x):
    z = np.dot(x,W)+b
    y = sigmoid(z)

    if y > 0.5:
        result= 1
    else:
        result= 0

    return y, result

#------------------------------------ [ B ]
#  Sigmoid
# f : 시그모이드 함수
#------------------------------------ #1-
def sigmoid(x):
     return 1/(1+np.exp(-x))

#------------------------------------ [ C ]
#  손실함수
# f : 시그모이드 함수
#------------------------------------
def error_val(x,t):
     delta = 1e-7

     z = np.dot(x,W)+b
     y = sigmoid(z)
     return -np.sum(t*np.log(y+delta) + (1-t)*np.log(1-y)+delta)

def loss_func(x,t):
     delta = 1e-7

     z = np.dot(x,W)+b
     y = sigmoid(z)
     return -np.sum(t*np.log(y+delta) + (1-t)*np.log(1-y)+delta)
     # return (np.sum((t-y)**2))/len(x)
#------------------------------------ [ D ]
#  수치미분
# f : 시그모이드 함수
#------------------------------------

def numerical_derivative(f,x):
    delta_x = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index

        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)

        x[idx] = tmp_val - delta_x
        fx2 = f(x)

        grad[idx] = (fx1-fx2)/(2+delta_x)

        # (f(x+delta_x) - f(x-delta_x)) / (2*delta_x)
        x[idx]= tmp_val

        it.iternext()
    return grad

#--------------------------------- [ F ]


print("ERROR VALUE(INIT)=", error_val(x_data, t_data),"WEIGHT(INIT)=",W, "BIAS(INIT)=",b)


for step in range(1000):
     W -= learning_rate * numerical_derivative(f,W)
     b -= learning_rate * numerical_derivative(f,b)
#     W = 0.88174501
#     b = 0.8346417

     if(step % 2 == 0):
         print("STEP=", step,"ERROR VALUE(LAST)=", loss_func(x_data, t_data),"WEIGHT(LAST)=",W, "BIAS(LAST)=",b)

#---------------------------------  [ Z ]

(real_val, logical_val) = predict(2)
print('CALC_Y=',real_val,'REAL_T=', logical_val)

#--------------------------------- END
