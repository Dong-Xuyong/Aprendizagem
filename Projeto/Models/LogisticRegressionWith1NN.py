import numpy as np
import copy

# GRADED FUNCTION: sigmoid

def sigmoid(z):
    
    s = 1/(1+np.exp(-z))
    return s

# GRADED FUNCTION: initialize_with_zeros

def initialize_with_zeros(dim):

    w = np.zeros((dim,1))
    b = 0.0
    
    return w, b

# GRADED FUNCTION: propagate

def propagate(w, b, X, Y):

    
    m = X.shape[1]
    

    A = sigmoid(np.dot(w.T, X) + b)
    
    cost = -1/m * np.sum(Y*np.log(A+0.0000000001) + (1-Y)*np.log(1.0000000001-A))
    #print(A.max())
    #print(np.sum(np.log(1-A)))

    dw = 1/m * np.dot(X, (A-Y).T)
    db = 1/m * np.sum(A-Y)
    
    # YOUR CODE ENDS HERE
    cost = np.squeeze(np.array(cost))

    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost



# GRADED FUNCTION: optimize

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):

    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)
    
        dw = grads["dw"]
        db = grads["db"]
      

        w = w - learning_rate * dw
        b = b - learning_rate * db

        #if i % 100 == 0:
        if i % 10 == 0:
            costs.append(cost)

            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
                
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs



# GRADED FUNCTION: predict

def predict(w, b, X):

    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    

    A = sigmoid(np.dot(w.T, X) + b)
    
    
    for i in range(A.shape[1]):
        

        if A[0, i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
        

    
    return Y_prediction

# GRADED FUNCTION: model

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    
    # YOUR CODE STARTS HERE
    w, b = initialize_with_zeros(X_train.shape[0])

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = params['w']
    b = params['b']
    print(costs)
    
    Y_prediction_test = predict(w,b, X_test)
    Y_prediction_train = predict(w,b, X_train)
    

    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d