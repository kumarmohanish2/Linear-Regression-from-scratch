#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Questions:01). Write a function to generate a data matrix X. Inputs: Number of samples, feature dimension. Output: Data matrix X. [1]#
import numpy as np
def Mat_Gen(R,C):  
    X = np.random.rand(R,C)     # Generating Matrix of R rows and C columns 
    return X
R = 8     # No. of Samples
C = 6     # No. of varibles
W0 = np.ones((R, 1))  # bias vector of all ones
W = np.random.rand(C,1)  # random coefficients of column vector of matrix X
mu, sigma = 0, 0.1 # mean and standard deviation
noise = np.random.normal(mu, sigma, (R,1)) # normal random variable guassian noise
Mat_Gen(R,C)   #Calling function to generate Data Matrix of R amples and C features  


# In[2]:


# Comment01:- Here In this Question:01 I have generated data matrix of R no. of samples and C no. of columns.


# In[3]:


#Questions:02:-Write a function to generated dependent variable column t.  [1]
# X is data matrix 
# W is weight vector
# W0 is the bias
# noise is the guassin random variable noise
def Indep_Variable(X,W,W0,noise):  
    t = np.dot(X, W) + W0 + noise  # target vector is defined
    return  t     # returning the target vector t
X = Mat_Gen(R,C)    # Data MAtrix is generated using the function Mat_Gen Defined in the question no. 01
Indep_Variable(X,W,W0,noise)    # Called the function to generate the t vector 


# In[4]:


#Comment02:-  Here in the above question:02 the the independent Variable t is generated taking X data matrix , W weight vector, W0 is bias and noise is the normal guassian random variable noise.


# In[5]:


#Questions:03:- Write a function to compute a linear regression estimate. [1]
# X is data matrix 
# W is weight vector
# y is the predicted variable
def Lin_Regression(X,W):
    
    y = np.dot(X, W)    # estimating the linear regression , multiplying X data matrix with the W weight vector
    return y   # returning the linear regression estimate
Lin_Regression(X,W)    # Calling the function to compute linear regression estimate
    
    


# In[6]:


# Comment:03:- Here the given problem above have calculated the linear regression estimate


# In[7]:


#Questions:04:-Write a function to compute the mean square error of two vectors y and t. [1]
# X is data matrix 
# W is weight vector
# y is predicted varible
# t is target variable

def Mean_Square(y,t): # defining a function to compute Mean square error
    MSE = np.square(np.subtract(y,t)).mean()   # Calculating mean square error  
    return MSE                 # returning Mean square error
t = Indep_Variable(X,W,W0,noise)   # Generating t target variable
y = Lin_Regression(X,W)   # estimating y predicted variable
Mean_Square(y,t)   # Calling function to comnpute the Mean square error


# In[8]:


# Comment:04:- In the question:04 mean square error between and y and t is calculated


# In[9]:


# question 5
from numpy.linalg import inv

def Wei_Of_Lin_Rig(X, t, Lambda, Identity_Dim):    #Defining function to estimate weights using pseodo inverse method
    X_t = X.transpose()          # Transpose of Data matrix X 
    I = np.identity(Identity_Dim)  # Identity matrix of dimention C*C
    res1 = np.dot(X_t, X)          # (X^T)*X 
    res2 = I*Lambda                # Multiplying Lambda with the Identity matrix
    res3 = np.dot(X_t, t)          # Multiplying transpose of X with the target vector t
    res4 = np.add(res2 , res1)     #Adding res1 and res2
    res5 = inv(res4)               # Calculating inverse 
    w = np.dot(res5, res3)         
    y = np.dot(X, w)            
    M = Mean_Square(y,t)           # Computing mean square error 
    return w, y, M                 # returning the weight vector , predicted vector , and Mean square error 

Wei_Of_Lin_Rig(X,t, R,C)          # Calling function to find weight vector using pseudo inverse method

    
    
    
    
    


# In[10]:


# Quetion No. 06


def Grad_MSE(X, t, W,C):
   
    bias = 0.5  # defining bias
    s= 0 
    G = np.zeros((C,1))    # Create an array of all zeros
    for j in range(C):
        for i in range(R):
            row_r1 = X[i, :] #Extraction of the features of Data matrix
            p = np.dot(row_r1, W)    # features multiplied by weight vector
            q = np.subtract(np.add(p, bias),t[i]) # error of i th element of the array
            r=np.multiply(q,X[i,j])  # multiplying error with the x[i,j] element of the data matrix
            s = np.add(s,r) # adding all errors after multiplying with x[i,j] element till No. of samples
        G[j]= s  # Storing Gradient in array called G
        s=0
    factor = (-5) # taking mean square error
    G1 = np.divide(G, factor)
    return G1    # returning the gradient vector of MSE w.r.t. weight,s of the feature
Grad_MSE(X,t,W,C)   # Calling funtion to calculate Gradient of MSE


# In[11]:


# Question No. 07:- Function to write L2 norm of a vector w. 
def L2_Norm(w):
    
    L2 = np.sum(np.power((w),2))  # computing the L2 norm
    return w,L2     # Returning wieght vector and L2 norm 


L2_Norm(W)  # returns the L2 norm of the weight vector w
  


# In[12]:


# Question No. 08:-Function to write the Gradient of L2 norm w.r.t. w
def Gradient_L2(w):     # define the function to calculate the L2 norm
   
    G = np.multiply(w, 2)  # computing the gradient of L2 norm
    return G      # Returning the gradient of L2 norm  

Gradient_L2(W)  # returns the gradient of L2 norm of the weight vector w


# In[13]:


# Question No. 09:-Function to write L1 norm of a vector w.
from scipy.linalg import norm
def L1_Norm(w):   # Defining  the function to calculate the L1 norm of w vector
   
    L1 = norm(w, 1)  # computing the L1 norm
    return w,L1 # returning the weight vector and L2 norm
L1_Norm(W)  # returns the L1 norm of the weight vector w
  


# In[14]:


# Question No. 10:- Function to write the Gradient of L1 norm w.r.t w 
def Gradient_L1(w,C):     # Defining Function to Calculate the Gradient of L1 norm 
    Grad_L1 = np.zeros((C, 1));   # Initializing the Vector Grad_L1 with zero entries 
    for k in range(C):
        if w[k] < 0 :
            Grad_L1[k]= -1
        else:
            Grad_L1[k] = 1
    return Grad_L1
     
Gradient_L1(W,C)  # Calling function to calculate the Gradient of L1 norm
#Comment:- The logic applied to find the Gradient of L1 norm is that if entry of weight vector element is negative then the corresponding Gradient vector entry is -1 one , otherwise entry for the gradient vector is +1; 
                
    
        
     
    


# In[15]:


# Question no. 11:-Write a function for a single update of weights of linear regression using gradient descent.
def gradient_Descent(X, t, W_old, Lambda1, Lambda2, eta,C ):   # Defining function to calculate to return weight vector using graient descent
   
 
    P_1 = Grad_MSE(X,t,W_old,C)   # Gradient Of MSE has been generated
    P_2 = Gradient_L2(W_old)  # returns the gradient of L2 norm of the weight vector W_old
    P_3 = Gradient_L1(W_old,C)  # returns the gradient of L1 norm of the weight vector W_old
    L= P_1 + (P_2*Lambda1) + (P_3*Lambda2)   # Gradient of the loss function
    W_new = W_old - (eta*L) # returns the new updated weight vector
    y = Lin_Regression(X,W_new)   # y is predicted 
    New_MSE = Mean_Square(y,t)    # Mean square error is comuted
    return W_new, New_MSE       # returning the Weight vector , and Updated MSE
    

W_old = np.ones((C,1))  # weight of all ones is generated 


gradient_Descent(X, t, W_old , 0.0000011, 0.0011, 0.0011,C)    # Calling function to calculate the weight vector using gradient dscent 
    


# In[16]:


# Question no. 12:-Write a function for a single update of weights of linear regression using gradient descent.
import math
def gradient_Descent_with_iteration(X, t, W_old, Lambda1, Lambda2, eta, m, n,C):   # Defining the function to Compute the weight vector using Gradient descent with iteration
    # m in min no. of iteration 
    # n min. change in NMRSE
    Newest_W = W_old
    Newest_MSE = 5 
    Old_NRMSE = 5
    Va = np.var(t)  # Variance of vactor t
    Std_Dev = math.sqrt(Va)  # Standard Deviation of the t
    for i in range(m):
        Newest_W , Newest_MSE = gradient_Descent(X, t, Newest_W , Lambda1, Lambda2 , eta,C) # storing Weight vecor and MSE in a tupke named result
        RMSE = math.sqrt(Newest_MSE)  # Calculates Root Mean Square MSE
        N_RMSE = np.divide(RMSE,Std_Dev) # Calculates the Normalized mean square error
        Diff = N_RMSE - Old_NRMSE  # Change in NRMSE
        Old_NRMSE = N_RMSE   # Assign New NRMSE to Old NRMSE
        Abs_Diff = np.absolute(Diff)  # Absolute Difference
        print(Abs_Diff)
        if Abs_Diff < n:   # Checking the condition if the Change in NRMSE is less than specified value n
            break
    #print("End of for loop")
   
    return Newest_W, N_RMSE  # Returning the weight vector and normalized root mean square value



gradient_Descent_with_iteration(X, t, W_old , 0.1, 0.1, 0.0011, 100, 0.0001,C)   # Calling function 
    


# In[21]:


#Question:13. Run multiple experiments (with different random seeds) for, plot the results of (box plots), and comment on the trends and potential reasons for the following relations:

# Part:-a) Training and validation NRMSE obtained using pseudo inverse with number of training samples [2]
import matplotlib.pyplot as plt
import random



# Here we are training the data
C = 4
lambada = 5
MSE_array = np.array([0])   # Initializing an array to store the NRMSE Values 
MSE_new = {}     # Initializing an array to store an array of NRMSE's for each seed
for j in range(6):
    rng = np.random.default_rng(seed=j*10)     # defining seed
    for i in range(1,5):
        X = Mat_Gen(10**i,C)      # Data matrix generation with No. of samples varying
        W0 = np.ones((10**i, 1))  # bias vector of all ones
        W = np.random.rand(4,1)  # random coefficients of column vector of matrix X
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, (10**i,1)) # normal random variable guassian noise
        t = Indep_Variable(X,W,W0, noise)    # target  variable
        W_true, Y_New, MSE = Wei_Of_Lin_Rig(X,t, lambada,4)
        NRMSE = np.sqrt(MSE/np.var(t))
        MSE_array =np.append(MSE_array, NRMSE)
    MSE_array = np.delete(MSE_array,0)    
    MSE_new[j] = MSE_array
    MSE_array = np.array([0])
for k in range(5):
    print(MSE_new[k])

fig = plt.figure(figsize = (5,5))
plt.boxplot([MSE_new[0],MSE_new[1],MSE_new[2],MSE_new[3],MSE_new[4]])
plt.show()

#Here Now we will test the data on smaller no. of samples
D = 4
lambada1 = 5
MSE_array1 = np.array([0])
MSE_new1 = {}
for j in range(6):
    rng = np.random.default_rng(seed=j*10)
    for i in range(1,5):
        X1 = Mat_Gen(8**i,D)
        W0 = np.ones((8**i, 1))  # bias vector of all ones
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise1 = np.random.normal(mu, sigma, (8**i,1)) # normal random variable guassian noise
        t_test = Indep_Variable(X1,W_true,W0, noise1)
        W_test, Y_test, MSE_test = Wei_Of_Lin_Rig(X1,t_test, lambada1,4)
        NRMSE_test = np.sqrt(MSE_test/np.var(t_test))
        MSE_array1 =np.append(MSE_array1, NRMSE_test)
    MSE_array1= np.delete(MSE_array1,0)
    MSE_new1[j] = MSE_array1
    MSE_array1 = np.array([0])
for l in range(5):
    print(MSE_new1[l])
    
fig1 = plt.figure(figsize = (5,5))
plt.boxplot([MSE_new1[0],MSE_new1[1],MSE_new1[2],MSE_new1[3],MSE_new1[4]])
plt.show()




#Comment:- For the varying no. of samples if we increases the no. of the samples the NRMSE Decreases 
        


# In[ ]:





# In[23]:


#Question:13. 
# Part:-b) Training and validation NRMSE obtained using pseudo inverse with number of variables [2]
import matplotlib.pyplot as plt
import random
R = 10
lambada = 5
MSE_array = np.array([0])
MSE_new = {}
R = 10
lambada1 = 5
MSE_array1 = np.array([0])
MSE_new1 = {}
for j in range(6):
    rng = np.random.default_rng(seed=j*10)
    for i in range(1,5):
        X = Mat_Gen(10,4**i)
        W0 = np.ones((10, 1))  # bias vector of all ones
        W = np.random.rand(4**i,1)  # random coefficients of column vector of matrix X
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, (10,1)) # normal random variable guassian noise
        t = Indep_Variable(X,W,W0, noise)
        W_true, Y_New, MSE = Wei_Of_Lin_Rig(X,t, lambada,4**i)
        NRMSE = np.sqrt(MSE/np.var(t))
        MSE_array =np.append(MSE_array, NRMSE)
        #*************** Here for validation set the NRMSE is generated*************# 
        X1 = Mat_Gen(R,4**i)
        W0 = np.ones((R, 1))  # bias vector of all ones
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise1 = np.random.normal(mu, sigma, (R,1)) # normal random variable guassian noise
        t_test = Indep_Variable(X1,W_true,W0, noise1)
        W_test, Y_test, MSE_test = Wei_Of_Lin_Rig(X1,t_test, lambada1,4**i )
        NRMSE_test = np.sqrt(MSE_test/np.var(t_test))
        MSE_array1 =np.append(MSE_array1, NRMSE_test)
    MSE_array = np.delete(MSE_array,0)   
    MSE_new[j] = MSE_array
    MSE_array = np.array([0])
    MSE_array1= np.delete(MSE_array1,0)
    MSE_new1[j] = MSE_array1
    MSE_array1 = np.array([0])
for k in range(5):
    print(MSE_new1[k])

fig1 = plt.figure(figsize = (5,5))
plt.boxplot([MSE_new1[0],MSE_new1[1],MSE_new1[2],MSE_new1[3],MSE_new1[4]])
plt.show()



#Comment:-  With Increase in no. of variables the The NRMSE decreses for Each seed 


# In[26]:


#Question:13.
# Part:-c) Training and validation NRMSE obtained using pseudo inverse with noise variance [2]
import matplotlib.pyplot as plt
import random
R = 10
lambada = 5
MSE_array = np.array([0])
MSE_new = {}
lambada1 = 5
MSE_array1 = np.array([0])
MSE_new1 = {}
for j in range(6):
    rng = np.random.default_rng(seed=j*10)
    for i in range(1,5):
        X = Mat_Gen(10,4)
        W0 = np.ones((10, 1))  # bias vector of all ones
        W = np.random.rand(4,1)  # random coefficients of column vector of matrix X
        mu, sigma = 0, 0.1*i # mean and standard deviation
        noise = np.random.normal(mu, sigma, (10,1)) # normal random variable guassian noise
        t = Indep_Variable(X,W,W0, noise)
        W_true, Y_New, MSE = Wei_Of_Lin_Rig(X,t, lambada,4)
        NRMSE = np.sqrt(MSE/np.var(t))
        #print(W_true)
        #print(W_true)
        #print(Y_New)
        #print(MSE)
        MSE_array =np.append(MSE_array, NRMSE)
        # Here we will generate Test Targest Vector and NRMSE for Validation set
        X1 = Mat_Gen(10,4)
       # print(X1)
        W0 = np.ones((10, 1))  # bias vector of all ones
        # W = np.random.rand(4,1)  # random coefficients of column vector of matrix X
        mu, sigma1 = 0, 0.2*i # mean and standard deviation
        noise1 = np.random.normal(mu, sigma1, (10,1)) # normal random variable guassian noise
        t_test = Indep_Variable(X1,W_true,W0, noise1)
        #print(t_test)
        W_test, Y_test, MSE_test = Wei_Of_Lin_Rig(X1,t_test, lambada1,4 )
        NRMSE_test = np.sqrt(MSE_test/np.var(t_test))
        #print(W_true)
        #print(Y_New)
        #print(MSE)
        MSE_array1 =np.append(MSE_array1, NRMSE_test)
    MSE_array = np.delete(MSE_array,0)       
    MSE_new[j] = MSE_array
    MSE_array = np.array([0])
    MSE_array1= np.delete(MSE_array1,0)
    MSE_new1[j] = MSE_array1
    MSE_array1 = np.array([0])
for k in range(5):
    print(MSE_new1[k])


fig1 = plt.figure(figsize = (5,5))
plt.boxplot([MSE_new1[0],MSE_new1[1],MSE_new1[2],MSE_new1[3],MSE_new1[4]])
plt.show()


#Comment:-  Here in the program Boxplot for the Validation set is shown and NRMSE is decreasing with increasing in the noise variance in standard normal guassian random variable



# In[31]:


#Question:13.
# Part:-d) Training and validation NRMSE obtained using pseudo inverse with w0 [2]
import matplotlib.pyplot as plt
import random
R = 10
lambada = 5
MSE_array = np.array([0])
MSE_new = {}
lambada1 = 5
MSE_array1 = np.array([0])
MSE_new1 = {}
for j in range(6):
    rng = np.random.default_rng(seed=j*10)
    for i in range(1,5):
        X = Mat_Gen(10,4)
        W0 = (np.ones((10, 1)))*i  # random bias vector
        W = np.random.rand(4,1)  # random coefficients of column vector of matrix X
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, (10,1)) # normal random variable guassian noise
        t = Indep_Variable(X,W,W0, noise)
        W_true, Y_New, MSE = Wei_Of_Lin_Rig(X,t, lambada,4)
        NRMSE = np.sqrt(MSE/np.var(t))
        MSE_array =np.append(MSE_array, NRMSE)
        
        # Here we will generate Test Targest Vector and NRMSE for Validation set
        X1 = Mat_Gen(10,4)
        W1 = (np.ones((10, 1)))*i*2  # random bias vector 
        mu, sigma1 = 0, 0.1 # mean and standard deviation
        noise1 = np.random.normal(mu, sigma1, (10,1)) # normal random variable guassian noise
        t_test = Indep_Variable(X1,W_true,W0, noise1)
        W_test, Y_test, MSE_test = Wei_Of_Lin_Rig(X1,t_test, lambada1,4 )
        NRMSE_test = np.sqrt(MSE_test/np.var(t_test))
        MSE_array1 =np.append(MSE_array1, NRMSE_test)
    MSE_array = np.delete(MSE_array,0)      
    MSE_new[j] = MSE_array
    MSE_array = np.array([0])
    MSE_array1= np.delete(MSE_array1,0)
    MSE_new1[j] = MSE_array1
    MSE_array1 = np.array([0])
for k in range(6):
    print(MSE_new1[k])


fig1 = plt.figure(figsize = (5,5))
plt.boxplot([MSE_new1[0],MSE_new1[1],MSE_new1[2],MSE_new1[3],MSE_new1[4]])
plt.show()


#Comment :- With increase in the values of the , bias vector in each iteration, No any trends has been seen in the NRMSE. Arbitrarily changing. 



# In[32]:


#Question:13.
# Part:-e) Training and validation NRMSE obtained using pseudo inverse with lambda2 [2]
import matplotlib.pyplot as plt
import random
R = 10
lambada = 5
MSE_array = np.array([0])
MSE_new = {}
lambada1 = 5
MSE_array1 = np.array([0])
MSE_new1 = {}
for j in range(6):
    rng = np.random.default_rng(seed=j*10)
    for i in range(1,5):
        X = Mat_Gen(10,4)
        W0 = (np.ones((10, 1)))  # random bias vector
        W = np.random.rand(4,1)  # random coefficients of column vector of matrix X
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, (10,1)) # normal random variable guassian noise
        t = Indep_Variable(X,W,W0, noise)
        W_true, Y_New, MSE = Wei_Of_Lin_Rig(X,t, lambada*i,4)
        
        MSE_array =np.append(MSE_array, NRMSE)
        
        
        
        # Here we will generate Test Targest Vector and NRMSE for Validation set
        X1 = Mat_Gen(10,4)
        W1 = (np.ones((10, 1)))  # random bias vector 
        mu, sigma1 = 0, 0.1 # mean and standard deviation
        noise1 = np.random.normal(mu, sigma1, (10,1)) # normal random variable guassian noise
        t_test = Indep_Variable(X1,W_true,W0, noise1)
        W_test, Y_test, MSE_test = Wei_Of_Lin_Rig(X1,t_test, lambada1*10*i,4 )
        NRMSE_test = np.sqrt(MSE_test/np.var(t_test))
        MSE_array1 =np.append(MSE_array1, NRMSE_test)
    MSE_array = np.delete(MSE_array,0)     
    MSE_new[j] = MSE_array
    MSE_array = np.array([0])
    MSE_array1= np.delete(MSE_array1,0)
    MSE_new1[j] = MSE_array1
    MSE_array1 = np.array([0])
for k in range(6):
    print(MSE_new1[k])

fig1 = plt.figure(figsize = (5,5))
plt.boxplot([MSE_new1[0],MSE_new1[1],MSE_new1[2],MSE_new1[3],MSE_new1[4]])
plt.show()

# Comment:-with incerese in lambda no any trends in the NRMSE has been seen.



# In[34]:


#Question:13.
# Part:-f) Time taken to solve pseudo inverse with number of samples and number of variables and its breaking points [2]
# Import time module
import time
# record start time
start = time.time()
import matplotlib.pyplot as plt
import random
C = 4
lambada = 5
for i in range(1,10000000000):
        X = Mat_Gen(100*i,20**i)  # Generating data matrix
        W0 = np.ones((100*i,1)) # bias vector of all ones
        W = np.random.rand(20**i,1) #random coefficients of column vector of matrix X
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, (100*i,1))###rmal random variable guassian noise
        t = Indep_Variable(X,W,W0, noise)  # Target variable
        W_true, Y_New, MSE = Wei_Of_Lin_Rig(X,t, lambada,20**i)   # Calling function to return the weight matrix by pseudo inverse method
        end = time.time()    #record the end time 
        print("The time of execution of above program is :",
        (end-start) * 10**3, "ms")
        
        print("The index i is :", i)
        
        
        
        
        #Comment:- While increasing rows 100 in each iteration and 20 to the power i column in each iteration the following error has occured;
        #MemoryError: Unable to allocate 191. GiB for an array with shape (160000, 160000) and data type float64
        #Time taken :- 126 seconds
        #Breaking point:- 191Gb
        
 


# In[35]:


#Question:13.
# Part:-g) Training and validation NRMSE obtained using gradient descent with max_iter [2]
import matplotlib.pyplot as plt
import random
import random as rnd
R = 10
eta = 1
MSE_array = np.array([0])
MSE_new = {}
MSE_array1 = np.array([0])
MSE_new1 = {}
for j in range(6):
    rng = np.random.default_rng(seed=j*10)
    for i in range(1,5):
        X = Mat_Gen(10,4)
        
        W0 = (np.ones((10, 1)))  # random bias vector
        W = np.random.rand(4,1)  # random coefficients of column vector of matrix X
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, (10,1)) # normal random variable guassian noise
        t = Indep_Variable(X,W,W0, noise)
        Lambda1 = rnd.uniform(0,1)   # Lambda is random normal Variable
        Lambda2 = rnd.uniform(0,1)    # Lambda is random normal Variable
        W_true, NRMSE = gradient_Descent_with_iteration(X, t, W, Lambda1, Lambda2, eta, 10*i , 1000,4)
        NRMSE = np.sqrt(MSE/np.var(t))
        MSE_array =np.append(MSE_array, NRMSE)
       
    # Here we will generate Test Targest Vector and NRMSE for Validation set
        X1 = Mat_Gen(10,4)
        W1 = (np.ones((10, 1)))  # random bias vector 
        t_test = Indep_Variable(X1,W_true,W0, noise)
        Lambda3 = rnd.uniform(0,1) # Lambda is random normal Variable
        Lambda4 = rnd.uniform(0,1)  # Lambda is random normal Variable
        W_test, MSE_test = gradient_Descent_with_iteration(X, t, W, Lambda3, Lambda4, eta, 100*i , 1000,4)

        NRMSE_test = np.sqrt(MSE_test/np.var(t_test))
        MSE_array1 =np.append(MSE_array1, NRMSE_test)
    MSE_array = np.delete(MSE_array,0)      
    MSE_new[j] = MSE_array
    MSE_array = np.array([0])
    MSE_array1= np.delete(MSE_array1,0)
    MSE_new1[j] = MSE_array1
    MSE_array1 = np.array([0])
for k in range(5):
    print(MSE_new1[k])
fig1 = plt.figure(figsize = (5,5))
plt.boxplot([MSE_new1[0],MSE_new1[1],MSE_new1[2],MSE_new1[3],MSE_new1[4]])
plt.show()

# No any trends has been seen 



# In[36]:


#Test
#Question:13. 
#Part:-h) Training and validation NRMSE obtained using gradient descent with eta [2]
import matplotlib.pyplot as plt
import random
import random as rnd
R = 10
eta = 1
MSE_array = np.array([0])
MSE_new = {}
MSE_array1 = np.array([0])
MSE_new1 = {}
for j in range(6):
    rng = np.random.default_rng(seed=j*10)
    for i in range(1,5):
        X = Mat_Gen(10,4)
        
        W0 = (np.ones((10, 1)))  # random bias vector
        W = np.random.rand(4,1)  # random coefficients of column vector of matrix X
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, (10,1)) # normal random variable guassian noise
        t = Indep_Variable(X,W,W0, noise)
        Lambda1 = rnd.uniform(0,1)   # Lambda is random normal Variable
        Lambda2 = rnd.uniform(0,1)    # Lambda is random normal Variable
        W_true, NRMSE = gradient_Descent_with_iteration(X, t, W, Lambda1, Lambda2, eta*i,  1000, 1000,4)
        NRMSE = np.sqrt(MSE/np.var(t))
        MSE_array =np.append(MSE_array, NRMSE)
       
    # Here we will generate Test Targest Vector and NRMSE for Validation set
        X1 = Mat_Gen(10,4)
        W1 = (np.ones((10, 1)))  # random bias vector 
        # W = np.random.rand(4,1)  # random coefficients of column vector of matrix X
        t_test = Indep_Variable(X1,W_true,W0, noise)
        
        Lambda3 = rnd.uniform(0,1) # Lambda is random normal Variable
        Lambda4 = rnd.uniform(0,1)  # Lambda is random normal Variable
        W_test, MSE_test = gradient_Descent_with_iteration(X, t, W, Lambda3, Lambda4, eta*i*2 , 1000, 1000,4)
        
        NRMSE_test = np.sqrt(MSE_test/np.var(t_test))
        
        MSE_array1 =np.append(MSE_array1, NRMSE_test)
        
    MSE_new[j] = MSE_array
    MSE_array = np.array([0])
    MSE_new1[j] = MSE_array1
    MSE_array1 = np.array([0])
for k in range(5):
    
    print(MSE_new1[k])

fig1 = plt.figure(figsize = (5,5))
plt.boxplot([MSE_new1[0],MSE_new1[1],MSE_new1[2],MSE_new1[3],MSE_new1[4]])
plt.show()

#Comment:- No any trends has been seen


# In[21]:



# Question No.13, i)
import time
# record start time
start = time.time()
import matplotlib.pyplot as plt
import random
# = 2
for i in range(1,10000000000):
#def Time_rec(R1, C1):
    
    C1 = 10000**i
    R1 = 50**i
    X1 = Mat_Gen(R1,C1)    # Data matrix has been generated
    W_0 = np.ones((R1, 1))  # bias vector of all ones
    W_1 = np.random.rand(C1,1)  # random coefficients of column vector of matrix X
    mu1, sigma1 = 0, 0.1 # mean and standard deviation
    noise1 = np.random.normal(mu1, sigma1, (R1,1)) # normal random variable guassian noise
    t1 = Indep_Variable(X1,W_1,W_0,noise1)    # Traget vector has been generated
    W_old1 = np.ones((C1,1))  # weight of all ones is generated 
    W_test, NRMSE =  gradient_Descent_with_iteration(X1, t1, W_old1 , 0.1, 0.1, 0.0011, 100, 100000000, C1)
    print("W_true is :", W_test)
    print("NRMSE is :", NRMSE)
    end = time.time()
    print("The time of execution of above program is :",
    (end-start) * 10**3, "ms")
    print("The index i is :", i)

    # Comment:-MemoryError: Unable to allocate 1.82 TiB for an array with shape (2500, 100000000) and data type float64.
    # this program ran for  2 seconds
    # Breaking point:-1.82TiB


    


# In[38]:


#Test
#Question:13. 
#Part:-k) Training and validation NRMSE and number of nearly zero weights obtained using gradient descent with lambda2 [2]
import matplotlib.pyplot as plt
import random
import random as rnd
R = 10
eta = 1
MSE_array = np.array([0])
MSE_new = {}
MSE_array1 = np.array([0])
MSE_new1 = {}
W_Array = {} 
for j in range(6):
    rng = np.random.default_rng(seed=j*10)
    for i in range(1,5):
        X = Mat_Gen(10,4)
        
        W0 = (np.ones((10, 1)))  # random bias vector
        W = np.random.rand(4,1)  # random coefficients of column vector of matrix X
        mu, sigma = 0, 0.1 # mean and standard deviation
        noise = np.random.normal(mu, sigma, (10,1)) # normal random variable guassian noise
        t = Indep_Variable(X,W,W0, noise)
        Lambda1 = 1  
        Lambda2 = 2   
        W_true, NRMSE = gradient_Descent_with_iteration(X, t, W, Lambda1, Lambda2*2*i, eta,  1000, 1000,4)
        NRMSE = np.sqrt(MSE/np.var(t))
        MSE_array =np.append(MSE_array, NRMSE)
       
    # Here we will generate Test Targest Vector and NRMSE for Validation set
        X1 = Mat_Gen(10,4)
        W1 = (np.ones((10, 1)))  # random bias vector 
        # W = np.random.rand(4,1)  # random coefficients of column vector of matrix X
        t_test = Indep_Variable(X1,W_true,W0, noise)
        #print(t_test)
        Lambda3 = 1
        Lambda4 = 2
        W_test, MSE_test = gradient_Descent_with_iteration(X, t, W, Lambda3, Lambda4*i, eta , 1000, 1000,4)
       
        NRMSE_test = np.sqrt(MSE_test/np.var(t_test))
        #print(W_true)
        #print(Y_New)
        #print(MSE)
        MSE_array1 =np.append(MSE_array1, NRMSE_test)
        
    MSE_new[j] = MSE_array
    MSE_array = np.array([0])
    MSE_new1[j] = MSE_array1
    MSE_array1 = np.array([0])
for k in range(5):
   
    print(MSE_new1[k])


fig1 = plt.figure(figsize = (5,5))
plt.boxplot([MSE_new1[0],MSE_new1[1],MSE_new1[2],MSE_new1[3],MSE_new1[4]])
plt.show()
# Decreasing with increase in lambda


# In[ ]:




