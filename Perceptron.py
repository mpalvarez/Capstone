import numpy as np
#from Logistic_regression import Logistic_Regression_momentum


class Perceptron(object):
     
    def __init__(self, In =1 , Out = 1, Hidden = 2, lr = .01,n_iterations = 1000):
        self.lr = lr
        self.In = In
        self.out = Out
        self.hidden_layers = Hidden
        self.epochs = n_iterations
        self.z_w = 0
        self.z_b = 0
        self.b = np.zeros((1,self.In))
        
        return
        
        
    def gradient(self, probs, X, y):
        #backpass
        dscores = probs
        dscores[range(X.shape[0]),y] -= 1
        dscores /= X.shape[0]
        
        dW = np.dot(X.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)
        
        new_dw , new_db = self.calculateMomentum(dW, db)
        
        return dW, db
    
    
    #def calculateMomentum(self, w_grad,b_grad):
    #    betha= 0.81
    #    
    #    #calculate the momentum
    #    self.z_w = self.lr*self.z_w + w_grad
    #    self.z_b = self.lr*self.z_b + b_grad
    #    
    #    #Set the new 'better' updated 'w' and 'b'   
    #    w_updated = self.W - self.lr*self.z_w
    #    b_updated = self.b - self.lr*self.z_b
    #    
    #    return w_updated, b_updated 
    
        
    def softmax(self,X,y,W):
        #forward pass
        scores = X.dot(W) + self.b
        exp_score = np.exp(scores)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        logitprobs = -np.log(probs[range(X.shape[0]),y])
        data_loss = np.sum(logitprobs)/X.shape[0]
        return data_loss, probs
    
    def train(self,X,y):
        self.In = X.shape[1]
        self.wh = np. random.uniform(size=(inputlayer_neurons,self.hidden))
        self.bh = np.random.uniform(size=(1,hiddenlayer_neurons))
        self.wout = np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
        self.bout = np.random.uniform(size=(1,output_neurons))
            
       # for i in self.epochs:
       # h1 = 
        
       
       
    
    
    def score(self,):
        pass
    
    
    
    