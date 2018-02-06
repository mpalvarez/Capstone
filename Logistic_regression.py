import numpy as np

class Logistic_Regression(object):
    
    def __init__(self, lr = .01, n_iterations = 1000):
        self.lr = lr
        self.n_iterations = n_iterations
        
        
        
    def softmax_grad(self,X,y, b, W):
        
        #forward pass
        scores = X.dot(W) + b
        exp_score = np.exp(scores)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        logitprobs = -np.log(probs[range(X.shape[0]),y])
        data_loss = np.sum(logitprobs)/X.shape[0]
        
        #backpass
        dscores = probs
        dscores[range(X.shape[0]),y] -= 1
        dscores /= X.shape[0]
        
        dW = np.dot(X.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)
        
        return data_loss, dW, db, probs
     

    def fit(self, X, y):
        K = X.shape[0]
        D = X.shape[1]
        b = np.zeros((1,K))
        W = 0.01 * np.random.randn(D,K)
        
        loss, gradW, gradb, probs = self.softmax_grad(X, y, b, W)
        for _ in range(self.n_iterations):
            loss_updated, gradW, gradb, probs = self.softmax_grad(X, y, b, W)
            print(loss)
            if loss_updated >= loss:
                b -= self.lr * gradb
                W -= self.lr * gradW
                
            loss = loss_updated
    
        return b , W
    

            
class Logistic_Regression_momentum(object):
    
    def __init__(self, lr = .01, n_iterations = 1000):
        self.lr = lr
        self.n_iterations = n_iterations
        self.z_w = 0
        self.z_b = 0
        
    def gradient(self,probs, X, y):
        #backpass
        dscores = probs
        dscores[range(X.shape[0]),y] -= 1
        dscores /= X.shape[0]
        
        dW = np.dot(X.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)
        
        new_dw , new_db = self.calculateMomentum(dW, db)
        
        return new_dw, new_db
    
    
    def calculateMomentum(self, w_grad,b_grad):
        betha= 0.81
        
        #calculate the momentum
        self.z_w = self.lr*self.z_w + w_grad
        self.z_b = self.lr*self.z_b + b_grad
        
        #Set the new 'better' updated 'w' and 'b'   
        w_updated = self.W - self.lr*self.z_w
        b_updated = self.b - self.lr*self.z_b
        
        return w_updated,b_updated 
    
        
    def softmax(self,X,y):
        #forward pass
        scores = X.dot(self.W) + self.b
        exp_score = np.exp(scores)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        logitprobs = -np.log(probs[range(X.shape[0]),y])
        data_loss = np.sum(logitprobs)/X.shape[0]
        return data_loss, probs
    
    def predict(self,X):
        scores = X.dot(self.W) + self.b
        exp_score = np.exp(scores)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        return probs

    def fit(self, X, y):
        v = 0
        K = X.shape[0]
        D = X.shape[1]
        self.b = np.zeros((1,K))
        self.W = 0.01 * np.random.randn(D,K)
        
        
        loss, probs = self.softmax(X, y)
        for _ in range(self.n_iterations):
            loss_updated, new_probs = self.softmax(X, y)
            gradW, gradb = self.gradient(new_probs,X, y)
            print(loss)
            self.b = gradb 
            #print(b)
            self.W = gradW
            #print(W)

                
            loss = loss_updated
    
        return self.b, self.W     
    
    def performance(self):
            
    #def loss(X, y, W):
    #    #log likelihood
    #    scores = np.dot(X, W)
    #    ll = np.sum( y*scores - np.log(1 + np.exp(scores)) )
    #    return ll
        
    #def sig_gradient(a):
    #    b =  a * -1
    #    c = np.exp(b)
    #    d = c +1
    #    e = d**-1
    #    
    #    #backpass
    #    de = 1
    #    dd = (-1/d**2) *de
    #    dc = (1)* dd
    #    db = (np.e**b) * dc
    #    da = -1 * db
    #
    #return e, da
        