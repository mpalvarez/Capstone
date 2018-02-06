import numpy as np
class Lin_Reg(object):
    
    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept
        self.fitted = False
        return
        
    def add_ones_column(self, X):
        ones = np.ones(X.shape[0])
        return np.column_stack((ones, X))
       
    def fit(self,X,y):
        
        if self.fit_intercept:
            ones = self.add_ones_column(X)
            X = self.add_ones_column(X)
            
      
        self.B = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.fitted = True
        
    def predict(self, X):
        assert self.fitted, "Model hasn't been fit yet!"
        if self.fit_intercept == True:
            X = self.add_ones_column(X)
            
            yhat = X.dot(self.B)
            return yhat

# Cost Functions
   
    #Distance from data points to our fit regression line
    def residuals(self, y_pred, y_actual):
        return y_pred - y_actual
    
    #Mean Absolute Error
    def mae(self,y_pred, y_actual):
        return abs(self.residuals(y_pred, y_actual)).mean()
    
    #Residual sum of squares
        #we square to penalize large errors more
    def rss(self, y_pred, y_actual):
        return (self.residuals(y_pred, y_actual)**2).sum()
    
    #Mean Squared Error
        #Also penalizes larger errors more heavily than small errors
    def mse(self,y_pred, y_actual):
        return (self.residuals(y_pred, y_actual)**2).mean()
    
    #Root of the MSE to bring value back into the scope of original data
    def rmse(self,y_pred, y_actual):
        return np.sqrt(self.mse(y_pred, y_actual))

    #R^2
    def sstot(self,y_actual):
        return ((y_actual - y_actual.mean())**2).sum()

    def r2(self,y_pred, y_actual):
        return 1 - (self.sse(y_pred, y_actual) / self.sstot(y_actual))
    
#LINEAR REGRESSION WITH GRADIENT DESCENT

class Lin_Reg_GD(object):
    
    def __init__(self,fit_intercept = True,n_iterations = 100, gamma = 0.1):
        self.n_iterations = n_iterations
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.fitted = False
        
    
    def add_ones_column(self,X):
        ones = np.ones(X.shape[0])
        return np.column_stack((ones, X))
    
    def get_loss_GRADIENT(self,X, y, B):
        ypred = X.dot(B)
        residuals = y - ypred
        #same as squarig residuals
        loss = residuals.T.dot(residuals)
        
        #gradient
        dloss = 1
        dr = (2*residuals) * dloss
        dypred = -1 * dr
        dB = X.T.dot(dypred)
        
        return loss , dB
        
    def predict(self, X, y):
        X_ones = self.add_ones_column(X)
        B = np.random.randn(X_ones.shape[1])
    
        loss, grad = self.get_loss_GRADIENT(X_ones, y, B)
    
        for _ in range(self.n_iterations):
        
            loss ,grad = self.get_loss_GRADIENT(X_ones, y, B)
        
            B = B - self.gamma * grad
        
            loss_updated, grad_updated = self.get_loss_GRADIENT(X_ones, y, B)
        
            if loss_updated > loss:
                self.gamma *= 0.1
            
            loss = loss_updated
            
            print("loss:", loss_updated)
    
        return B
    
    def fit(self,X,y):
        if self.fit_intercept:
            ones = np.ones(X.shape[0])
            X = np.column_stack((ones, X))
            
            
        self.B = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.fitted = True





















