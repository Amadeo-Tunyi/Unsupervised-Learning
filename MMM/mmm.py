import numpy as np
import math
from scipy.stats import dirichlet, multinomial

class MultinomialMixtureModel():


    def __init__(self, n_clusters, restarts = 10,  tol = 1e-2, max_iter = 100):
        self.n_clusters = n_clusters
        self.tol = tol
        self.max_iter = max_iter
        self.restarts = restarts



    def init_params(self, data):
        seed = 42
        rng = np.random.default_rng(seed = 42)
        dim = data.shape[1]
        weights = rng.uniform(0,1, size = self.n_clusters)
        alpha = weights / weights.sum()
        beta = dirichlet.rvs([2 * dim] * dim, self.n_clusters)
        return alpha, beta
    

    def multinomial_pmf(self, data, beta, log = False):
        n = data.sum(axis= 1)
        m = multinomial(n, beta)
        if log:
            return m.logpmf(data)
        return m.pmf(data)
    


    
    def e_step(self, data,  beta, pi):
        likelihood = np.zeros((len(data), self.n_clusters))
        #responsibility = np.zeros_like(likelihood)
        for i in range(self.n_clusters):
            
            betai = beta[i] 
            # if any(np.isnan(mui)):
            #     new_mu, new_sigma, new_pi = init(data, n_clusters)
            #     mui = new_mu[i]
            #     covi = new_sigma[i]
            

            #normal_dist = scipy.stats.multivariate_normal(mean = mu[i], cov = sigma[i])
            likelihood[:,i] = self.multinomial_pmf(data= data,  beta=betai)
        numerator = (likelihood*pi) + 1e-18
        denominator = numerator.sum(axis = 1)[:, np.newaxis] 
        denominator = denominator 
        

        responsibility = numerator/denominator

        return responsibility
    

    def m_step(self, data,beta, pi, responsibility):
    

        for i in range(self.n_clusters):
            weight = responsibility[:, [i]]
            total_weight = weight.sum()
            beta[i] = (data * weight).sum(axis=0) / (data * weight).sum()       

            pi[i] = responsibility[:,i].sum()/len(data)
        return np.array(beta), np.array(pi)
    


    def log_loss(self,  X, alpha, beta, gamma):

        loss = 0
        for k in range(beta.shape[0]):
            weights = gamma[:, k]
            loss += np.sum(weights * (np.log(alpha[k]) + self.multinomial_pmf(X, beta[k], log=True)))
            loss -= np.sum(weights * np.log(weights))
        return -1*loss
    


    def mmm(self, data):
        import matplotlib.pyplot as plt
        iter = 0
        pi, beta= self.init_params(data)
        l = []
        loss = 0
        while iter < self.max_iter:
            prev_loss = loss
            responsibility = self.e_step(data, beta, pi)
            beta, pi = self.m_step(data,beta, pi,  responsibility)
            loss = self.log_loss( data, pi, beta, responsibility)
            l.append(loss)
        
            
            iter += 1
            #print(f'loss: {loss}')
            print('Loss: %f' % loss)
            if iter > 0 and np.abs((loss - prev_loss)) < self.tol:
                print(iter)
                break
        return np.array(pi), np.array(beta), np.argmax(np.array(responsibility), axis = 1), loss
    
    def fit(self, X):
        self.X = X
        self.best_loss = np.float('inf')
        self.best_pi = None
        self.best_beta = None
        self.best_responsibility = None

        for it in range(self.restarts):
            print('iteration %i' % it)
            pi, beta, responsibility, loss = self.mmm(X)
            if loss < self.best_loss:
                print('better loss on iteration %i: %.10f' % (it, loss))
                self.best_loss = loss
                self.best_pi = pi
                self.best_beta = beta
                self.best_responsibility = responsibility

        return self.best_loss, self.best_pi, self.best_beta, self.best_responsibility
    



    def score_sample(self, x_i):
        x = np.array(x_i).reshape((1,len(x_i)))
        resp_at_point = self.e_step(x, self.best_beta, self.best_pi )
 
    
        return np.exp(-1*self.log_loss(x, self.best_pi, self.best_beta, resp_at_point))

        
        

    

    




    



        