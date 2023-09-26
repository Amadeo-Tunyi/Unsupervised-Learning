import numpy as np
import matplotlib.pyplot as plt
import random
import operator
import math

class FCM1:
    def __init__(self, num_clusters, fuzzy_coefficient = 2):
        self.k = num_clusters
        self.m = fuzzy_coefficient

    def initialize_membership(self, dataset):
        SEED =42
        rng = np.random.default_rng(SEED)
        mat = rng.random((self.k, len(dataset)))
        matrix = np.zeros_like(mat)
        for i in range(len(dataset)):
            for j in range(self.k):
                n = mat[:,i]
                matrix[j][i] = mat[j][i]/np.sum(n)
        return matrix    
    
    def Cluster_Center(self, mu,dataset):
        mat = np.zeros_like(mu) # calculating the cluster center
        for i in range(self.k):
            for j in range(len(dataset)):
                mat[i][j] = (mu[i][j])**self.m
        numerator = np.matmul(mat, dataset)
        center  =  []
        for k in range(mat.shape[0]):
            center.append(np.divide(numerator[k], sum(mat[k])))
        return center
    

    def updatemu(self,cluster_center, mat,dataset): # Updating the membership value
        p = float(2/(self.m-1))
        for i in range(len(dataset)):
            x = list(dataset[i])
            distances = [np.linalg.norm(np.array(list(map(operator.sub, x, cluster_center[j])))) for j in range(self.k)]
            for j in range(self.k):
                den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(self.k)])
                mat[j][i] = float(1/den)       
        return mat

    def getClusters(self, mat, dataset): # getting the clusters
        cluster_labels = list()
        for j in range(len(dataset)):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(mat[:,j]))
            cluster_labels.append(idx)
        return cluster_labels
    

    def cost(self, data, membership_mat, cluster_center):
        inner = []
        for i in range(len(data)):
            exp_mat = (membership_mat[:,i]**self.m)#.reshape((1, self.k))
            diffs = data[i] - cluster_center
            c = []
            for j in range(len(exp_mat)):
                c.append(exp_mat[j]*diffs[j])

            prod = np.array(c).sum()
            inner.append(prod)
        return np.array(inner).sum()


    


    def fit(self, dataset, max_iter = 100, print_loss = False): #Third iteration Random vectors from data
        # Membership Matrix
        self.membership_mat = self.initialize_membership(dataset)
        curr = 0
        acc=[]
        while curr < max_iter:
            self.cluster_centers = self.Cluster_Center(self.membership_mat, dataset)
            self.membership_mat = self.updatemu(self.cluster_centers, self.membership_mat, dataset)
            loss = self.cost(dataset, self.membership_mat, self.cluster_centers)
            cluster_labels = self.getClusters(self.membership_mat, dataset)
            
            acc.append(cluster_labels)
            if print_loss == True:
                print(f'Epoch {curr}.......loss{loss}')
            
            # if(curr == 0):
            #     print("Cluster Centers:")
            #     print(np.array(self.cluster_centers))
            curr += 1

        #return cluster_labels, cluster_centers
        return np.array(cluster_labels), np.array(self.cluster_centers), np.array(self.membership_mat)
    



    def predict(self, data):
        if data.shape == (len(data), ):
            raise ValueError('reshape data (1, data.shape)')
        for i in range(len(data)):
            dist = [np.linalg.norm(data[i] - self.cluster_centers[k]) for k in range(self.cluster_centers.shape[0])]
        
        return np.argmin(np.array(dist))
    