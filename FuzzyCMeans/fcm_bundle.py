
import numpy as np
import matplotlib.pyplot as plt
import random
import operator
import math

class FCM:
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
    








class Noise_Robust_FCM:
    def __init__(self, num_clusters, fuzzy_coefficient = 2, initialization_method = 'fcm', square_window_radius = 1e-1, spatial_relevance_index = 0):
        self.k = num_clusters
        self.m = fuzzy_coefficient
        self.r = square_window_radius
        self.init_met = initialization_method
        self.alpha = spatial_relevance_index



    def initialization_membership(self, dataset):
        if self.init_met == 'all_equal':
            SEED =42
            rng = np.random.default_rng(SEED)
            mat = rng.random((self.k, len(dataset)))
            matrix = np.zeros_like(mat)
            for i in range(len(dataset)):
                for j in range(self.k):
                    n = mat[:,i]
                    matrix[j][i] = mat[j][i]/np.sum(n)
            return matrix
        
        elif self.init_met == 'fcm':
            from fcm_utils_ import FCM1
            model = FCM1(self.k)
            _,_, matrix = model.fit(dataset)
            return matrix
        

    def abs_vec(self, x,y):
        r =  [(x[i] -y[i])**2 for i in range(len(x))]
        return np.array(r).sum()

    def dist(self, x,y):
        
        r =  [(x[i] -y[i])**2 for i in range(len(x))]
        f = np.sqrt(np.array(r).sum())
        return f


    def Square_Windows(self, x,  dataset):
        
        N_r = [y for y in dataset for i in range(len(x))  if abs(x[i] - y[i]) < self.r and abs(x[i] - y[i]) != 0]
        return np.array(N_r)
    
    
    def Spatial_Info(self, dataset, mem, point_index,  cluster_index):
        xi = dataset[point_index]
        Neighbours = self.Square_Windows(xi,dataset)
        N_r = len(Neighbours)
        numerator = []
        denominator = []

        for i in range(N_r):
            r = np.argwhere(dataset == Neighbours[i])[0]
            mem_value = mem[cluster_index][r]
                
            spa_inf = self.dist(xi, Neighbours[i])
            inv_d = 1/(self.abs_vec(xi, Neighbours[i]))
            inner_1 = mem_value*spa_inf*inv_d
            numerator.append(inner_1) 
            inner_2 = spa_inf*inv_d
            denominator.append(inner_2)

        num = np.array(numerator).sum()
        den = np.array(denominator).sum()
    
        SDSM = num/den
        return SDSM
    


    def Sdmv(self, dataset, mem, point_index,  cluster_index):
        xi = dataset[point_index]
        Neighbours = self.Square_Windows(xi, dataset)
        N_r = len(Neighbours)
    
        sdmv = []
        for i in range(N_r):
            r = np.argwhere(dataset == Neighbours[i])[0]
            mem_value = mem[cluster_index][r]
                
            
            inv_d = 1/(self.abs_vec(xi, Neighbours[i]))

            inner_3 = mem_value*inv_d
            sdmv.append(inner_3)
        
        SDMV =np.array(sdmv).sum()

        return SDMV
    


    def Cluster_Center1(self, mu, dataset):
        mat = np.zeros_like(mu) # calculating the cluster center
        for i in range(self.k):
            for j in range(len(dataset)):
                mat[i][j] = (mu[i][j])**self.m
        numerator = np.matmul(mat, dataset)
        center  =  []
        for k in range(mat.shape[0]):
            center.append(np.divide(numerator[k], sum(mat[k])))
        return np.array(center)
    

    def updatemu1(self, mat, cluster_center, dataset): # Updating the membership value
        p = float(2/(self.m-1))
        
        for i in range(len(dataset)):
            x = dataset[i]
            distances = [self.abs_vec(x, cluster_center[j])*(1 - self.alpha * self.Spatial_Info(dataset, mat, i, j)) for j in range(self.k)]
            for j in range(self.k):
                den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(self.k)])
                mat[j][i] = float(1/den)       
        return np.array(mat)
    


    def spa_inf(self, mat, dataset):
        matrix = np.zeros_like(mat)
        k = mat.shape[0]
        for i in range(mat.shape[1]):
            SD = [self.Sdmv(dataset, mat, i,j) for j in range(k)]
            for j in range(k):
                inn = mat[j][i] * (SD[j])**2
                prod = np.array(SD)*np.array(SD)
            
                some = mat[:,i] * prod
                matrix[j][i] = float(inn/(np.array(some).sum())) 
        return matrix
    

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


    def getClusters1(self, mat, dataset): # getting the clusters
        cluster_labels = list()
        for j in range(len(dataset)):
            idx = np.argmax(mat[:,j])
            cluster_labels.append(idx)
        return cluster_labels
    

    def fit(self, dataset, max_iter = 100): #Third iteration Random vectors from data
        SEED =42
        rng = np.random.default_rng(SEED)
        self.cluster_center = rng.uniform(size = (self.k, dataset.shape[1])) 
        self.membership_mat = self.initialization_membership(dataset)
        curr = 0
        while curr < max_iter:
            self.membership_mat = self.updatemu1(self.membership_mat,self.cluster_center, dataset)
            self.new_matrix = self.spa_inf(self.membership_mat, dataset)
            self.cluster_center = self.Cluster_Center1(self.new_matrix, dataset)
            
            cluster_labels = self.getClusters1(self.new_matrix, dataset)
            

                
            curr += 1

        #return cluster_labels, cluster_centers
        return cluster_labels,self.cluster_center, self.membership_mat
    



    def Xie_Beni_Index(self, data):
        inner = []
        for i in range(len(data)):
            exp_mat = (self.membership_mat[:,i]**self.m)#.reshape((1, self.k))
            diffs = data[i] - self.cluster_center
            c = []
            for j in range(len(exp_mat)):
                c.append(exp_mat[j]*diffs[j])

            prod = np.array(c).sum()
            inner.append(prod)
        num=np.array(inner).sum()
        d = []
        
        for m in range(self.k):
            for n in range(self.k):
                if m !=n:
                    d.append(np.linalg.norm(self.cluster_center[m] - self.cluster_center[n]))
        M = np.array(d).min()
        den = M*len(data)
        return num/den









            
