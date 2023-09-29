
import numpy as np
import matplotlib.pyplot as plt
import random
import operator
import math

class FCM:
    def __init__(self, num_clusters, fuzzy_coefficient = 2, distance = 'Euclidean'):
        self.k = num_clusters
        self.m = fuzzy_coefficient
        self.distance = distance

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
    
    def covariance_matrix(self, data, mat, cluster_center, index):
        matrix1 = np.zeros((data.shape[1], data.shape[1]))
        for i in range(len(data)):
            xi = data[i]
            diff = np.array(xi - cluster_center[index]).reshape((1,data.shape[1]))
            matrix = (mat[index][i]**self.m)*np.matmul(diff.T, diff)
            new_matrix = matrix + matrix1
            matrix1 = new_matrix
            
        den = mat[index].sum()

        return (1/den)*new_matrix
    

    def inverse_matrix(self, matrix):
        return np.linalg.inv(matrix)
    
    def maha_base(self,data, mat, cluster_center, index):
        SEED =42
        rng = np.random.default_rng(SEED)
        n = data.shape[1]
        ro = rng.uniform(low = 0, high = 1, size=self.k)
        a = ro[index]*np.linalg.det(self.covariance_matrix(data, mat, cluster_center, index))
        b = self.inverse_matrix(self.covariance_matrix(data, mat, cluster_center, index))
        return b*(a**(1/n))
    

        

        
    

    def updatemu(self,cluster_center, mat,dataset): # Updating the membership value
        
        p = float(2/(self.m-1))
        if self.distance == 'Euclidean':
            
            for i in range(len(dataset)):
                x = list(dataset[i])
                distances = [np.linalg.norm(np.array(list(map(operator.sub, x, cluster_center[j])))) for j in range(self.k)]
                for j in range(self.k):
                    den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(self.k)])
                    mat[j][i] = float(1/den)       
            return mat
        elif self.distance == 'Mahanobilis':
            for i in range(len(dataset)):
                xi = dataset[i]
                distances = [np.matmul((np.array(xi - cluster_center[j]).reshape((1,dataset.shape[1]))),np.matmul(self.maha_base(dataset, mat, cluster_center, j),(np.array(xi - cluster_center[j]).reshape((1,dataset.shape[1]))).T)) for j in range(self.k)]
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
        self.data =dataset
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
    

    def Xie_Beni_Index(self):
        inner = []
        for i in range(len(self.data)):
            exp_mat = (self.membership_mat[:,i]**self.m)#.reshape((1, self.k))
            diffs = self.data[i] - self.cluster_centers
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
                    d.append(np.linalg.norm(self.cluster_centers[m] - self.cluster_centers[n]))
        M = np.array(d).min()
        den = M*len(self.data)
        return num/den
    

    def Partition_Entropy(self):
        total = 0
        N = len(self.data)
        for i in range(self.membership_mat.shape[0]):
            for j in range(self.membership_mat.shape[1]):
                total += self.membership_mat[i][j]*math.log(self.membership_mat[i][j])
        return (-1/N)*total
    
    def FPC(self):
        total = 0
        N = len(self.data)
        for i in range(self.membership_mat.shape[0]):
            for j in range(self.membership_mat.shape[1]):
                total += self.membership_mat[i][j]**self.m
        return (1/N)*total

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
        self.data = dataset
        curr = 0
        while curr < max_iter:
            self.membership_mat = self.updatemu1(self.membership_mat,self.cluster_center, dataset)
            self.new_matrix = self.spa_inf(self.membership_mat, dataset)
            self.cluster_center = self.Cluster_Center1(self.new_matrix, dataset)
            
            cluster_labels = self.getClusters1(self.new_matrix, dataset)
            

                
            curr += 1

        #return cluster_labels, cluster_centers
        return cluster_labels,self.cluster_center, self.membership_mat
    
    def predict(self, data):
        if data.shape == (len(data), ):
            raise ValueError('reshape data (1, data.shape)')
        for i in range(len(data)):
            dist = [np.linalg.norm(data[i] - self.cluster_centers[k]) for k in range(self.cluster_centers.shape[0])]
        
        return np.argmin(np.array(dist))


    def Xie_Beni_Index(self):
        inner = []
        for i in range(len(self.data)):
            exp_mat = (self.membership_mat[:,i]**self.m)#.reshape((1, self.k))
            diffs = self.data[i] - self.cluster_center
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
        den = M*len(self.data)
        return num/den
    

    def Partition_Entropy(self):
        total = 0
        N = len(self.data)
        for i in range(self.membership_mat.shape[0]):
            for j in range(self.membership_mat.shape[1]):
                total += self.membership_mat[i][j]*math.log(self.membership_mat[i][j])
        return (-1/N)*total

    def FPC(self):
        total = 0
        N = len(self.data)
        for i in range(self.membership_mat.shape[0]):
            for j in range(self.membership_mat.shape[1]):
                total += self.membership_mat[i][j]**self.m
        return (1/N)*total
    





class Kernel_FCM:
    def __init__(self, num_clusters, initialization_type = 'fcm',kernel = 'Gaussian', 
                 fuzzy_coefficient = 2, poly_coef = 1, poly_exponent =1, poly_theta = 0,
                  Gaussian_sigma = 1, Laplacian_coef = 1, Sigmoid_theta = 0, Sigmoid_coef = 0 ):
        self.k = num_clusters
        self.m = fuzzy_coefficient
        self.kernel = kernel
        self.initilization_type = initialization_type
        self.alpha_p = poly_coef
        self.theta_p = poly_theta
        self.d_p = poly_exponent
        self.sigma_g =Gaussian_sigma
        self.alpha_l =Laplacian_coef
        self.theta_s = Sigmoid_theta
        self.alpha_s = Sigmoid_coef
        if self.kernel not in ['Gaussian', 'linear', 'Laplacian', 'Sigmoid','polynomial']:
            raise ValueError('Kernel not found')
        
        if self.initialization_type not in ['fcm', 'random1', 'random2']:
            raise ValueError('Initialisation type not found')




    
    def initilization(self, dataset):
        if self.initilization_type == 'fcm':
            from fcm_utils_ import FCM1
            model = FCM1(self.k)
            _, matrix,_ = model.fit(dataset)
            return matrix
        
        elif self.initilization_type == 'random1':
            SEED =42
            rng = np.random.default_rng(SEED)
            indices = np.random.choice(np.arange(len(dataset)), size = self.k, replace = False)
            matrix = dataset[indices] + rng.uniform(low = 0, high = 0.1)
            return matrix


        elif self.initilization_type == 'random2':
            SEED =42
            rng = np.random.default_rng(SEED)
            
            matrix = rng.uniform(size = (self.k, dataset.shape[1]))
            return matrix 
        
    def linear(self, x, y):
        return  x.T @ y
    def polynomial(self, x,y):
        prod = x.T @ y
        inner = self.alpha_p*prod + self.theta_p
        return inner**self.d_p
    def Gaussian(self, x, y):
        r =  [(x[i] -y[i])**2 for i in range(len(x))]
        distance = np.array(r).sum()
        inner = distance/(2*self.sigma_g)
        return np.exp(-1*inner) 
    def Laplacian(self, x, y):
        r =  [(x[i] -y[i])**2 for i in range(len(x))]
        distance = np.sqrt(np.array(r).sum())
        inner = self.alpha_l * distance
        return np.exp(-1*inner) 
    def Sigmoid(self, x,y):
        prod = x.dot(y)
        inner = self.alpha_s*prod + self.theta_s
        return np.tanh(inner) 
    

    def Cluster_Center2_(self, mu,cluster_center, dataset):
        mat = np.zeros_like(mu) # calculating the cluster center
        center = []
        if self.kernel == 'Gaussian':
            for i in range(self.k):
                distances = [self.Gaussian(x, cluster_center[i]) for x in dataset]
                #inner_mat = np.array([dataset[t]*distances[t] for t in range(len(dataset))]).reshape(dataset.shape)
                for j in range(len(dataset)):
                
                    mat[i][j] = ((mu[i][j])**self.m)*(self.Gaussian(dataset[j], cluster_center[i] ))
            numerator = np.matmul(mat, dataset)
            
            for k in range(mat.shape[0]):
                center.append(np.divide(numerator[k], sum(mat[k])))    
        elif self.kernel == 'polynomial':
            for i in range(self.k):
                distances = [self.polynomial(x, cluster_center[i]) for x in dataset]
                #inner_mat = np.array([dataset[t]*distances[t] for t in range(len(dataset))]).reshape(dataset.shape)
                for j in range(len(dataset)):
                
                    mat[i][j] = ((mu[i][j])**self.m)*(self.polynomial(dataset[j], cluster_center[i] ))  
            numerator = np.matmul(mat, dataset)
            
            for k in range(mat.shape[0]):
                center.append(np.divide(numerator[k], sum(mat[k])))     
        elif self.kernel == 'linear':
            for i in range(self.k):
                distances = [self.linear(x, cluster_center[i]) for x in dataset]
                #inner_mat = np.array([dataset[t]*distances[t] for t in range(len(dataset))]).reshape(dataset.shape)
                for j in range(len(dataset)):
                
                    mat[i][j] = ((mu[i][j])**self.m)*(self.linear(dataset[j], cluster_center[i] )) 
            numerator = np.matmul(mat, dataset)
            
            for k in range(mat.shape[0]):
                center.append(np.divide(numerator[k], sum(mat[k])))       
        elif self.kernel == 'Laplacian':
            for i in range(self.k):
                distances = [self.Laplacian(x, cluster_center[i]) for x in dataset]
                #inner_mat = np.array([dataset[t]*distances[t] for t in range(len(dataset))]).reshape(dataset.shape)
                for j in range(len(dataset)):
                
                    mat[i][j] = ((mu[i][j])**self.m)*(self.Laplacian(dataset[j], cluster_center[i] ))  
            numerator = np.matmul(mat, dataset)
            
            for k in range(mat.shape[0]):
                center.append(np.divide(numerator[k], sum(mat[k])))       
        elif self.kernel == 'Sigmoid':
            for i in range(self.k):
                distances = [self.Sigmoid(dataset[j], cluster_center[i]) for j in range(len(dataset))]
                #inner_mat = np.array([dataset[t]*distances[t] for t in range(len(dataset))]).reshape(dataset.shape)
                for j in range(len(dataset)):
                
                    mat[i][j] = ((mu[i][j])**self.m)*(self.Sigmoid(dataset[j], cluster_center[i] ))
            numerator = np.matmul(mat, dataset)
            
            for k in range(mat.shape[0]):
                center.append(np.divide(numerator[k], sum(mat[k])))       

        return np.array(center)
    

    def updatemu2(self, cluster_center, dataset): # Updating the membership value    
        mat = np.zeros((self.k,len(dataset)))
        if self.kernel == 'Gaussian' :
            p = float(1/(self.m-1))
        
            for i in range(len(dataset)):
                x = dataset[i]
                distances = [np.subtract(1 , self.Gaussian(x, cluster_center[c])) for c in range(self.k)]
                for j in range(self.k):
                    den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(len(distances))])
                    mat[j][i] = float(1/den)      
        elif self.kernel == 'linear' :
            p = float(1/(self.m-1))
        
            for i in range(len(dataset)):
                x = dataset[i]
                distances = [self.linear(x, x) - 2*self.linear(x, np.array(cluster_center)[c]) + self.linear(np.array(cluster_center)[c], np.array(cluster_center)[c]) for c in range(self.k)]
                for j in range(self.k):
                    den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(self.k)])
                    mat[j][i] = float(1/den) 
        elif self.kernel == 'Laplacian' :
            p = float(1/(self.m-1))
        
            for i in range(len(dataset)):
                x = dataset[i]
                distances = [self.Laplacian(x, x) - 2*self.Laplacian(x, np.array(cluster_center)[c]) + self.Laplacian(np.array(cluster_center)[c], np.array(cluster_center)[c]) for c in range(self.k)]
                for j in range(self.k):
                    den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(self.k)])
                    mat[j][i] = float(1/den) 
        elif self.kernel == 'Sigmoid' :
            p = float(1/(self.m-1))
        
            for i in range(len(dataset)):
                x = dataset[i]
                distances = [self.Sigmoid(x, x) - 2*self.Sigmoid(x, np.array(cluster_center)[c]) + self.Sigmoid(np.array(cluster_center)[c], np.array(cluster_center)[c]) for c in range(self.k)]
                for j in range(self.k):
                    den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(self.k)])
                    mat[j][i] = float(1/den)     
        elif self.kernel == 'polynomial' :
            p = float(1/(self.m-1))
        
            for i in range(len(dataset)):
                x = dataset[i]
                distances = [self.polynomial(x, x) - 2*self.polynomial(x, np.array(cluster_center)[c]) + self.polynomial(np.array(cluster_center)[c], np.array(cluster_center)[c]) for c in range(self.k)]
                for j in range(self.k):
                    den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(len(distances))])
                    mat[j][i] = float(1/den)             
        return mat
                        



    def getClusters2(self, mat, dataset): # getting the clusters
        cluster_labels = list()
        for j in range(len(dataset)):
            max_val, idx = max((val, idx) for (idx, val) in enumerate(mat[:,j]))
            cluster_labels.append(idx)
        return cluster_labels
    def fit(self, dataset, max_iter = 100): #Third iteration Random vectors from data
        # Membership Matrix
        #membership_mat, cluster_centers = init(k,m, dataset)
        self.data = dataset
        self.cluster_centers = self.initilization(dataset)
        curr = 0
        while curr < max_iter:
            #cluster_centers = Cluster_Center2(membership_mat, m, k, dataset)
            self.membership_mat = self.updatemu2(self.cluster_centers, dataset)
            self.cluster_centers  = self.Cluster_Center2_(self.membership_mat, self.cluster_centers,dataset)
            cluster_labels = self.getClusters2(self.membership_mat, dataset)
            
                
            curr += 1

        #return cluster_labels, cluster_centers
        return np.array(cluster_labels), self.cluster_centers, self.membership_mat
            








            








                    
