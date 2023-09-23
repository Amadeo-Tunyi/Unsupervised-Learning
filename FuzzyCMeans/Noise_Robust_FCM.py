
import numpy as np
import matplotlib.pyplot as plt
import random
import operator
import math
from sklearn import datasets

#note that this code works for numpy arrays, but can easily be fine tuned for dataframes
def mu(k, dataset):
    SEED =42
    rng = np.random.default_rng(SEED)
    mat = rng.random((k, len(dataset)))
    matrix = np.zeros_like(mat)
    for i in range(len(dataset)):
        for j in range(k):
            n = mat[:,i]
            matrix[j][i] = mat[j][i]/np.sum(n)
    return matrix
def Cluster_Center(mu, m, k, dataset):
    mat = np.zeros_like(mu) # calculating the cluster center
    for i in range(k):
        for j in range(len(dataset)):
            mat[i][j] = (mu[i][j])**m
    numerator = np.matmul(mat, dataset)
    center  =  []
    for k in range(mat.shape[0]):
        center.append(np.divide(numerator[k], sum(mat[k])))
    return center
def updatemu(mat, m, k, dataset): # Updating the membership value
    p = float(2/(m-1))
    cluster_center = Cluster_Center(mat, m,k, dataset )
    for i in range(len(dataset)):
        x = list(dataset[i])
        distances = [np.linalg.norm(np.array(list(map(operator.sub, x, cluster_center[j])))) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            mat[j][i] = float(1/den)       
    return mat
def getClusters(mat, dataset): # getting the clusters
    cluster_labels = list()
    for j in range(len(dataset)):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(mat[:,j]))
        cluster_labels.append(idx)
    return cluster_labels
def fcm(k, m, dataset, max_iter): #Third iteration Random vectors from data
    # Membership Matrix
    membership_mat = mu(k, dataset)
    curr = 0
    acc=[]
    while curr < max_iter:
        cluster_centers = Cluster_Center(membership_mat, m, k, dataset)
        membership_mat = updatemu(membership_mat, m, k, dataset)
        cluster_labels = getClusters(membership_mat, dataset)
        
        acc.append(cluster_labels)
        
        if(curr == 0):
            print("-----------")
            
        curr += 1

    #return cluster_labels, cluster_centers
    return np.array(cluster_labels), np.array(cluster_centers), np.array(membership_mat)

def init(k,m, dataset):
    SEED =42
    rng = np.random.default_rng(SEED)
    mat = rng.uniform(size = (k, dataset.shape[1]))
    _, _, matrix = fcm(k, m, dataset, 100)
    
    
    return mat, matrix
def abs_vec(x,y):
    r =  [(x[i] -y[i])**2 for i in range(len(x))]
    return np.array(r).sum()

def dist(x,y):
    
    r =  [(x[i] -y[i])**2 for i in range(len(x))]
    f = np.sqrt(np.array(r).sum())
    return f


def Square_Windows(x,r,  dataset):
    
    N_r = [y for y in dataset for i in range(len(x))  if abs(x[i] - y[i]) < r and abs(x[i] - y[i]) != 0]
    return np.array(N_r)
def Spatial_Info(dataset, mem, point_index,  cluster_index):
    xi = dataset[point_index]
    Neighbours = Square_Windows(xi, 1e-1, dataset)
    N_r = len(Neighbours)
    numerator = []
    denominator = []

    for i in range(N_r):
        r = np.argwhere(dataset == Neighbours[i])[0]
        mem_value = mem[cluster_index][r]
            
        spa_inf = dist(xi, Neighbours[i])
        inv_d = 1/(abs_vec(xi, Neighbours[i]))
        inner_1 = mem_value*spa_inf*inv_d
        numerator.append(inner_1) 
        inner_2 = spa_inf*inv_d
        denominator.append(inner_2)

    num = np.array(numerator).sum()
    den = np.array(denominator).sum()
   
    SDSM = num/den
    return SDSM
def Sdmv(dataset, mem, point_index,  cluster_index):
    xi = dataset[point_index]
    Neighbours = Square_Windows(xi,1e-1, dataset)
    N_r = len(Neighbours)
  
    sdmv = []
    for i in range(N_r):
        r = np.argwhere(dataset == Neighbours[i])[0]
        mem_value = mem[cluster_index][r]
            
        
        inv_d = 1/(abs_vec(xi, Neighbours[i]))

        inner_3 = mem_value*inv_d
        sdmv.append(inner_3)
    
    SDMV =np.array(sdmv).sum()

    return SDMV
            
            

def Cluster_Center1(mu, m, k, dataset):
    mat = np.zeros_like(mu) # calculating the cluster center
    for i in range(k):
        for j in range(len(dataset)):
            mat[i][j] = (mu[i][j])**m
    numerator = np.matmul(mat, dataset)
    center  =  []
    for k in range(mat.shape[0]):
        center.append(np.divide(numerator[k], sum(mat[k])))
    return center
def updatemu1(mat, cluster_center, m, k,alpha, dataset): # Updating the membership value
    p = float(2/(m-1))
    
    for i in range(len(dataset)):
        x = dataset[i]
        distances = [abs_vec(x, cluster_center[j])*(1 - alpha * Spatial_Info(dataset, mat, i, j)) for j in range(k)]
        for j in range(k):
            den = sum([math.pow(float(distances[j]/distances[c]), p) for c in range(k)])
            mat[j][i] = float(1/den)       
    return mat
def spa_inf(mat, dataset):
    matrix = np.zeros_like(mat)
    k = mat.shape[0]
    for i in range(mat.shape[1]):
        SD = [Sdmv(dataset, mat, i,j) for j in range(k)]
        for j in range(k):
            inn = mat[j][i] * (SD[j])**2
            prod = np.array(SD)*np.array(SD)
        
            some = mat[:,i] * prod
            matrix[j][i] = float(inn/(np.array(some).sum())) 
    return matrix

def getClusters1(mat, dataset): # getting the clusters
    cluster_labels = list()
    for j in range(len(dataset)):
        idx = np.argmax(mat[:,j])
        cluster_labels.append(idx)
    return cluster_labels
def nr_fcm(k, m, dataset,alpha, max_iter): #Third iteration Random vectors from data
    # Membership Matrix
    cluster_center, membership_mat = init(k,m, dataset)
    curr = 0
    acc=[]
    while curr < max_iter:
        membership_mat = updatemu1(membership_mat,cluster_center, m, k, alpha, dataset)
        new_matrix = spa_inf(membership_mat, dataset)
        cluster_center = Cluster_Center1(new_matrix, m, k, dataset)
        
        cluster_labels = getClusters1(new_matrix, dataset)
        
        
        
        if(curr == 0):
            print("---------------")
            
        curr += 1

    #return cluster_labels, cluster_centers
    return np.array(cluster_labels), np.array(cluster_center), np.array(membership_mat)