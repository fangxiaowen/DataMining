import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import KDTree

class KMeans():
   
    # Change this!
    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=1, algorithm='auto'):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        #self.precompute_distances = precompute_distances
        self.n_init = n_init
        #self.verbose = verbose
        #self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        #self.algorithm = algorithm
        self._clusterAssment = None
        self._labels = None
        self._sse = None
    
    
    def fit(self, X):
        m = X.shape[0]    # sample size
        #一个m*2的二维矩阵，矩阵第一列存储样本点所属的族的索引值，
        #第二列存储该点与所属族的质心的平方误差
        self._clusterAssment = np.zeros((m,2)) 
        # initialize centroids
        if self.init=='k-means++':
            self._centroids = _init_centroids_kmeansPlusPlus(X, self.n_clusters)
        else:
            self._centroids = _init_centroids(X, self.n_clusters)
        clusterChanged = True   # a flag
        print(self._centroids)
        for _ in range(self.n_init):    # iterate several times howe to converge
            clusterChanged = False
         
            tree = KDTree(self._centroids, leaf_size=2)
            
            distance, indices = tree.query(X)
                        
            if not (self._clusterAssment[:,0] == indices).all():
                clusterChanged = True
                #print(indices, distance)
                self._clusterAssment = np.concatenate((indices.reshape(len(indices),1), distance.reshape(len(distance),1)), axis=1)
                #print(self._clusterAssment)
            if not clusterChanged:#若所有样本点所属的族都不改变,则已收敛,结束迭代
                break
            
            for i in range(self.n_clusters):
                #print(self._clusterAssment[:,0])
                points = X[np.nonzero(self._clusterAssment[:,0] == i)]
                if len(points) == 0:
                    print('What is going on?')
                #print(points)
                self._centroids[i,:] = np.mean(points, axis=0)
        
        self._labels = self._clusterAssment[:,0]
        self._sse = sum(self._clusterAssment[:,1])
        print(self._labels)
        return self._labels
        
    def predict(self, X):#根据聚类结果，预测新输入数据所属的族
        #类型检查
        if not isinstance(X,np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")
        
        m = X.shape[0]#m代表样本数量
        preds = np.empty((m,))
        for i in range(m):#将每个样本点分配到离它最近的质心所属的族
            minDist = np.inf
            for j in range(self._k):
                distJI = self._calEDist(self._centroids[j,:],X[i,:])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds
    
        
# use a better initialization method    
def _init_centroids(X, k, random_state=None, x_squared_norms=None,
                    init_size=None):
   """Generate k center within the range of data set."""
   n = X.shape[1] # features
   centroids = np.zeros((k,n)) # init with (0,0)....
    
   sample = X[np.unique(np.random.choice(len(X),len(X)//5))]  # sample the data
   
   #find centroids
   centroids[0] = sample[np.random.choice(len(sample), 1)]
   for i in range(1, k):
       dis = pairwise_distances(centroids[:i], sample)
       dis_sum = np.sum(dis, axis=0)
       ind = np.argmax(dis_sum)
       centroids[i] = sample[ind]
   return centroids



def _init_centroids_kmeansPlusPlus(X, k, random_state=None, x_squared_norms=None,
                    init_size=None):
    
   n = X.shape[1] # features
   centroids = np.zeros((k,n)) # init with (0,0)....
   centroids[0] = X[np.random.choice(len(X), 1)]
   for i in range(1, k):
       dis = pairwise_distances(centroids[:i], X)   # should be ^2
       dis_min = np.amin(dis, axis=0)
       prob = dis_min / np.sum(dis_min)
       ind = np.random.choice(np.arange(len(X)), p=prob)
       centroids[i] = X[ind]
   return centroids












