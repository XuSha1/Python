##encoding=utf-8
import random
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
    def over_sampling(self):
        N=int(self.N/100)*3
        #N1=3
        self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print('neighbors',neighbors)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            self._populate(N,i,nnarray)
        return self.synthetic
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1

def ReadData(file_name):
    with open(file_name, 'r') as f:
        data = []
        for line in f.readlines():
            line = line.replace('[', '')
            line = line.replace(']', '')
            data.append(eval(line))
        df = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', \
                                             'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16'])
    return df

if __name__ == '__main__':
    lis = ['G:/geetest/gt-kaggle/train_data/train_people/feature_355808/part-00000', \
            'G:/geetest/gt-kaggle/train_data/train_robot/feature_93900/part-00000', \
            'G:/geetest/gt-kaggle/test_data/test_people/feature_43955/part-00000', \
            'G:/geetest/gt-kaggle/test_data/test_robot_c/feature_37702/part-00000', \
            'G:/geetest/gt-kaggle/test_data/test_robot_Cl/feature_17139/part-00000']
    train_people = ReadData(lis[0])##355808
    train_robot = ReadData(lis[1])##93900
    X = pd.concat([train_people, train_robot], axis=0, ignore_index=True)  ##axis=0按列合并
    y = pd.DataFrame(np.vstack((np.zeros((355808, 1)),
                                    np.ones((93900, 1)))), columns=['label'])  # 0标记为人，1标记为机器
    X = np.array(X)
    y = np.array(y).reshape(-1)
    print(X[:3])
    print(y[:3])
    a=np.array(train_robot)
    s=Smote(a,N=100)
    data=s.over_sampling()
    data=pd.DataFrame(data,columns=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', \
                                    'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16'])
    print(data[:3])
    data.to_csv('G:/geetest/smote_data01.csv',index=False,sep=',')

