#Author: Siddheshwari Bankar

from sklearn.neighbors import NearestNeighbors
from random import randrange
import numpy as np
import pandas as pd
import random
from math import *
from random import randrange
from scipy import stats
from scipy.stats import gamma
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class SMOTEBase:
    def __init__(self, original_df: pd.DataFrame, minority_column_label: str, minority_class_label: str) -> None:
        self.original_df = original_df
        self.minority_column_label = minority_column_label
        self.minority_class_label = minority_class_label

    def pre_processing(self, num_to_synthesize:int) -> pd.DataFrame:
        self.original_df = self.original_df.dropna()
        self.original_df['synthetic_data'] = 0
        self.minority_df = self.original_df[self.original_df[self.minority_column_label] == self.minority_class_label].copy()
        self.majority_df = self.original_df[self.original_df[self.minority_column_label] != self.minority_class_label].copy()   
        
        if num_to_synthesize <= 0 and (self.majority_df.shape[0] - self.minority_df.shape[0]) == 0:
                num_to_synthesize = self.minority_df.shape[0] * 2
                
        if num_to_synthesize <= 0 and (self.majority_df.shape[0] - self.minority_df.shape[0]) > 0:
            num_to_synthesize = self.majority_df.shape[0] - self.minority_df.shape[0]
            
        self.num_to_synthesize = num_to_synthesize
        return self.minority_df, self.majority_df , self.num_to_synthesize


    def data_generator(self, num_to_synthesize) -> pd.DataFrame:
        return self.pre_processing(num_to_synthesize)
    
    def combine_data(self, synthetic_data):
        class_counts_og = self.original_df['class'].value_counts()
        synthetic_data_notog = synthetic_data['class'].value_counts()
        balanced_df = pd.concat([self.original_df, synthetic_data], axis=0)
        return balanced_df
         
    def findNeighbors(self, value, allValues, k):
        nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(allValues)
        dist, indices = nn.kneighbors(value, return_distance=True)
        return dist, indices
    
    def calculateRadius(self, mi, mi_num, ma, ma_num, numAttr, mu = 1):
        E = 0
        for i in range(mi_num):
            x = mi[i]
            x = np.reshape(x, (-1, numAttr))
            
            dist, _ = self.findNeighbors(x, ma, ma_num-1)
            Ei = dist.sum()
            E += Ei
        Emean = mu * (E / (mi_num * ma_num))
        return Emean

    def calculateSampleDensity(self, value, r, mi, ma):
        mi_nn = NearestNeighbors(radius = r).fit(mi)
        ma_nn = NearestNeighbors(radius = r).fit(ma)
        mi_rng = mi_nn.radius_neighbors(value)
        ma_rng = ma_nn.radius_neighbors(value)
        density = 0.8 * len(np.asarray(ma_rng[1][0])) + 0.2 * len(np.asarray(mi_rng[1][0]))
        return density

    def calculateAverageDistance(self, c1, c1_num, c2, numAttr, k):
        D = 0
        for i in range(c1_num):
            x = c1[i]
            x = np.reshape(x, (-1, numAttr))
            
            dist, _ = self.findNeighbors(x, c2, k)
            Di = dist.sum() / k
            D += Di
        D = D/c1_num
        return D

    def LogisticRegressionCLF(self, sample_set, org_set):
        from sklearn.model_selection import train_test_split
        x_train = sample_set.values
        y_train = sample_set["class"].values
        x_test = org_set.values
        y_test = org_set["class"].values
        model = LogisticRegression()
        clf = model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        AUC = roc_auc_score(y_test, prediction)

        return AUC
    
    def calculateControlCoefficient(self, value, k, mi, ma, Dn, Dp, printDebug=False):
        dist, _ = self.findNeighbors(value, mi, k)
        D1 = dist.sum() / k

        dist, _ = self.findNeighbors(value, ma, k)
        D2 = dist.sum() / k

        u = (D1 * Dn) / (D2 * Dp)
        if u < 1:
            cc = random.uniform(0, 1)
        elif (u >= 1) & (u <= 2):
            cc = 0.5 + 0.5 * random.uniform(0, 1)
        elif u > 2:
            cc = 0.8 + 0.2 * random.uniform(0, 1)
        if printDebug == True:
            print("D1 = ", D1)
            print("D2 = ", D2)
            print("u = ", u)
        return cc

    def split(self, x, n):
        arr = []
        if x < n:  
            print("ERROR")
            return arr
        elif (
            x % n == 0
        ):  
            for i in range(n):
                arr.append(x // n)
        else:
            zp = n - (x % n)
            pp = x // n
            for i in range(n):
                if i >= zp:
                    arr.append(pp + 1)
                else:
                    arr.append(pp)
        return arr

class SMOTE(SMOTEBase):
    def __init__(self, original_df: pd.DataFrame, minority_column_label: str, minority_class_label: str) -> None:
        self.model_name = 'SMOTE'
        super().__init__(original_df, minority_column_label, minority_class_label)

    def smote(self):
        
        self.ma_num = self.majority_df.shape[0]
        self.mi_num = self.minority_df.shape[0]
        self.df_num =  self.ma_num +  self.mi_num
         
        if (self.mi_num - 1) < 5:
            k = self.mi_num - 1
        else:
            k = 5
        
        mi = self.minority_df.to_numpy()
        num_attributes = mi.shape[1]
        syntheticIndex = 0
        syntheticArray = np.empty((self.num_to_synthesize, num_attributes))

        for i in range(self.num_to_synthesize):
            x = random.choice(mi)
            x = np.reshape(x, (-1, num_attributes))
            _, knn = self.findNeighbors(x, mi, k)
            y = randrange(1, k + 1)

            diff = mi[knn[0, y]] - x
            gap = random.uniform(0, 1)
            syntheticArray[syntheticIndex] = x + gap * diff
            syntheticIndex += 1
            
        synthetic_data = pd.DataFrame(syntheticArray, columns=self.minority_df.columns.values)
        synthetic_data['synthetic_data'] = 1
        return synthetic_data

    def data_generator(self, num_to_synthesize:int=0) -> pd.DataFrame:
        synthetic_df = None
        self.minority_df, self.majority_df, self.num_to_synthesize = super().data_generator(num_to_synthesize)
        synthetic_data = self.smote()
        return super().combine_data(synthetic_data)

class SDD_SMOTE(SMOTEBase):
    def __init__(self, original_df: pd.DataFrame, minority_column_label: str, minority_class_label: str) -> None:
        self.model_name = 'SDD_SMOTE'
        super().__init__(original_df, minority_column_label, minority_class_label)    

    def sdd_smote(self):
        
        self.ma_num = self.majority_df.shape[0]
        self.mi_num = self.minority_df.shape[0]
        self.df_num =  self.ma_num +  self.mi_num
        
        if (self.mi_num - 1) < 5:
            k = self.mi_num - 1
        else:
            k = 5
        
        df = self.original_df.to_numpy()    
        mi = self.minority_df.to_numpy()
        ma = self.majority_df.to_numpy()
        num_attributes = mi.shape[1]
        syntheticIndex = 0
        syntheticArray = np.empty((self.num_to_synthesize, num_attributes))
        
        radius = self.calculateRadius(mi, self.mi_num, ma, self.ma_num, num_attributes, mu = 2)
        
        densityList = []
        for i in range(self.mi_num):
            sample = mi[i]
            sample = np.reshape(sample, (-1, num_attributes))
            
            D = self.calculateSampleDensity(sample, radius, mi, ma)

            densityList.append((i, D))
        densityList.sort(key = lambda x: x[1], reverse = True)
        
        Dpos = self.calculateAverageDistance(mi, self.mi_num, mi, num_attributes, k)
        Dneg = self.calculateAverageDistance(mi, self.mi_num, ma, num_attributes, k)

        densityIndex = 0
        for i in range(self.num_to_synthesize):
            if densityIndex >= len(densityList):
                densityIndex = 0
                
            instance = densityList[densityIndex]
            index = instance[0]
            x = mi[index]
            x = np.reshape(x, (-1, num_attributes))
            _, knn = self.findNeighbors(x, mi, k)

            y = randrange(1, k+1)
            diff = mi[knn[0, y]] - x
            syntheticArray[syntheticIndex] = x + diff
            syntheticIndex += 1
            densityIndex += 1
            
        synthetic_data = pd.DataFrame(syntheticArray, columns=self.minority_df.columns.values)
        synthetic_data['synthetic_data'] = 1
        return synthetic_data

    def data_generator(self, num_to_synthesize:int=0) -> pd.DataFrame:
        synthetic_df = None
        self.minority_df, self.majority_df, self.num_to_synthesize = super().data_generator(num_to_synthesize)
        synthetic_data  = self.sdd_smote()
        return super().combine_data(synthetic_data)

class Gamma_SMOTE(SMOTEBase):
    def __init__(self, original_df: pd.DataFrame, minority_column_label: str, minority_class_label: str) -> None:
        self.model_name = 'Gamma_SMOTE'
        super().__init__(original_df, minority_column_label, minority_class_label)

    def gamma_smote(self):
        
        self.ma_num = self.majority_df.shape[0]
        self.mi_num = self.minority_df.shape[0]
        self.df_num =  self.ma_num +  self.mi_num
        
        mi = self.minority_df.to_numpy()
        num_attributes = mi.shape[1]
        syntheticIndex = 0
        syntheticArray = np.empty((self.num_to_synthesize, num_attributes))
        
        if (self.mi_num - 1) < 5:
            k = self.mi_num - 1
        else:
            k = 5
            
        alpha = 0.5
        beta = 0.0125
        maxCD = beta*(alpha-1)

        for i in range(self.num_to_synthesize):
            x = random.choice(mi) 
        
            x = np.reshape(x, (-1, num_attributes))
            
            _, ind = self.findNeighbors(x, mi, k)
            y = randrange(1, k+1)
            x_prime = mi[ind[0, y]]
            v = x_prime - x
            t = stats.gamma.rvs(beta, alpha, random_state=None)
            p = x + (t - maxCD) * v

            syntheticArray[syntheticIndex] = p
            syntheticIndex += 1
                 
            synthetic_data = pd.DataFrame(syntheticArray, columns=self.minority_df.columns.values)
            synthetic_data['synthetic_data'] = 1

        return synthetic_data

    def data_generator(self, num_to_synthesize:int=0) -> pd.DataFrame:
        synthetic_df = None
        self.minority_df, self.majority_df, self.num_to_synthesize = super().data_generator(num_to_synthesize)
        synthetic_data = self.gamma_smote()
        return super().combine_data(synthetic_data)

class Gaussian_SMOTE(SMOTEBase):
    def __init__(self, original_df: pd.DataFrame, minority_column_label: str, minority_class_label: str) -> None:
        self.model_name = 'Gaussian_SMOTE'
        super().__init__(original_df, minority_column_label, minority_class_label)

    def gaussian_smote(self):
        
        self.ma_num = self.majority_df.shape[0]
        self.mi_num = self.minority_df.shape[0]
        self.df_num =  self.ma_num +  self.mi_num
        
        classIndex='class' 
        minorityLabel=0
        printDebug = True 
        sigma = 0.05
        
        if (self.mi_num - 1) < 5:
            k = self.mi_num - 1
        else:
            k = 5
        
        mi = self.minority_df.to_numpy()
        num_attributes = mi.shape[1]
        syntheticIndex = 0
        syntheticArray = np.empty((self.num_to_synthesize, num_attributes))

        gap = np.empty(num_attributes)
        gaussianRange = np.empty(num_attributes)
        classColNum = self.original_df.columns.get_loc(key=classIndex)

        for i in range(self.num_to_synthesize):
            x = random.choice(mi)
            x = np.reshape(x, (-1, num_attributes)) 
            _, knn = self.findNeighbors(x, mi, k)
            
            neighbor = mi[knn[0, randrange(1, k+1)]]
            
            x = x.flatten()
            diff = neighbor - x
            diff[classColNum] = minorityLabel
            
            for j in range(len(gap)):

                if diff[j] > 0:
                    gap[j] = np.random.uniform(0, diff[j])
                elif diff[j] < 0:
                    gap[j] = np.random.uniform(diff[j], 0)
                else: 
                    gap[j] = 0 
            
            gap[classColNum] = minorityLabel
            for j in range(len(gaussianRange)):
                if j != classColNum:
                    gaussianRange[j] = random.gauss(x[j] + gap[j], sigma)
                else:
                    gaussianRange[j] = minorityLabel 
                    
            syntheticArray[syntheticIndex] = x + np.multiply(diff, gaussianRange) 
            syntheticArray[syntheticIndex][classColNum] = minorityLabel
            syntheticIndex += 1
            synthetic_data = pd.DataFrame(syntheticArray, columns=self.minority_df.columns.values)
            synthetic_data['synthetic_data'] = 1

        return synthetic_data

    def data_generator(self, num_to_synthesize:int=0) -> pd.DataFrame:
        synthetic_df = None
        self.minority_df, self.majority_df, self.num_to_synthesize = super().data_generator(num_to_synthesize)
        synthetic_data = self.gaussian_smote()
        return super().combine_data(synthetic_data)

class Gamma_BoostCC(SMOTEBase):
    def __init__(self, original_df: pd.DataFrame, minority_column_label: str, minority_class_label: str) -> None:
        self.model_name = 'Gamma_BoostCC'
        super().__init__(original_df, minority_column_label, minority_class_label)

    def gamma_boostCC(self):
        
        self.ma_num = self.majority_df.shape[0]
        self.mi_num = self.minority_df.shape[0]
        self.df_num =  self.ma_num +  self.mi_num
        
        newDF = self.original_df.copy()
        
        numIterations = 5
        k = 5
        for iteration in range(numIterations):
            mi = self.minority_df.to_numpy()
            ma = self.majority_df.to_numpy()
            num_attributes = mi.shape[1]
            # syntheticIndex = 0
            syntheticArray = np.empty((self.num_to_synthesize, num_attributes))
            
            Dpos = self.calculateAverageDistance(mi, self.mi_num, mi, num_attributes, k)
            Dneg = self.calculateAverageDistance(mi, self.mi_num, ma, num_attributes, k)
            
            alpha = 0.5
            beta = 0.0125
            
            maxCD = beta * (alpha - 1)
            
            for i in range(self.num_to_synthesize):
                x_index = random.randint(0, self.mi_num - 1)
                x = mi[x_index]
                x = np.reshape(x, (-1, num_attributes))
                
                _, ind = self.findNeighbors(x, mi, k)
                y = random.randint(0, k-1)
                x_prime = mi[ind[0, y]]
                v = x_prime - x
                t = stats.gamma.rvs(beta, alpha, random_state=None)
                cc = self.calculateControlCoefficient(x, k, mi, ma, Dneg, Dpos)
                p = x + cc * (t - maxCD) * v
                syntheticArray[i] = p
                    
                synthetic_data = pd.DataFrame(syntheticArray, columns=self.minority_df.columns.values)
                synthetic_data['synthetic_data'] = 1

            return synthetic_data

    def data_generator(self, num_to_synthesize:int=0) -> pd.DataFrame:
        synthetic_df = None
        self.minority_df, self.majority_df, self.num_to_synthesize = super().data_generator(num_to_synthesize)
        synthetic_data = self.gamma_boostCC()
        return super().combine_data(synthetic_data)
    
class SDD_BoostCC(SMOTEBase):
    def __init__(self, original_df: pd.DataFrame, minority_column_label: str, minority_class_label: str) -> None:
        self.model_name = 'SDD_BoostCC'
        super().__init__(original_df, minority_column_label, minority_class_label)    

    def sdd_boostCC(self):
        
        self.ma_num = self.majority_df.shape[0]
        self.mi_num = self.minority_df.shape[0]
        self.df_num =  self.ma_num +  self.mi_num
        
        if (self.mi_num - 1) < 5:
            k = self.mi_num - 1
        else:
            k = 5
        
        df = self.original_df.to_numpy()    
        mi = self.minority_df.to_numpy()
        ma = self.majority_df.to_numpy()
        num_attributes = self.original_df.shape[1]
        syntheticIndex = 0
        syntheticArray = np.empty((self.num_to_synthesize, num_attributes))
        
        radius = self.calculateRadius(mi, self.mi_num, ma, self.ma_num, num_attributes, mu = 2)
        
        densityList = []
        for i in range(self.mi_num):
            sample = mi[i]
            sample = np.reshape(sample, (-1, num_attributes))
            
            D = self.calculateSampleDensity(sample, radius, mi, ma)

            densityList.append((i, D))
        densityList.sort(key = lambda x: x[1], reverse = True)
        
        Dpos = self.calculateAverageDistance(mi, self.mi_num, mi, num_attributes, k)
        Dneg = self.calculateAverageDistance(mi, self.mi_num, ma, num_attributes, k)

        densityIndex = 0
        for i in range(self.num_to_synthesize):
            if densityIndex >= len(densityList):
                densityIndex = 0
                
            instance = densityList[densityIndex]
            index = instance[0]
            x = mi[index]
            x = np.reshape(x, (-1, num_attributes))
            cc = self.calculateControlCoefficient(x, k, mi, ma, Dneg, Dpos)
            _, knn = self.findNeighbors(x, mi, k)

            y = randrange(1, k+1)
            diff = mi[knn[0, y]] - x
            syntheticArray[syntheticIndex] = x + cc * diff
            syntheticIndex += 1
            densityIndex += 1
            
        synthetic_data = pd.DataFrame(syntheticArray, columns=self.minority_df.columns.values)
        synthetic_data['synthetic_data'] = 1
        return synthetic_data

    def data_generator(self, num_to_synthesize:int=0) -> pd.DataFrame:
        synthetic_df = None
        self.minority_df, self.majority_df, self.num_to_synthesize = super().data_generator(num_to_synthesize)
        synthetic_data  = self.sdd_boostCC()
        return super().combine_data(synthetic_data)

class ANVO(SMOTEBase):
    def __init__(self, original_df: pd.DataFrame, minority_column_label: str, minority_class_label: str) -> None:
        self.model_name = 'ANVO'
        super().__init__(original_df, minority_column_label, minority_class_label)

    def anvo(self):
        kMin = 2
        kMax = 6
        classIndex = 'class'
        minorityLabel = 0
        majorityLabel = 1
        printDebug = True
        x1Attribute = 'grdt'
        x2Attribute = 'tempin'
        epsilon = 0.0001

        self.ma_num = self.majority_df.shape[0]
        self.mi_num = self.minority_df.shape[0]
        self.df_num = self.ma_num + self.mi_num

        if (self.mi_num - 1) < kMax:
            kMax = self.mi_num - 1

        if kMin >= kMax:
            kMin = int(np.ceil(kMax / 2))

        mi = self.minority_df.to_numpy()
        num_attributes = mi.shape[1]
        syntheticIndex = 0
        syntheticArray = np.empty((self.num_to_synthesize, num_attributes))

        averageOfNeighbors = np.zeros(num_attributes)
        classColNum = self.original_df.columns.get_loc(key=classIndex)
        pointDict = {}
        for point in mi:
            pointDict[tuple(point)] = []

        for i in range(self.num_to_synthesize):
            done = True

            for val in pointDict.values():
                if len(val) < kMax - kMin + 1:
                    done = False

            if not done:
                point = random.choice(mi)
                t = tuple(point)

                while len(pointDict[t]) == kMax - kMin + 1:
                    point = random.choice(mi)
                    t = tuple(point)

                point = np.reshape(point, (1, -1))

                k = random.randint(kMin, kMax)
                while k in pointDict[t]:
                    k = random.randint(kMin, kMax)
                pointDict[t].append(k)

                _, neighborIndices = self.findNeighbors(point, mi, k)

                point = point.flatten()
                neighborIndices = neighborIndices[0][1::]

                for j in range(len(neighborIndices)):
                    neighbor = mi[neighborIndices[j]]
                    diff = neighbor - point
                    averageOfNeighbors = np.add(averageOfNeighbors, diff)

                averageOfNeighbors = np.divide(averageOfNeighbors, float(k))

                newPoint = np.add(point, averageOfNeighbors)

                if np.array_equal(averageOfNeighbors, np.zeros(num_attributes)):
                    newPoint = np.add(newPoint, np.full(num_attributes, epsilon))

                syntheticArray[syntheticIndex] = newPoint

                syntheticArray[syntheticIndex][classColNum] = minorityLabel
                averageOfNeighbors = np.zeros(num_attributes)
                syntheticIndex += 1

        if syntheticIndex < self.num_to_synthesize:
            syntheticArray = syntheticArray[0:syntheticIndex]

        synthetic_data = pd.DataFrame(syntheticArray, columns=self.minority_df.columns.values)
        synthetic_data['synthetic_data'] = 1
        return synthetic_data

    def data_generator(self, num_to_synthesize:int=0) -> pd.DataFrame:
        synthetic_df = None
        self.minority_df, self.majority_df, self.num_to_synthesize = super().data_generator(num_to_synthesize)
        synthetic_data = self.anvo()
        return super().combine_data(synthetic_data)
