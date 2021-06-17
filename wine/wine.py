import pandas as pd 

wine = pd.read_csv("D:/BLR10AM/Assi/08Dimension reduction/Datasets_PCA/wine.csv")



#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary


data_details =pd.DataFrame({"column name":wine.columns,
                "data types ":wine.dtypes})

###### without type column
wine_1 = wine.iloc[:,1:15]


#3.	Data Pre-processing
#3.1 Data Cleaning, Feature Engineering, etc


#checking unique value for each columns
col_uni = wine.nunique()

#details of df 
wine.info()
wine.describe()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """


EDA =pd.DataFrame({"column ": wine.columns,
      "mean": wine.mean(),
      "median":wine.median(),
      "mode":wine.mode(),
      "standard deviation": wine.std(),
      "variance":wine.var(),
      "skewness":wine.skew(),
      "kurtosis":wine.kurt()})
EDA

# covariance for data set 

covariance = wine.cov()
covariance


####### graphical repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(wine.iloc[:, :])


#boxplot for every columns
wine.columns
wine.nunique()
boxplot = wine.boxplot(column=['Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols'
                               ,'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue','Dilution', 'Proline'])

import matplotlib.pyplot as plt

#boxplot for every column
for column in wine:
    plt.figure()
    wine.boxplot([column])


""" 5.	Model Building
5.1	Build the model on the scaled data (try multiple options)
5.2	Perform PCA analysis and get the maximum variance between components
5.3	Perform clustering before and after applying PCA to cross the number of clusters formed.
"""


#5.1	Build the model on the scaled data (try multiple options)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)



# Normalizing the numerical data 
wine_norm =norm_func(wine)





#######5.3 Hierarchical  clustring befor PCA 

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(wine_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 1 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering cheacking 3 clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

########## with 3 cluster 
with_4clust = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(wine_norm) 
with_4clust.labels_

cluster4_labels = pd.Series(with_4clust.labels_)

wine_1['Hclust3_bfPCA'] = cluster4_labels # creating a new column and assigning it to new column 






#######5.3 Kmean   clustring befor PCA 
###### scree plot or elbow curve ############
from sklearn.cluster import	KMeans

TWSS = []
k = list(range(2, 6))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(wine_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3  clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(wine_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
wine_1['Kclust3_bfPCA'] = mb # creating a  new column and assigning it to new column 








####5.2	Perform PCA analysis and get the maximum variance between components

from sklearn.decomposition import PCA
import numpy as np
# Considering only numerical data 

pca = PCA(n_components = 14)
pca_values = pca.fit_transform(wine_norm)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

pca.components_
pca.components_[0]
# Cumulative variance 

var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)

########## creating new data frame with 3 PCAs
wine_PCA =  pca_data.iloc[:, 0:6]


#######5.3 Hierarchical  clustring after PCA 

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(wine_PCA, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 1 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering cheacking 4 clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

########## with 4 cluster 
with_4clust = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(wine_norm) 
with_4clust.labels_

cluster4_labels = pd.Series(with_4clust.labels_)

wine_1['Hclust4_afPCA'] = cluster4_labels # creating a new column and assigning it to new column 






#######5.3 Kmean   clustring after PCA 
###### scree plot or elbow curve ############
from sklearn.cluster import	KMeans

TWSS = []
k = list(range(2, 6))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(wine_PCA)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 3  clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(wine_PCA)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
wine_1['Kclust3_afPCA'] = mb   # creating a  new column and assigning it to new column 

