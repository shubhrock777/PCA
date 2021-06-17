import pandas as pd 

heart = pd.read_csv("D:/BLR10AM/Assi/08Dimension reduction/Datasets_PCA/heart disease.csv")



#2.	Work on each feature of the dataset to create a data dictionary as displayed in the below image
#######feature of the dataset to create a data dictionary


data_details =pd.DataFrame({"column name":heart.columns,
                "data types ":heart.dtypes})

###### without type column
heart_1 = heart.iloc[:,[0,3,4,7,9,1,2,5,6,8,10,11,12,13]]


#3.	Data Pre-processing
#3.1 Data Cleaning, Feature Engineering, etc


#checking unique value for each columns
col_uni=heart_1.nunique()

#details of df 
heart.info()
heart.describe()


"""	Exploratory Data Analysis (EDA):
	Summary
	Univariate analysis
	Bivariate analysis """


EDA =pd.DataFrame({"column ": heart.columns,
      "mean": heart.mean(),
      "median":heart.median(),
      "mode":heart.mode(),
      "standard deviation": heart.std(),
      "variance":heart.var(),
      "skewness":heart.skew(),
      "kurtosis":heart.kurt()})
EDA

# covariance for data set 

covariance = heart.cov()
covariance


####### graphical repersentation 

##historgam and scatter plot
import seaborn as sns
sns.pairplot(heart.iloc[:, 0:10])


#boxplot for every columns
heart.columns
heart.nunique()

boxplot = heart.boxplot(column=["age","trestbps","chol","thalach","oldpeak"])

import matplotlib.pyplot as plt

#boxplot for every column
for column in heart:
    plt.figure()
    heart.boxplot([column])


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
heart_norm =norm_func(heart)





#######5.3 Hierarchical  clustring befor PCA 

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(heart_norm, method = "complete", metric = "euclidean")

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
with_4clust = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(heart_norm) 
with_4clust.labels_

cluster4_labels = pd.Series(with_4clust.labels_)

heart_1['Hclust4_bfPCA'] = cluster4_labels # creating a new column and assigning it to new column 






#######5.3 Kmean   clustring befor PCA 
###### scree plot or elbow curve ############
from sklearn.cluster import	KMeans

TWSS = []
k = list(range(2, 6))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(heart_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4  clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(heart_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
heart_1['Kclust4_bfPCA'] = mb # creating a  new column and assigning it to new column 








####5.2	Perform PCA analysis and get the maximum variance between components

from sklearn.decomposition import PCA
import numpy as np
# Considering only numerical data 

pca = PCA(n_components = 14)
pca_values = pca.fit_transform(heart_norm)

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
heart_PCA =  pca_data.iloc[:, 0:8]


#######5.3 Hierarchical  clustring after PCA 

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(heart_PCA, method = "complete", metric = "euclidean")

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
with_4clust = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(heart_norm) 
with_4clust.labels_

cluster4_labels = pd.Series(with_4clust.labels_)

heart_1['Hclust4_afPCA'] = cluster4_labels # creating a new column and assigning it to new column 






#######5.3 Kmean   clustring after PCA 
###### scree plot or elbow curve ############
from sklearn.cluster import	KMeans

TWSS = []
k = list(range(2, 6))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(heart_PCA)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 4  clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 4)
model.fit(heart_PCA)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
heart_1['Kclust4_afPCA'] = mb   # creating a  new column and assigning it to new column 

