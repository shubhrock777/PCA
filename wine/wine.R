# Load the dataset
library(readr)
wine <- read_csv(file.choose())


summary(wine)

# Normalize the data
normalized_data <- scale(wine) # Excluding the nominal column

summary(normalized_data)
attach(wine)
#########PCA
pcaObj <- princomp(wine, cor = TRUE, scores = TRUE, covmat = NULL)

str(pcaObj)
summary(pcaObj)

loadings(pcaObj)

plot(pcaObj) # graph showing importance of principal components 

biplot(pcaObj)

plot(cumsum(pcaObj$sdev * pcaObj$sdev) * 100 / (sum(pcaObj$sdev * pcaObj$sdev)), type = "b")

pcaObj$scores
pcaObj$scores[, 1:8]

# Top 3 pca scores 
final <- cbind(wine[, 1], pcaObj$scores[, 1:8])
View(final)

