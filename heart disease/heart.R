# Load the dataset
library(readr)
heart <- read_csv(file.choose())


summary(heart)

# Normalize the data
normalized_data <- scale(heart) # Excluding the nominal column

summary(normalized_data)
attach(heart)
#########PCA
pcaObj <- princomp(heart, cor = TRUE, scores = TRUE, covmat = NULL)

str(pcaObj)
summary(pcaObj)

loadings(pcaObj)

plot(pcaObj) # graph showing importance of principal components 

biplot(pcaObj)

plot(cumsum(pcaObj$sdev * pcaObj$sdev) * 100 / (sum(pcaObj$sdev * pcaObj$sdev)), type = "b")

pcaObj$scores
pcaObj$scores[, 1:10]

# Top 3 pca scores 
final <- cbind(heart[, 1], pcaObj$scores[, 1:10])
View(final)

