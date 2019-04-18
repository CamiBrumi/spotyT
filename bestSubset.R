library(ISLR)
library(knitr)
library(printr)
library(leaps)
data("Hitters")
attach(Hitters)

Hitters
?str

str(Hitters)
# First of all, we note that the Salary variable is missing for some of the players.
sum(is.na(Hitters$Salary))
nrow(Hitters)
#So before we proceed we will remove them. The na.omit() function removes all of the rows that have
# missing values in any variable.

Hitters=na.omit(Hitters)
sum(is.na(Hitters))

# The regsubsets() function (part of the leaps library) performs best sub- set selection by identifying the 
# best model that contains a given number of predictors, where best is quantified using RSS. The summary() 
# command outputs the best set of variables for each model size.
# library(leaps)
regfit.full = regsubsets(Salary ~ ., data = Hitters, nvmax = 19) # change to the number of vars of out dataset
reg.summary = summary(regfit.full)
summary(regfit.full)
# The summary() function also returns R2, RSS, adjusted R2, Cp, and BIC. We can examine these to try to 
# select the best overall model.
?names
names(reg.summary)
# rsq is the R-squared statistic

# What Is R-squared?
# R-squared is a statistical measure of how close the data are to the fitted regression line. It is also 
# known as the coefficient of determination, or the coefficient of multiple determination for multiple 
# regression.
# 
# The definition of R-squared is fairly straight-forward; it is the percentage of the response variable variation that is explained by a linear model. Or:
#   
#   R-squared = Explained variation / Total variation
# 
# R-squared is always between 0 and 100%:
#   
#   0% indicates that the model explains none of the variability of the response data around its mean.
# 100% indicates that the model explains all the variability of the response data around its mean.
# In general, the higher the R-squared, the better the model fits your data. However, there are important 
# conditions for this guideline that I’ll talk about both in this post and my next post.

reg.summary$rsq
# the rsq increases as expected, if we add more predictors, it is higher

#plot rss
library(ggvis)
rsq <- as.data.frame(reg.summary$rsq)
names(rsq) <- "R2"
rsq %>% 
  ggvis(x=~ c(1:nrow(rsq)), y=~R2 ) %>%
  layer_points(fill = ~ R2 ) %>%
  add_axis("y", title = "R2") %>% 
  add_axis("x", title = "Number of variables")

# Plotting RSS, adjusted R2, Cp, and BIC for all of the models will help us decide which model to select. Lets have all the plots at once to better compare:
  
par(mfrow=c(2,2))
plot(reg.summary$rss ,xlab="Number of Variables ",ylab="RSS",type="l")
plot(reg.summary$adjr2 ,xlab="Number of Variables ", ylab="Adjusted RSq",type="l")
# which.max(reg.summary$adjr2)
points(11,reg.summary$adjr2[11], col="red",cex=2,pch=20)
plot(reg.summary$cp ,xlab="Number of Variables ",ylab="Cp", type='l')
# which.min(reg.summary$cp )
points(10,reg.summary$cp [10],col="red",cex=2,pch=20)
plot(reg.summary$bic ,xlab="Number of Variables ",ylab="BIC",type='l')
# which.min(reg.summary$bic )
points(6,reg.summary$bic [6],col="red",cex=2,pch=20)

# The regsubsets() function has a built-in plot() command which can be used to display the selected 
# variables for the best model with a given number of predictors, ranked according to the BIC, Cp, adjusted 
# R2, or AIC. For example :
plot(regfit.full,scale="rsq")
?regsubsets

# The top row of each plot contains a black square for each variable selected according to the optimal 
# model associated with that statistic. For instance, we see that several models share a BIC close to −150.
# However, the model with the lowest BIC is the six-variable model that contains only AtBat, Hits, Walks, 
# CRBI, DivisionW, and PutOuts.
# 
# We can use the coef() function to see the coefficient estimates associated with this model.

coef(regfit.full ,6)

