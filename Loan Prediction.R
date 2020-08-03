rm(list=ls())

library(tidyr)
library(dplyr)

library(ggplot2)
library(corrgram)
library(gridExtra) 

install.packages(c("JGR","Deducer","DeducerExtras"))
library(Deducer)
library(caret)
install.packages("pscl")
library(pscl)

install.packages("iplots")
library(rJava)
library(JavaGD)
library(iplots)

dfrModel <- read.csv("C:/Users/bvrav/OneDrive/Desktop/bank-loan.csv", header=T,nrows=700, stringsAsFactors=F)
head(dfrModel)




dfrGraph <- gather(dfrModel, variable, value, -default)
head(dfrGraph)

#missing values
is.na(dfrModel)


#finding outliers

boxplot(dfrModel)

#income has more outliers


lapply(dfrModel, FUN=summary)


dim(dfrModel)


# Correlation

vctCorr = numeric(0)
for (i in names(dfrModel)){
  cor.result <- cor(as.numeric(dfrModel$default), as.numeric(dfrModel[,i]))
  vctCorr <- c(vctCorr, cor.result)
}
dfrCorr <- vctCorr
names(dfrCorr) <- names(dfrModel)
dfrCorr

# Data For Visualization

dfrGraph <- gather(dfrModel, variable, value, -default)
head(dfrGraph)

# Data Visualization

ggplot(dfrGraph) +
  geom_jitter(aes(value,default, colour=variable)) + 
  facet_wrap(~variable, scales="free_x") +
  labs(title="Relation Of Default With Other Features")


lapply(dfrModel, FUN=summary)

#Observation
#Mean and Median are nearly equal after doing data imputation whicih help to reduce Outliers.
#Data summary is looking good we can continue with logistic model

#Find Best Multi Logistic Model
#Choose the best logistic model by using step().

stpModel=step(glm(data=dfrModel, formula=default~., family=binomial), trace=0, steps=100)
summary(stpModel)

#Make Final Multi Linear Model

# make model
mgmModel <- glm(data=dfrModel, formula=default ~ age + employ + address + debtinc + creddebt, family=binomial(link="logit"))
# print summary
summary(mgmModel)

pR2 = 1 - mgmModel$deviance / mgmModel$null.deviance
pR2

mgmModel_null <- glm(dfrModel$default~1, family = binomial, data = dfrModel)
pR21= 1- logLik(mgmModel)/logLik(mgmModel_null)
pR21

install.packages('rcompanion')
library(rcompanion)
nagelkerke(mgmModel)

pR2(mgmModel)

# Confusion Matrix

prdVal <- predict(mgmModel, type='response')
prdBln <- ifelse(prdVal > 0.5, 1, 0)
cnfmtrx <- table(prd=prdBln, act=dfrModel$default)
confusionMatrix(cnfmtrx)

#Observations
#Accuracy is good which is around 0.8143

#Regression Data

dfrPlot <- mutate(dfrModel, PrdVal=prdVal, POutcome=prdBln)
head(dfrPlot)

#Regression Visulaization

#dfrPlot
ggplot(dfrPlot, aes(x=PrdVal, y=POutcome))  + 
  geom_point(shape=19, colour="blue", fill="blue") +
  geom_smooth(method="gam", formula=y~s(log(x)), se=FALSE) +
  labs(title="Binomial Regression Curve") +
  labs(x="") +
  labs(y="")

#ROC Visulaization

#rocplot(logistic.model,diag=TRUE,pred.prob.labels=FALSE,prob.label.digits=3,AUC=TRUE)
rocplot(mgmModel)

#Test Data


dfrTest <- read.csv("C:/Users/bvrav/OneDrive/Desktop/bank-loan.csv", header=T, stringsAsFactors=F)
dfrTests <- dfrTest[c(701:850), c(1:8)]
head(dfrTests)

#Missing Data

#sum(is.na(dfrModel$Age))
is.na(dfrTests)

#Predict

resVal <- predict(mgmModel, dfrTests, type="response")
prdSur <- ifelse(resVal > 0.5, 1, 0)

#Observations
#Accuracy is very good, it is around 0.77

#Test Data Confusion Matrix

resVal <- predict(mgmModel, dfrTests, type="response")
prdSur <- ifelse(resVal > 0.5, 1, 0)
prdSur <- as.factor(prdSur)
levels(prdSur) <- c("0", "1")
dfrTests <- mutate(dfrTests, Result=resVal, Prd_Outcome=prdSur)
head(dfrTests)

write.csv(dfrTests, "Test_Prediction_Q2.CSV")
#Summary
#There was total data set of 700 records of past bank loan default history of a private bank, It contains their details along with who is defaulted or not.
#Data Set is divided in two part one is Train data for building a model while other one is Test data to test the model.

