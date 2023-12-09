#### install and load packages ####
install.packages("tree")
install.packages("randomForest")
install.packages("caret")
install.packages("pastecs")

#load the packages or manually select them under the Packages tab
library(tree)
library(randomForest)
library(caret)
library(pastecs)

#Read my CSV data
data <- read.csv('C:/Users/notso/Downloads/Customer_behaviour_Tourism.csv')
class(data) #the type is dataframe. 

#Data investigation
names(data) 
head(data,5) #first 5 records
tail(data) #last 6 records

summary(data) #attribute values and their ranges
str(data) #attributes' types and sample attribute values

#We have a column that is about whether or not a customer is an adult. 
#We are going to remove this column.
data <- subset(data, select = -c(Adult_flag))

#Now, let's check for null values
sum(is.na(data))

#We can see that of 11,760 rows, 1,169 have null values. 
#We will still be left with plenty of data to work with, if we remove these values
#we will use the listwise approach to removing these values. 
data <- na.omit(data)



#Now, we can see a lot of our data is stored as chr, when it should be factor 
#For the benefit of our model, we need to convert these. 
#Also one needs to be an integer.
data$Taken_product <- as.factor(data$Taken_product)
data$preferred_device <- as.factor(data$preferred_device)
data$yearly_avg_Outstation_checkins <- as.integer(data$yearly_avg_Outstation_checkins)
data$member_in_family  <- as.factor(data$member_in_family)
data$preferred_location_type <- as.factor(data$preferred_location_type)
data$following_company_page <- as.factor(data$following_company_page)
data$working_flag <- as.factor(data$working_flag)



#reduce factor levels for following company page. 
summary(data$following_company_page)

data <- subset(data, data$following_company_page!=0) 
data <- subset(data, data$following_company_page!=1) 

#Last but not least, let's just remove the unnecessary UserID column.
data <- subset(data,select = -UserID)


#Now we have pre-processed our data, and we are ready to build our classification model.
set.seed(650)
train.index <- sample(1:nrow(data), .8*nrow(data)) #we will keep 80% of the cases for training
train.set <- data[train.index,] #80% of the cases used to train the model
test.set <- data[-train.index,] #20% of the cases used to test the predictive accuracy of the model
taken.test <- data$Taken_product[-train.index]


#Train
dtree.trained <- tree(Taken_product~., train.set)
summary(dtree.trained)
dtree.trained 
plot(dtree.trained)
text(dtree.trained, pretty = 0)


#Now let's see the confusion matrix.
dtree.tested <- predict(dtree.trained, test.set, type = 'class')
table(dtree.tested,taken.test)

accuracy.tested <- (1741+78)/2131
accuracy.tested
#the predictive accuracy of the trained model is at 85.36%. 

#CROSS VALIDATE, prune, and test
dtree.cv <- cv.tree(dtree.trained, FUN = prune.misclass)
dtree.cv

#PRUNE
dtree.pruned <- prune.misclass(dtree.trained, best = 4)
summary(dtree.pruned)

#We see that the results remained the same. This is great news and means we can leave it
#Simplified to 4 terminal nodes. 

#Plot the pruned tree.
dtree.pruned
plot(dtree.pruned)
text(dtree.pruned, pretty = 0)

#See the matrix. It looks the same as the previous tree did with 7 nodes, 
dtree.pruned.tested <- predict(dtree.pruned, test.set, type = 'class')
table(dtree.pruned.tested,taken.test)
accuracy.pruned.tested <- (1741+78)/2131
accuracy.pruned.tested

#troubleshooting
colSums(is.na(data))

#RandomForest
dtree.rf <- randomForest(Taken_product~., train.set, na.action=na.roughfix)
#test the predictive accuracy of the model trained by randomForest
dtree.rf.tested <- predict(dtree.rf, test.set, type = 'class')
table(dtree.rf.tested,taken.test)
accuracy.rf.tested <- (1771 +325)/2119
accuracy.rf.tested
#Random forest greatly improved our accuracy- up to 98.9% - this is likely because
#The dataframe was built with highly correlated data. 




