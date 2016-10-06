#remove all objects from R
rm(list=ls())

#set current working directory
setwd("G:/Analytics/Edwisor/Edwisor/Case Study/Titanic/")

#load data into R
data = read.csv("Data_Titanic.csv", header = T)

# Load packages
library('ggplot2')
library('ggthemes')
library('scales') 
library('dplyr') 
library('randomForest')
library("outliers")
library("caret")

#Exploratory data Analysis
#understand the data type
str(data)

#Convert into proper data types
data$Name = as.character(data$Name)
data$Cabin = as.character(data$Cabin)
data$Survived = as.factor(data$Survived)
data$Ticket = as.character(data$Ticket)
str(data)

#Look at the block of data
head(data, 10)
tail(data, 10)

#Let us drive some variables (feature engineering)
# Grab title from passenger names
data$Title = gsub('(.*, )|(\\..*)', '', data$Name)

# Show title counts by sex
table(data$Sex, data$Title)

## create few categories
rare_title = c('Lady', 'the Countess','Capt', 'Col', 'Don', 
                'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer')

# Also reassign mlle, ms, and mme accordingly
data$Title[data$Title == 'Mlle'] = 'Miss' 
data$Title[data$Title == 'Ms'] = 'Miss'
data$Title[data$Title == 'Mme'] = 'Mrs' 
data$Title[data$Title %in% rare_title] = 'Rare Title'

# Show title counts by sex again
table(data$Sex, data$Title)

# Create a family size variable including the passenger themselves
data$Fsize = data$SibSp + data$Parch + 1

#Bin family size
data$FsizeD[data$Fsize == 1] = 'single'
data$FsizeD[data$Fsize > 1 & data$Fsize < 4] = 'small'
data$FsizeD[data$Fsize > 4] = 'large'

#extract first letter from string
data$block = substring(data$Cabin, 1, 1)

#replace empty spaces with No cabin
data$block[data$block == ""] = "NoCabin"

#Missing value analysis
apply(data,2, function(x)sum(is.na(x)))

#replace all empty spaces with missing values
data = data.frame(apply(data, 2, function(x) gsub("^$|^ $", NA, x)))

#test missingness
apply(data,2, function(x)sum(is.na(x)))

#store in dataframe
df_missing = data.frame(Variables = colnames(data), 
                        Count = apply(data,2, function(x)sum(is.na(x))))
row.names(df_missing) = NULL

#impute missing value with different methods
#Let us start experiment
#Impute missing value with mean/median
data$Age = as.numeric(data$Age)
# data$Age[is.na(data$Age)] = mean(data$Age, na.rm = TRUE) 
# data$Age[is.na(data$Age)] = median(data$Age, na.rm = TRUE) 

#actual value of data[5,6] = 48, mean = 39.8331, median = 37, KNN = 41.47
#KNN imputation
library(DMwR)
data = knnImputation(data)

#Normalized Data
data$Age = (data$Age - min(data$Age))/(max(data$Age) - min(data$Age))

data$Fare = as.numeric(data$Fare)
data$Fare = (data$Fare - min(data$Fare))/(max(data$Fare) - min(data$Fare))

#Identify the row and remove outlier
outlier_tf = outlier(data$Age, opposite=FALSE)
find_outlier = which(outlier_tf == TRUE)
data = data[-find_outlier, ]

#divide the data into train and test
train = data[sample(nrow(data), 800, replace = F), ]
test = data[!(1:nrow(data)) %in% as.numeric(row.names(train)), ]

# Build the model (note: not all possible variables are used)
rf_model <- randomForest(factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + 
                           Fare + Embarked + Title + 
                           Fsize + block, data = train)

# Show model error
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:2)

# Get importance
importance = importance(rf_model)
varImportance = data.frame(Variables = row.names(importance), 
                Importance = round(importance[ , 'MeanDecreaseGini'], 2))

# Create a rank variable based on importance
rankImportance = varImportance %>%
  mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance), 
       y = Importance, fill = Importance)) +
       geom_bar(stat='identity') + 
       geom_text(aes(x = Variables, y = 0.5, label = Rank),
       hjust=0, vjust=0.55, size = 4, colour = 'red') +
       labs(x = 'Variables') + coord_flip() + theme_few()

## Predict using the test set
prediction <- predict(rf_model, test[,c(3,5:8,10,12:14,16)])
xtab = table(observed = test[,2], predicted = prediction)
confusionMatrix(xtab)




