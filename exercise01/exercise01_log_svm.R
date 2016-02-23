# Importing functions
library(RSQLite)
library(DBI)
library(sqldf)

library(plyr)

library(glmnet)
library(car)
library(MASS)
library(mice)
library(kernlab)

library(AUC)
library(ggplot2)
library(gridExtra)
library(RColorBrewer)


#######################################
#              FUNCTIONS              #
#######################################
plot.roc = function(pred, label, roc.title) {
  roc <- roc(pred, label)
  auc <- AUC::auc(roc)
  roc.df <- data.frame(roc$fpr, roc$tpr)

  # plot using ggplot2
  roc.plot <- ggplot(roc.df,aes(x=roc.fpr,y=roc.tpr)) +
    geom_point(size = 0.8, alpha = 0.7, color="navyblue") + theme(legend.position='none') +
    geom_abline(linetype='dashed') +
    labs(list(title = paste0("ROC using ",roc.title), x = "False Positive Rate", y = "True Positive Rate")) +
    annotate("text", x = .7, y = .5, label=paste0("AUC = ",round(auc,4)))

  return(roc.plot)
}

#######################################
#    FLATTEN AND EXTRACT RECORDS      #
#######################################

# Create the database connection
con <- dbConnect(RSQLite::SQLite(), dbname='exercise01.sqlite')
dbListTables(con)

# Using SQL, left join each table with the records table, each on the corresponding
# *_id fields. Since all the value columns are named "name", rename these columns to
# the original table name.
flatten.command <- "SELECT r.id, r.age, r.education_num, r.capital_gain, r.capital_loss, 
                      r.hours_week, r.over_50k, c.name as country, e.name as education_level, 
                      m.name as marital_status, o.name AS occupation, ra.name AS race, 
                      re.name AS relationship, s.name AS sex, w.name AS workclass
                    FROM records as r
                    LEFT JOIN countries AS c ON c.id=r.country_id
                    LEFT JOIN education_levels AS e on e.id=r.education_level_id
                    LEFT JOIN marital_statuses AS m ON m.id=r.marital_status_id
                    LEFT JOIN occupations AS o ON o.id=r.occupation_id
                    LEFT JOIN races AS ra ON ra.id=r.race_id
                    LEFT JOIN relationships AS re ON re.id=r.relationship_id
                    LEFT JOIN sexes AS s ON s.id=r.sex_id
                    LEFT JOIN workclasses AS w ON w.id=r.workclass_id"

records.original <- dbGetQuery(con, flatten.command)

write.table(records.original, file = "records_database.csv", row.names=F, col.names=T, sep=",")

records <- read.csv("records_database.csv", header = TRUE, na.strings='?')

#######################################
# EXPLORE, IMPUTE, AND CREATE SETS    #
#######################################

# Verify all joined fields are factors
sapply(records,class)
# View a summary, look for abmormalities
summary(records)

# Using plyr, looking through the counts of the categorical variables
count(records, "country")
count(records, "education_level")
count(records, "marital_status")
count(records, "occupation")
count(records, "race")
count(records, "relationship")
count(records, "sex")
count(records, "workclass")

# It can be easier to look at their counts in a histogram.
plt1 <- ggplot(records, aes(x = country)) + 
  geom_histogram(color = "purple", fill = "white") + labs(title = "Country") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
plt2 <- ggplot(records, aes(x = race)) + 
  geom_histogram(color = "purple", fill = "white") + labs(title = "Race") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
plt3 <- ggplot(records, aes(x = education_level)) + 
  geom_histogram(color = "purple", fill = "white") + labs(title = "Edu Level") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
plt4 <- ggplot(records, aes(x = marital_status)) + 
  geom_histogram(color = "purple", fill = "white") + labs(title = "Marital") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
plt5 <- ggplot(records, aes(x = occupation)) + 
  geom_histogram(color = "purple", fill = "white") + labs(title = "Occupation") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
plt6 <- ggplot(records, aes(x = relationship)) + 
  geom_histogram(color = "purple", fill = "white") + labs(title = "Relations") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
plt7 <- ggplot(records, aes(x = sex)) + 
  geom_histogram(color = "purple", fill = "white") + labs(title = "Sex") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
plt8 <- ggplot(records, aes(x = workclass)) + 
  geom_histogram(color = "purple", fill = "white") + labs(title = "Workclass") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))
grid.arrange(plt1, plt2, plt3, plt4, plt5, plt6, plt7, plt8, ncol=4)

# Looking at the data visually, you can see the overwhelming individuals born in the
# U.S., 2:1 Male:Female ratio, and a large spike for private sector work class.

# Looking at our target variable:
summary(as.factor(records$over_50k))
# We have about 3 times those under $50,000 than those above, an unbalanced
# classification.


# Summarize the numerical data, excluding NAs
data.frame( t(sapply(numerical.records, function(cl) list(mean=mean(cl,na.rm=TRUE),
                                          median=median(cl,na.rm=TRUE),
                                          std.dev=sd(cl,na.rm=TRUE),
                                          min=min(cl,na.rm=TRUE),
                                          max=max(cl,na.rm=TRUE)
                                          )) ))

# The standard deviation for capital gain is extremely high, and the max
# is a large outlier from the median. Shown on a box plot (with race),
# it shows the significant amount of outliers.
ggplot(records, aes(x= race, y = capital_gain)) +
  geom_boxplot(fill = "grey80", colour = "blue") +
  scale_x_discrete() +
  ylab("Capital Gains")
summary(records)

# View the relationships between variables using "pairs"
pairs(records)

# Looking at the output, it was hard to determine any information, except there appeared
# to be a significant amount of data at only a few points for the scatter plot of education
# level and education number of years
plot(records$education_num, records$education_level)
unique(records[c("education_num", "education_level")])
# With only 16 combinations, these appear to provide the same information, and one was
# likely derived from the other. As education level is on an ordinal scale, and education
# number is an continuous variable, I will drop education level from my predictions.

# Better understand the missing variables
md.pattern(records)
#OUTPUT: 811 rows have no country, 10 have no occupation, 2753 have no occupation or
# workclass, 46 are missing all three
# Attempt to impute data

# Dataset is too large to impute, so will impute a random set of 5,000 records
mice.index <- sample(1:nrow(records),5000)
mice.set <- droplevels(records[mice.index,])
records.imputed <- mice(mice.set,m=5,maxit=3,meth='rf',seed=15)
records.imputed$imp

# It appears that mice cannot consistently impute occupation, and workclass is only missing
# within all records missing occupation. I will remove these rows from the training set.
# Since mice appears to be accurate at imputing country, I will impute these values in the
# training set for those of the 811 in the training set.

# for results to be reproducible
set.seed(15)
# Create Training, Validation, and Test data sets, using a 60/20/20 split
totalset <- c(1:nrow(records))
train.index <- sample(totalset, nrow(records)*0.6)
valid.index <- sample(totalset[-train.index], nrow(records)*0.2)
test.index <- totalset[c(-train.index, -valid.index)]

# Verify sample indices sets do not intersect
intersect(train.index,c(valid.index,test.index))
intersect(valid.index,test.index)

# Create sample sets
train.records <- records[train.index,]
valid.records <- records[valid.index,]
test.records <- records[test.index,]

# remove rows with missing occupation from the training set
nrow(train.records)
train.records <- train.records[complete.cases(train.records[,c("occupation","workclass")]),]
nrow(train.records)
md.pattern(train.records)
# Impute country values using a random forest model
mice.set <- droplevels(train.records)
records.imputed <- mice(mice.set,m=5,maxit=3,meth='rf',seed=15)
records.imputed$imp
# OUTPUT: Most records has United States imputed in 4 or 5 data sets, few have 3 of 5,
# and one has Mexico. These values will be imputed
summary(train.records$country)
# Make sure to use quotes on the rowname, because mice used the original row index, not the
# current one
train.records[is.na(train.records$country),"country"] <- 'United-States'
summary(train.records$country)

# Make sure all NAs are taken care of
nrow(train.records)
nrow(train.records[complete.cases(train.records),])


#######################################
#      CREATE PREDICTIVE MODELS       #
#######################################

colnames(valid.records)
# subset out education_level
train.records.subset <- train.records[,-c(1,9)]
train.glm1 <- glm(over_50k~., data=train.records.subset, family = "binomial")
# OUTPUT: glm.fit: fitted probabilities numerically 0 or 1 occurred 
# This means that due to the rarity of certain values, some of the categorical
# coefficients are likely over estimated.
summary(train.glm1)
# This is not clear in the data, so testing if the error can disappear
train.records.subset <- train.records[,-c(1,4,9)]
train.glm2 <- glm(over_50k~., data=train.records.subset, family = "binomial")
# The error is due to captial gains. Graphing the data:
hist1 <- ggplot(train.records, aes(x = capital_gain)) + 
  geom_histogram(color = "orange", fill = "white") + labs(title = "Capital Gains")
hist2 <- ggplot(train.records[train.records$capital_gain > 0,], aes(x = capital_gain)) + 
  geom_histogram(color = "orange", fill = "white") + labs(title = "Gains Excluding Zero")
grid.arrange(hist1, hist2, ncol=2)

# It is apparent that the capital gains skew is the issue. Instead of removing it
# I will switch to cv.glmnet, which uses a cross-validation, and the lasso regularization,
# which will likely penalize capital gains accordingly.

# Before that is done, looking at the summary there is no variable due to high p-values 
# that I would remove, yet race doesn't appear to significantly help the model. Running without
# race:
train.records.subset <- train.records[,-c(1,4,9,12)]
train.glm3 <- glm(over_50k~., data=train.records.subset, family = "binomial")

# The aic doesn't significantly change without race.
train.glm1$aic
train.glm2$aic
train.glm3$aic

# Testing to see if prediction can occur:
valid.records.subset <- valid.records[,-c(1,4,9,12)]
valid.glm3.pred <- predict(train.glm3, newdata=valid.records.subset, type="response")
# Because of so many factors in workclass, our training set didn't have a factor that 
# exists in our validation data. This could be a problem with an additional workclass
# in our test data, so removing workclass as a predictor

train.records.subset <- train.records[,-c(1,4,9,12,15)]
valid.records.subset <- valid.records[,-c(1,4,9,12,15)]
train.glm4 <- glm(over_50k~., data=train.records.subset, family = "binomial")
valid.glm4.pred <- predict(train.glm4, newdata=valid.records.subset, type="response")

# Predicting with the model works. Moving along to glmnet, testing with and without race
train.y <- train.records[,7]

train.records.glmnet1 <- train.records[,-c(1,7,9,15)]
train.matrix1 <- model.matrix(~.,data=train.records.glmnet1)
train.glmnet1 <- cv.glmnet(y=train.y, x=train.matrix1, family = "binomial")

train.records.glmnet4 <- train.records[,-c(1,7,9,12,15)]
train.matrix4 <- model.matrix(~.,data=train.records.glmnet4)
train.glmnet4 <- cv.glmnet(y=train.y, x=train.matrix4, family = "binomial")


#######################################
#           AUC PREDICTIONS           #
#######################################


# One way to compare models will be looking at the AUC values. a negative of glmnet
# is that it will not provide data for rows with NAs.
valid.records.glmnet <- valid.records[complete.cases(valid.records),]

valid.records.glmnet1 <- valid.records.glmnet[,-c(1,7,9,15)]
valid.matrix1 <- model.matrix(~.,valid.records.glmnet1)

valid.records.glmnet4 <- valid.records.glmnet[,-c(1,7,9,12,15)]
valid.matrix4 <- model.matrix(~.,valid.records.glmnet4)

valid.glmnet1.pred <- predict(train.glmnet1, newx=valid.matrix1, type.measure="response")
valid.glmnet4.pred <- predict(train.glmnet4, newx=valid.matrix4, type.measure="response")

valid.glm4.pred <- predict(train.glm4, newdata=valid.records.subset, type="response")

valid.glm.label <- as.factor(valid.records$over_50k)
valid.glmnet.label <- as.factor(valid.records.glmnet$over_50k)


plot.glm4 <- plot.roc(valid.glm4.pred, valid.glm.label, "Logistic Regression")
plot.glmnet1 <- plot.roc(valid.glmnet1.pred, valid.glmnet.label, "Logistic w/ Lasso")

plot.glmnet4 <- plot.roc(valid.glmnet4.pred, valid.glmnet.label, "Logistic, Lasso, No Race")
grid.arrange(plot.glm4, plot.glmnet1, plot.glmnet4, ncol=2)

# The results are quite positive, and the cross-validation with lasso was helpful.

# Next, I will attmpt to predict using Support Vector Machines, using the variables from our
# most successful model above.


train.records.svm <- train.records[,-c(1,9,12,15)]
valid.records.svm <- valid.records[,-c(1,9,12,15)]
train.svm <- ksvm(over_50k~., data = train.records.svm, type="C-svc", kernel="vanilladot",
                  C=100, prob.model=TRUE)
svm.pred <- predict(train.svm, newdata=valid.records.svm, type="probabilities")
valid.svm.pred <- svm.pred[,2]
valid.svm.label <- valid.glmnet.label

plot.svm <- plot.roc(valid.svm.pred, valid.svm.label, "Support Vector Machines")

grid.arrange(plot.glm4, plot.glmnet1, plot.glmnet4, plot.svm, ncol=2)
# ****Keep in mind that the AUC curves for the Lasso and SVM models should also have the 
# gap because of the NAs, and because of this the AUC value is inflated.

# Next, I will attempt to build a couple models in python. In order to keep the same training,
# validation, and test sets, we will export those created in R.
write.table(train.records, file = "records_train.csv", row.names=F, col.names=T, sep=",")
write.table(valid.records, file = "records_valid.csv", row.names=F, col.names=T, sep=",")
write.table(test.records, file = "records_test.csv", row.names=F, col.names=T, sep=",")

######################################
#           POST SCRIPT              #
######################################

# After creating decision trees in python, it appears that race was a larger determiner (see tree1.png).
# Therefore, I decided to graph the relationship between race and the target varaible.
records_chart <- records
records_chart$over_50k <- as.factor(records_chart$over_50k)


ggplot() +
  geom_bar(data=records_chart, aes(x=race, fill=over_50k)) +
  scale_fill_brewer(type="qual") + 
  labs(title = "Individuals Making Over $50K a Year by Race", y="Individuals")
  


