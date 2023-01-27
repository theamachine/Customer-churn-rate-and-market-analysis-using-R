#Declaring libraries
library(dplyr)       
library(xgboost)     
library(magrittr)    
library(car)         
library(stringr)     
library(ggplot2)     
library(caTools)     
library(corrplot)  
library(reshape2)    
library(party)       
library(partykit)    
library(recipes)  
library(themis)      
library(caret)
library(ROSE) 
library(mice)
library(rpart.plot)  
library(workflows)   
library(ISLR)        
library(finalfit)    
library(rpart)       
library(caret)       
library(mltools)     
library(Matrix)      


#load the dataset:
dataset <- read.csv('Churn Dataset.csv' ,stringsAsFactors = TRUE)
#Part A
# 1.

#Correlation Matrix

data = subset(dataset, select = c("tenure","MonthlyCharges","TotalCharges"))
pairs(data, pch = 19)

correlationMatrix <- cor(data)
print(correlationMatrix)

corrplot(cor(data), method = "square", type = "full", diag = TRUE, tl.col = "red", bg = "white", title = "",                      
         col = NULL,                      
         tl.cex =0.7,
         cl.ratio =0.2)   


#Heat Map

correlationmatrix <- round(x = cor(data), digits = 2)
head(correlationmatrix)

heat_dataset <- as.matrix(data)
heatmap(heat_dataset, Rowv = NA, Colv = NA)

# 2.
anyNA(dataset)

#find the columns with missing values(NA)
missing <- colnames(dataset)[apply(dataset, 2, anyNA) ]
missing
#Remove missing values
dataset_N <- na.omit(dataset)
#Drop "CustomerID"
dataset_N <- dataset_N[,!(names(dataset_N) %in% c("customerID"))]
dataset_N
#check again if any missing values exits
anyNA(dataset_N)
sum(is.na(dataset_N))

# Convert categorical data into numerical ones 
md.pattern(dataset_N, plot = FALSE)

# 3.
#Split the data
set.seed(123)
state <- sample.split(Y = dataset_N$Churn, SplitRatio = 0.8)
training_Set <- subset(x = dataset_N, state == TRUE)
testing_Set <- subset(x = dataset_N, state == FALSE)
dim(training_Set)
dim(testing_Set)
#Decision tree
DecisionTree <- rpart(Churn ~ ., data = training_Set, method = "class")
rpart.plot(DecisionTree)

#Prediction
y_pred <- predict(DecisionTree, newdata = testing_Set , type = "class")

#Confusion Matrix
confmatrix1 <- confusionMatrix(as.factor(testing_Set$Churn), factor(y_pred), 
                               mode = "prec_recall", dnn = c("Actual", "Prediction"))
confmatrix1 
plot_confusionMatrix <- function(cm) {
  plot <- as.data.frame(cm$table)
  plot$Prediction <- factor(plot$Prediction, levels=rev(levels(plot$Prediction)))
  ggplot(plot, aes(Actual , Prediction, fill= Freq)) +
    geom_tile() + geom_text(aes(label=Freq)) +
    scale_fill_gradient(low="white", high="blue") +
    labs(x = "Prediction",y = "Actual") +
    scale_x_discrete(labels=c("Class_1","Class_2")) +
    scale_y_discrete(labels=c("Class_2","Class_1"))
}
plot_confusionMatrix(confmatrix1)

#ROC curve
ROSE::roc.curve(testing_Set$Churn, y_pred)

# 4.
decisionTree_Information <- rpart(Churn ~ ., data = training_Set, method = "class", 
                                 parms = list(split = "information")) 
decisionTree_Information
plotcp(decisionTree_Information)

y_ipred <- predict(decisionTree_Information, newdata = testing_Set , type = "class")

infoCM <- confusionMatrix(as.factor(testing_Set$Churn), factor(y_ipred), 
                          mode = "prec_recall", dnn = c("Actual", "Prediction"))
infoCM 

plot_confusionMatrix(infoCM)
ROSE::roc.curve(testing_Set$Churn, y_ipred) 

#Prune
decisionTree_Prune <- rpart(Churn ~ ., data = training_Set, method = "class", 
                           control = rpart.control(cp = 0.0083, 
                                                   maxdepth = 3,
                                                   minsplit = 2))
rpart.plot(decisionTree_Prune)

prune_pred <- predict(decisionTree_Prune, newdata = testing_Set , type = "class")

CM_prune <- confusionMatrix(as.factor(testing_Set$Churn), factor(prune_pred), mode = "prec_recall",
                            dnn = c("Actual", "Prediction"))
CM_prune  

plot_confusionMatrix(CM_prune)

ROSE::roc.curve(testing_Set$Churn, prune_pred) 

# 5.
dataset_N$Churn = factor(dataset_N$Churn, level = c("Yes", "No"), 
                                     labels = c(0,1))
set.seed(42)
status <- sample.split(Y = dataset_N$Churn, SplitRatio = 0.8)
train <- subset(x = dataset_N, status == TRUE)
test <- subset(x = dataset_N, status == FALSE)

X_train = data.matrix(train[,-20])                  
y_train = train[,20]                               

X_test = data.matrix(test[,-20])                    
y_test = test[,20]                                   

# XGboost
xgboost_train = xgb.DMatrix(data=X_train, label=y_train)
xgboost_test = xgb.DMatrix(data=X_test, label=y_test)

XGBoost_model <- xgboost(data = xgboost_train,  
                    max.depth=3,  
                    nrounds=70) 
summary(XGBoost_model)


xgb_pred_test = predict(XGBoost_model, newdata= X_test)
xgb_pred = as.factor((levels(y_test))[round(xgb_pred_test)])
xgb_pred

xgb_accuracy <- mean(xgb_pred == test$Churn)
print(paste('Accuracy for XGB test is ', xgb_accuracy))
xgb_precision <- posPredValue(xgb_pred, test$Churn, positive="1")
print(paste('precision for XGB test is ',xgb_precision))
xgb_recall <- sensitivity(xgb_pred, test$Churn, positive="1")
print(paste('Recall for XGB test is ',xgb_recall))
xgb_F1 <- (2 * xgb_precision * xgb_recall) / (xgb_precision + xgb_recall)
print(paste('F1-score for XGB test is ',xgb_F1))


xgb_cm = confusionMatrix(y_test, xgb_pred)
print(xgb_cm)  


xgb_cm <- confusionMatrix(factor(xgb_pred), factor(y_test), dnn = c("Prediction", "Reference"))
print(xgb_cm)
plot <- as.data.frame(xgb_cm$table)
plot$Prediction <- factor(plot$Prediction, levels=rev(levels(plot$Prediction)))
ggplot(plot, aes(Prediction,Reference, fill= Freq)) +
  geom_tile() + geom_text(aes(label=Freq)) +
  scale_fill_gradient(low="white", high="blue") +
  labs(x = "Reference",y = "Prediction") +
  scale_x_discrete(labels=c("Class_1","Class_2")) +
  scale_y_discrete(labels=c("Class_1","Class_2"))


ROSE::roc.curve(y_test, xgb_pred_test) 

# 6.
library(keras)
library(magrittr)
library(reticulate)
library(caTools)
library(tensorflow)

set.seed(123)
DNN_split<- sample.split(Y = dataset_N$Churn, SplitRatio = 0.8)
DNN_train <- subset(x = dataset_N, DNN_split == TRUE)
DNN_test <- subset(x = dataset_N, DNN_split == FALSE)

DNN_X_train = data.matrix(DNN_train[,-20])                  
DNN_y_train = DNN_train[,20]                                

DNN_X_test = data.matrix(DNN_test[,-20])                    
DNN_y_test = DNN_test[,20] 

model <- keras_model_sequential() 

x_train_keras <- array(DNN_X_train, dim = c(dim(DNN_X_train)[1], prod(dim(DNN_X_train)[-1]))) 
x_test_keras <- array(DNN_X_test, dim = c(dim(DNN_X_test)[1], prod(dim(DNN_X_test)[-1]))) 

#One hot encoding
y_train_keras<-to_categorical(DNN_y_train,2)
y_test_keras<-to_categorical(DNN_y_test,2)

model %>%
  layer_dense(units = 128, input_shape = 19) %>%
  layer_dropout(rate=0.3)%>%
  layer_activation(activation = 'tanh') %>%
  layer_dense(units = 64)%>%
  layer_activation(activation = 'tanh')%>%
  layer_dropout(rate=0.3)%>%
  layer_dense(units = 2) %>%
  layer_activation(activation = 'sigmoid')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy')
)

#fitting
model %>% fit(x_train_keras, y_train_keras, epochs = 50, batch_size = 128)


#Evaluating model
loss_and_metrics <- model %>% evaluate(x_test_keras, y_test_keras, batch_size = 128)
pred1 <- model %>% predict(x_test_keras) 
pred1
pred1 = predict(model,data.matrix(x_test_keras), type = "response")
pred1 <-as.factor(as.numeric(pred1>0.5))
cm_model1 <- confusionMatrix(as.factor(y_test_keras), factor(pred1),  
                             mode = "prec_recall", dnn = c("Actual", "Prediction"))
cm_model1   
plot_confusionMatrix(cm_model1)
ROSE::roc.curve(y_test_keras, pred1) 

model2 <- keras_model_sequential() 
model2 %>%
  layer_dense(units = 128, input_shape = 19) %>%
  layer_dropout(rate=0.3)%>%
  layer_activation(activation = 'relu') %>%
  layer_dense(units = 64)%>%
  layer_activation(activation = 'relu')%>%
  layer_dropout(rate=0.3)%>%
  layer_dense(units = 2) %>%
  layer_activation(activation = 'sigmoid')

#cross entropy
model2 %>% compile(
  loss = 'categorical_crossentropy', optimizer = 'adam',
  metrics = c('accuracy')
)

#fitting
model2 %>% fit(x_train_keras, y_train_keras, epochs = 50, batch_size = 128)
#Evaluation
loss_and_metrics <- model2 %>% evaluate(x_test_keras, y_test_keras, batch_size = 128)


pred2 = predict(model2,data.matrix(x_test_keras), type = "response")
pred2 <-as.factor(as.numeric(pred2>0.5))


# Confusion Matrix
cm_model2 <- confusionMatrix(as.factor(y_test_keras), factor(pred2),  
                             mode = "prec_recall", dnn = c("Actual", "Prediction"))
cm_model2                                 
plot_confusionMatrix(cm_model2)


ROSE::roc.curve(y_test_keras, pred2) 