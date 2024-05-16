# Introduction

# Cancer is significantly less common in children than in adults. However, the incidence of cancer in young people is far from zero. In fact, at least 15,000 children
# and teens between 0 and 19 years of age were diagnosed with cancer in 2022 in the United States alone (Siegel et al. 2022). Fortunately, many childhood cancers
# may be treated with techniques such as hematopoietic stem cell transplant, and more than 85% of children diagnosed with cancer in the US will survive more than
# five years after their diagnosis (Russel et al. 2024). The increasing availability of healthcare data stands poised to provide a heretofore unprecedented ability
# to understand diseases and guide treatment, although leveraging these data is not without its challenges (Sweeney et al. 2023). The use of machine learning approaches
# may be of particular value for organizations seeking to optimize outcomes for transplant patients, as many factors related to the demographics of the donor and recipient,
# donor and recipient human leukocyte antigen (HLA) profiles, and stem cell sources may play a significant role in treatment success.

# As the capstone project for the HarvardX Professional Certificate in Data Science, this report describes the application of machine learning approaches
# to predict transplant patient survival time. In this report, I detail the steps taken to develop a predictive algorithm using a dataset of stem cell transplants
# in child cancer patients. The data describes pediatric patients with a variety of hematologic diseases, including acute lymphoblastic leukemia, acute myelogenous
# leukemia, and chronic myelogenous leukemia, among others. The goal of my project is to investigate the importance of various factors influencing post-transplant
# survival time, and my primary objective was to develop a model to predict survival time with an RMSE \(\leq\) 400 days.
#
# # Methods
# 
# ## Data Download and Preparation

# As a first step, we will import the dataset, which is from Sikora et al. (2020) and is available on the Kaggle website at 
# https://www.kaggle.com/datasets/adamgudys/bone-marrow-transplant-children/data. After ensuring the necessary R packages are installed, we will import the dataset 
# and inspect it.
#--------------------
# load packages #
library(openxlsx)
library(dplyr)
library(lubridate)
library(ggplot2)
library(reshape2)
library(caret)

# data from https://www.kaggle.com/datasets/adamgudys/bone-marrow-transplant-children

dir<-"/Users/jackesch/projects/Bone_Marrow_Transplant_ML/"
setwd(dir)

data <- read.csv("bone_marrow_dataset.csv")

head(data)

# We can begin by inspecting the data for missing values.

# Sum the number of rows with NA values by column
na_counts <- colSums(is.na(data))

# Print the result
print(na_counts)

# We see that some columns have missing data. In particular, extensive_chronic_GvHD is missing data for more than 16% of the rows. Because several of the machine learning
# approaches we will be using cannot handle missing data, we will remove that column from the dataframe, and then all remaining rows with missing data before beginning our
# model development.

# Remove extensive_chronic_GvHD column
data <- data[, !names(data) %in% "extensive_chronic_GvHD"]

# Remove extensive_chronic_GvHD column
data <- data[, !names(data) %in% "extensive_chronic_GvHD"]

# Remove all rows with missing values in any column
data <- data[complete.cases(data), ]

# Round donor age to nearest year
data$donor_age <- round(data$donor_age)

# Round recipient age to nearest year
data$recipient_age <- round(data$recipient_age)

# We can now examine the structure of the cleaned data. We see that it retains data for 164 patients across 37 variables, which include various factors related to 
# the recipient, the donor, and treatment.
str(data)

# Identify character variables
char_vars <- sapply(data, is.character)

# Convert character variables to factors
data[char_vars] <- lapply(data[char_vars], factor)

# Our model will be designed to predict a continuous variable: survival_time. This variable refers to Time of observation (if alive) or time to event (if dead) in days.

# To begin with, we'll examine correlations among the numeric variables

# Select only numeric variables
numeric_data <- select(data, where(is.numeric))

# Compute the correlation matrix
correlation_matrix <- cor(numeric_data, use = "complete.obs")

# Mask the lower triangle of the correlation matrix
correlation_matrix[lower.tri(correlation_matrix, diag = TRUE)] <- NA

# Plot heatmap
ggplot(data = reshape2::melt(correlation_matrix, na.rm = TRUE), aes(Var2, Var1, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) +
  coord_fixed()

# We can now check to see if any of the included variables are perfectly correlated
perfect_correlation_indices <- which(abs(correlation_matrix) == 1, arr.ind = TRUE)

# Print indices and corresponding variables
perfect_correlation_vars <- colnames(correlation_matrix)[perfect_correlation_indices[, 1]]
perfect_correlation_pairs <- rownames(correlation_matrix)[perfect_correlation_indices[, 2]]
perfect_correlation <- data.frame(Variable1 = perfect_correlation_vars, 
                                  Variable2 = perfect_correlation_pairs)

print(perfect_correlation)

# We see that none are, so we can retain all variables for our modeling

# First, we can perform some basic visualizations to explore the data

# Plot survival status by patient sex
ggplot(data, aes(x = factor(survival_status), fill = recipient_gender)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("steelblue", "coral1"))+
  labs(x = "Survival Status", y = "Count", fill = "Sex") +
  scale_x_discrete(labels = c("Survived", "Deceased")) +
  ggtitle("Survival Status by Patient Sex")

# Mean survival time by recipient sex
ggplot(data, aes(x = recipient_gender, y = survival_time, fill = recipient_gender)) +
  geom_boxplot(color = "black") +
  scale_fill_manual(values = c("coral1", "steelblue"), labels = c("Female", "Male"))+
  labs(title = "Mean Survival Time by Recipient Sex",
       x = "Recipient Sex",
       y = "Survival Time (days)")+
  theme_minimal()+
  theme(legend.position = "none")

# Survival status and survival time appear fairly evenly split by patient sex.

# Density plot of survival by HLA Match
# Compatibility of antigens of the main histocompatibility complex of the donor and the recipient of hematopoietic stem cells (10/10, 9/10, 8/10, 7/10)
ggplot(data, aes(x = HLA_match, fill = factor(survival_status))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("darkseagreen1", "cornsilk4"), labels = c("Survived", "Deceased")) +
  labs(x = "HLA Match", y = "Density", fill = "Survival Status") +
  ggtitle("Density Plot of Survival Status by HLA Match")

# Survival does appear to vary by the percent HLA match, indicating this may be a useful predictor variable.

# Density plot of survival by donor age
ggplot(data, aes(x = donor_age, fill = factor(survival_status))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("darkseagreen1", "cornsilk4"), labels = c("Survived", "Deceased"))+
  labs(x = "Donor Age", y = "Density", fill = "Survival Status") +
  ggtitle("Density Plot of Survival Status by Donor Age")

# Survival also varies with the age of the stem cell donor.

# Survival time by patient age
survival_by_patient_age <- data %>%
  group_by(recipient_age) %>%
  summarize(mean_survival = mean(survival_time))

ggplot(survival_by_patient_age, aes(x = recipient_age, y = mean_survival)) +
  geom_point(stat = "identity", fill = "steelblue", color = "black") +
  geom_smooth(method = "loess", se = TRUE, color = "coral1") + 
  labs(title = "Mean Survival by Patient Age",
       x = "Patient Age",
       y = "Mean Survival (days)")+
  theme_minimal() 

# Survival time by donor age
survival_by_donor_age <- data %>%
  group_by(donor_age) %>%
  summarize(mean_survival = mean(survival_time))

ggplot(survival_by_donor_age, aes(x =donor_age, y = mean_survival)) +
  geom_point(stat = "identity", fill = "steelblue", color = "black") +
  geom_smooth(method = "loess", se = TRUE, color = "coral1") + 
  labs(title = "Mean Survival by Donor Age",
       x = "Donor Age",
       y = "Mean Survival (days)")+
  theme_minimal() 

# Survival time does seem to vary with patient age, as well as with donor age. In particular, it seems younger donors lead to better survival.

## Data preparation
# Split the data into training and test sets

# Before developing the model, the dataset must be split into a "test set" and a "train set." This was achieved by partitioning 10% of the 
# data as the test set, and having the remainder housed in the training set. These two sets were created to allow for subsequent model development 
# and testing.

set.seed(1701, sample.kind="Rounding") # if using R 3.6 or later
# set.seed(1) # if using R 3.5 or earlier
test_index <-createDataPartition(y = data$survival_status, times = 1, p = 0.2, list = F)
train <-data[-test_index,]
test <-data[test_index,]

## Modeling

# Before beginning our experimentation with different modeling approaches, we have to define root-mean-square deviation (RMSE). This is also referred 
# to as the loss function, and, as noted above, RMSE is the approach we use for evaluating the accuracy of the predictions our model generates compared 
# to true rating values. The equation for RMSE is:
  
# $$ RMSE = \sqrt{\frac{1}{N} \sum_{i=1} (\hat{y}_{i} - y_{i})^2} $$
#   Where: $N$ is the number of observations, $\hat{y}_{i}$ is the predicted value (in other words, the predicted survival time of the patient), and $y_{i}$ is the true value (in other words, the actual survival time of the patient).

# As a benchmark model, we can first calculate RMSE based on the mean survival time of the train data set. We will call this the "Naive" model.
# The formula for this is as follows:

# Apply naive model
mu_hat <- mean(train$survival_time)
mu_hat

naive_rmse <- RMSE(train$survival_time, mu_hat)
naive_rmse

results_table <-tibble(Model_Type = "Naive RMSE", RMSE = naive_rmse) %>% 
  mutate(RMSE = sprintf("%0.4f", RMSE))
results_table

# Adding HLA Match Effects
# Compatibility of antigens of the main histocompatibility complex of the donor and the recipient of hematopoietic stem cells (10/10, 9/10, 8/10, 7/10)

mu <- mean(train$HLA_match)
b_HLA_match <- train %>%
  group_by(HLA_match) %>%
  summarize(b_HLA_match = mean(survival_time - mu))

qplot(b_HLA_match, data = b_HLA_match, bins = 5, geom = "histogram", fill = I("steelblue"), color = I("black"))+
  labs(title = "Distribution of HLA Match Effects", x = "b_HLA_match", y = "Frequency")+
  theme_minimal() 

predicted_b_HLA_match <- mu + test %>%
  left_join(b_HLA_match, by='HLA_match') %>%
  pull(b_HLA_match)
b_HLA_match_rmse<-RMSE(predicted_b_HLA_match, test$survival_time)
b_HLA_match_rmse

results_table <-tibble(Model_Type = c("Naive RMSE", "HLA_match"),
                       RMSE =  c(naive_rmse,b_HLA_match_rmse)) %>% 
  mutate(RMSE = sprintf("%0.4f", RMSE))
results_table

# Adding Antigen Effect
# In how many antigens there is a difference between the donor and the recipient (0-3)

b_antigen <-train %>% left_join(b_HLA_match, by = "HLA_match") %>% group_by(antigen) %>%
  summarize(b_antigen = mean(survival_time - mu - b_HLA_match))

ggplot(b_antigen, aes(x = b_antigen)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  labs(title = "Distribution of Antigen Effects", x = "b__antigen", y = "Frequency")+
  theme_minimal() 

predicted_b_antigen <- test %>%
  left_join(b_HLA_match, by='HLA_match') %>% 
  left_join(b_antigen, by='antigen')%>%
  mutate(predictions = mu + b_HLA_match + b_antigen) %>% .$predictions

b_antigen_rmse<-RMSE(test$survival_time, predicted_b_antigen)
b_antigen_rmse

results_table <-tibble(Model_Type = c("Naive RMSE", "HLA_match", 
                                      "HLA_match + antigen"),
                       RMSE =  c(naive_rmse, b_HLA_match_rmse, b_antigen_rmse)) %>% 
  mutate(RMSE = sprintf("%0.4f", RMSE))
results_table

# Adding Relapse status
# Disease reoccurred

b_relapse <-train %>% left_join(b_antigen, by = "antigen") %>% 
  left_join(b_HLA_match, by = "HLA_match")%>%
  group_by(relapse) %>%
  summarize(b_relapse = mean(survival_time - mu - b_HLA_match - b_antigen))

ggplot(b_relapse, aes(x = b_relapse)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  labs(title = "Distribution of Relapse Effects", x = "b_relapse", y = "Frequency")+
  theme_minimal() 

predicted_b_relapse <- test %>%
  left_join(b_HLA_match, by='HLA_match') %>% 
  left_join(b_antigen, by='antigen')%>%
  left_join(b_relapse, by='relapse')%>%
  mutate(predictions = mu + b_HLA_match + b_antigen + b_relapse) %>% .$predictions

b_relapse_rmse<-RMSE(test$survival_time, predicted_b_relapse, na.rm = TRUE)
b_relapse_rmse

results_table <-tibble(Model_Type = c("Naive RMSE", "HLA_match", 
                                      "HLA_match + antigen", 
                                      "HLA_match + antigen + relapse"),
                       RMSE =  c(naive_rmse, b_HLA_match_rmse, b_antigen_rmse, b_relapse_rmse)) %>% 
  mutate(RMSE = sprintf("%0.4f", RMSE))
results_table

# Adding time to acute GVHD
# Time to development of acute graft versus host disease stage III or IV

b_time_to_acute_GvHD <-train %>% left_join(b_relapse, by = "relapse") %>%
  left_join(b_antigen, by = "antigen") %>% 
  left_join(b_HLA_match, by = "HLA_match")%>%
  group_by(time_to_acute_GvHD_III_IV) %>%
  summarize(b_time_to_acute_GvHD = mean(survival_time - mu - b_HLA_match - b_antigen - b_relapse))

ggplot(b_time_to_acute_GvHD, aes(x = b_time_to_acute_GvHD)) +
  geom_histogram(bins = 30, fill = "steelblue", color = "black") +
  labs(title = "Distribution of Time to Acute GvHD Effects", x = "b_time_to_acute_GvHD", y = "Frequency")+
  theme_minimal() 

predicted_b_time_to_acute_GvHD <- test %>%
  left_join(b_HLA_match, by='HLA_match') %>% 
  left_join(b_antigen, by='antigen')%>%
  left_join(b_relapse, by='relapse')%>%
  left_join(b_time_to_acute_GvHD, by = "time_to_acute_GvHD_III_IV")%>%
  mutate(predictions = mu + b_HLA_match + b_antigen + b_relapse + b_time_to_acute_GvHD) %>% .$predictions

b_time_to_acute_GvHD_rmse<-RMSE(test$survival_time, predicted_b_time_to_acute_GvHD, na.rm = TRUE)
b_time_to_acute_GvHD_rmse

results_table <-tibble(Model_Type = c("Naive RMSE", "HLA_match", 
                                      "HLA_match + antigen", 
                                      "HLA_match + antigen + relapse",
                                      "HLA_match + antigen + relapse + time to acute GvHD"),
                       RMSE =  c(naive_rmse, b_HLA_match_rmse, b_antigen_rmse, b_relapse_rmse,b_time_to_acute_GvHD_rmse)) %>% 
  mutate(RMSE = sprintf("%0.4f", RMSE))
results_table

# By incorporating these variables as predictors, we've been able to reduce RMSE by 54 compared to the naive model. However, adding single predictor
# variables to the model one at a time in this fashion is time consuming and may involve significant trial and error. As an alternative, implementing
# more sophisticated machine learning approaches such as k-Nearest Neighbor and Random Forest may leverage all of the available data and generate models 
# with improved predictive accuracy and reduced RMSE.

# We will begin by evaluating the k-Nearest Neighbors approach.

# The goal is to identify the best k, meaning the one that minimizes the prediction error RMSE

# Fit the model on the training set
set.seed(1701)
model <- train(
  survival_time ~., data = train, method = "knn",
  trControl = trainControl("cv", number = 10),
  preProcess = c("center","scale"),
  tuneLength = 10
)
# Plot model error RMSE vs different values of k
plot(model)
# Best tuning parameter k that minimize the RMSE
model$bestTune
# Make predictions on the test data
predictions <- model %>% predict(test)
head(predictions)
# Compute the prediction error RMSE
kNN_rmse<-RMSE(predictions, test$survival_time)

results_table <-tibble(Model_Type = c("Naive RMSE", "HLA_match", 
                                      "HLA_match + antigen", 
                                      "HLA_match + antigen + relapse",
                                      "HLA_match + antigen + relapse + time to acute GvHD",
                                      "kNN"),
                       RMSE =  c(naive_rmse, b_HLA_match_rmse, b_antigen_rmse, b_relapse_rmse,b_time_to_acute_GvHD_rmse, kNN_rmse)) %>% 
  mutate(RMSE = sprintf("%0.4f", RMSE))
results_table

# We see that use of the kNN model actually leads to a higher RMSE, and therefore a poorer prediction. There are several reasons this
# may happen. First, kNN models are known to suffer from overfitting when the number of neighbors (k) is very small, as may be the case
# for some of the more noisy predictor variables. Second, kNN considers all features equally when computing distances, which can lead
# to poorer prediction if some variables are not related to the variable that the model is trying to predict. And finally, the limited size
# of this training set (only 131 individuals) may mean that kNN does not have enough data to accurate estimate the underlying distribution.

# Although this model did not perform as well, we can attempt to use a different method: Gradient Boosting Machine, or GBM. This machine
# learning algorithm builds an ensemble of weak learners (typically decision trees) in a sequential manner to improve predictive performance. 
# It's known for its high predictive accuracy and robustness against overfitting. We can apply this model to our data to see if it allows
# us to achieve a better RMSE value.

library(gbm)

# Train the GBM model
gbm_model <- gbm(
  formula = survival_time ~ ., 
  data = train,
  distribution = "gaussian",  # Specify distribution for regression
  n.trees = 100,  # Number of trees in the ensemble
  interaction.depth = 3,  # Maximum depth of each tree
  shrinkage = 0.1,  # Learning rate (also known as shrinkage)
  bag.fraction = 0.5,  # Fraction of observations to use for each tree (bagging)
  cv.folds = 5  # Number of cross-validation folds
)

# Make predictions on the test data
predictions <- predict(gbm_model, newdata = test, n.trees = 100)

# Evaluate the model
GBM_rmse <- RMSE(predictions, test$survival_time)

results_table <-tibble(Model_Type = c("Naive RMSE", "HLA_match", 
                                      "HLA_match + antigen", 
                                      "HLA_match + antigen + relapse",
                                      "HLA_match + antigen + relapse + time to acute GvHD",
                                      "kNN",
                                      "GBM"),
                       RMSE =  c(naive_rmse, b_HLA_match_rmse, b_antigen_rmse, b_relapse_rmse,b_time_to_acute_GvHD_rmse, kNN_rmse, GBM_rmse)) %>% 
  mutate(RMSE = sprintf("%0.4f", RMSE))
results_table


# This model performed much better than kNN, and also better than all of our previous linear models. This decreased RMSE by 302 days compared to the
# naive model, which has already achieved our stated objective of reducing RMSE by six month over the naive approach. However, we can attempt to use 
# one more machine learning approach- random forest - to see if we can further improve upon this. Random forest is an ensemble method that works
# by building multiple decision trees during training and outputting the mean prediction of the individual trees. We can implement this approach
# to see if it improves our predictive accuracy.

# Use the Random Forest Approach
library(randomForest)
model <- randomForest(survival_time ~ ., data = train, ntree = 1000)
predictions <- predict(model, newdata = test)

varImpPlot(model)

plot(test$survival_time, predictions, xlab = "Actual", ylab = "Predicted", main = "Actual vs. Predicted")

# Compute the prediction error RMSE
RF_rmse <- RMSE(predictions, test$survival_time)

results_table <-tibble(Model_Type = c("Naive RMSE", "HLA_match", 
                                      "HLA_match + antigen", 
                                      "HLA_match + antigen + relapse",
                                      "HLA_match + antigen + relapse + time to acute GvHD",
                                      "kNN",
                                      "randomForest Model"),
                       RMSE =  c(naive_rmse, b_HLA_match_rmse, b_antigen_rmse, b_relapse_rmse,b_time_to_acute_GvHD_rmse, kNN_rmse, RF_rmse)) %>% 
  mutate(RMSE = sprintf("%0.4f", RMSE))
results_table

# We can see that the random forest approach resulted in a substantial improvement in our ability to predict survival time. The RMSE of the random forest
# model is more accurate by 341 days, nearly a full year. This is sufficient to achieve our primary objective of achieving an RMSE that is more accurate than
# the naive model by more than six months.

# Conclusion

# Ultimately, the random forest model yielded the lowest RMSE, at a value of 502.2066. Although this achieves the goal set forth for this project, it is likely that 
# this RMSE value could be further improved by expansion of the dataset through the inclusion of more individuals and more predictor variables, which may be achieved
# by ongoing clinical research. In addition, alternative modeling approaches could be leveraged to further improve model predictions, including the use of matrix 
# factorization. Although larger datasets and more complex models may be more computationally intensive, they have the potential to yield valuable insight that will
# help achieve better outcomes for pediatric cancer patients. As the fields of cancer treatment and machine learning continues to advance, it is likely that more and 
# more sophisticated approaches will be developed and allow for much more accurate predictions of patient outcomes, which will in turn greatly improve treatment processes.

# Literature Cited

# Russell, H., Hord, J., Orr, C. J., & Moerdler, S. (2024). Child Health and the Pediatric Hematology-Oncology Workforce: 2020–2040. Pediatrics, 153(Supplement 2).
 
# Siegel, R. L., Miller, K. D., Fuchs, H. E., & Jemal, A. (2022). Cancer statistics, 2022. CA: a cancer journal for clinicians, 72(1).

# Sikora, M., Wróbel, Ł., and Gudyś, A. (2020). Bone marrow transplant: children. UCI Machine Learning Repository. https://doi.org/10.24432/C5NP6Z.

# Sweeney, S. M., Hamadeh, H. K., Abrams, N., Adam, S. J., Brenner, S., Connors, D. E., ... & Srivastava, S. (2023). Challenges to using big data in cancer. Cancer research, 83(8), 1175-1182.










