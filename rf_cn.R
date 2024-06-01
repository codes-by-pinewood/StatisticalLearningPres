
library(data.table)
library(dplyr)
library(corrplot)
library(randomForest)
library(caret)
#data(Boston)

# Read the column name file and extract the column names
names_file <- "communities.names"
column_names <- readLines(names_file)

# # Extract the line starting with @attribute and extract the column name from it
attribute_lines <- grep("^@attribute", column_names, value = TRUE)
column_names <- gsub("^@attribute\\s+([a-zA-Z0-9_]+)\\s+.*$", "\\1", attribute_lines)


file_path <- "communities.data"
data <- fread(file_path, sep = ",", header = FALSE, na.strings = "?")

# Assign column names to data
colnames(data) <- column_names


head(data)


# 
# Calculate the proportion of missing values ​​in each column
null_prop <- colMeans(is.na(data))
print(null_prop)
# Set the missing value ratio threshold, columns exceeding this ratio will be deleted
threshold <- 0.5  


columns_to_remove <- names(null_prop[null_prop > threshold])
print(columns_to_remove)
# Remove columns with missing value ratio exceeding threshold
data <- data[, !(colnames(data) %in% columns_to_remove), with = FALSE]

# Remove rows containing missing values
data_cleaned <- na.omit(data)


numeric_data <- data_cleaned %>% select_if(is.numeric)


head(numeric_data)
summary(numeric_data)



target_var <- "ViolentCrimesPerPop"

# Calculate the correlation between each variable and the target variable
correlations <- cor(numeric_data, use = "complete.obs")
target_correlations <- correlations[, target_var]

# Eliminate the correlation of the target variable itself
target_correlations <- target_correlations[names(target_correlations) != target_var]

# Select the 10 most influential variables
top_10_vars <- names(sort(abs(target_correlations), decreasing = TRUE))[1:10]


top_10_correlations <- target_correlations[top_10_vars]
print(top_10_correlations)



selected_data <- as.data.frame(numeric_data[, c(top_10_vars, target_var), with = FALSE])

# Make sure all columns are numeric
selected_data <- selected_data %>% mutate(across(everything(), as.numeric))




pairs(selected_data, main = "Top 10 Variables and Target Variable")

# Compute and visualize the correlation matrix
top_10_matrix <- cor(selected_data, use = "complete.obs")
print(top_10_matrix)
corrplot(top_10_matrix, method = "circle")



set.seed(2303299)  
train_indices <- createDataPartition(selected_data[[target_var]], p = 0.8, list = FALSE)
train_data <- selected_data[train_indices, ]
test_data <- selected_data[-train_indices, ]
y_train <- train_data[[target_var]]
y_test <- test_data[[target_var]]

# Make sure to use the top 10 most important variables
train_features <- train_data[, top_10_vars, drop = FALSE]
test_features <- test_data[, top_10_vars, drop = FALSE]

# Defining the parameter grid
param_grid <- expand.grid(
  mtry = c(2, 4, 6, 8, 10), 
  ntree = c(100, 200, 500, 1000),
  nodesize = c(1, 5, 10, 20),
  maxnodes = c(50, 100, 200, 400)
)

# cross function
cross_validation_rf <- function(data, y, k, param_grid) {
  folds <- createFolds(y, k = k, list = TRUE)
  results <- data.frame(mtry = integer(), ntree = integer(), nodesize = integer(), 
                        maxnodes = integer(), mse = numeric(), stringsAsFactors = FALSE)
  
  for(i in 1:nrow(param_grid)) {
    mse_values <- numeric(k)
    for(j in 1:k) {
      cv_train_indices <- unlist(folds[-j])
      cv_test_indices <- unlist(folds[j])
      
      cv_train_data <- data[cv_train_indices, ]
      cv_test_data <- data[cv_test_indices, ]
      cv_y_train <- y[cv_train_indices]
      cv_y_test <- y[cv_test_indices]
      
      model <- randomForest(x = cv_train_data, y = cv_y_train, 
                            mtry = param_grid$mtry[i], 
                            ntree = param_grid$ntree[i],
                            nodesize = param_grid$nodesize[i], 
                            maxnodes = param_grid$maxnodes[i])
      
      pred <- predict(model, newdata = cv_test_data)
      mse_values[j] <- mean((pred - cv_y_test)^2)
    }
    avg_mse <- mean(mse_values)
    results <- rbind(results, data.frame(mtry = param_grid$mtry[i], 
                                         ntree = param_grid$ntree[i], 
                                         nodesize = param_grid$nodesize[i], 
                                         maxnodes = param_grid$maxnodes[i], 
                                         mse = avg_mse, 
                                         stringsAsFactors = FALSE))
  }
  
  return(results)
}

# 5folds
cv_results <- cross_validation_rf(train_features, y_train, 5, param_grid)
best_params <- cv_results[which.min(cv_results$mse), ]
print(best_params)

# best pras to train final 
final_model <- randomForest(x = train_features, y = y_train, 
                            mtry = best_params$mtry, 
                            ntree = best_params$ntree,
                            nodesize = best_params$nodesize, 
                            maxnodes = best_params$maxnodes)


preds <- predict(final_model, newdata = test_features)

# MSE R^2
mse <- mean((preds - y_test)^2)
response_variance <- var(y_test)
r2 <- 1 - mse / response_variance

cat("Final model MSE:", mse, "\n")
cat("Response variable variance:", response_variance, "\n")
cat("Final model R²:", r2, "\n")




print(final_model$results)


importance <- varImp(final_model, scale = FALSE)
print(importance)
plot(importance)

