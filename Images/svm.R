library(e1071)
library(data.table)
library(dplyr)
library(corrplot)
library(caret)

set.seed(2303299)  
train_indices <- createDataPartition(selected_data[[target_var]], p = 0.8, list = FALSE)
train_data <- selected_data[train_indices, ]
test_data <- selected_data[-train_indices, ]
y_train <- train_data[[target_var]]
y_test <- test_data[[target_var]]

# Make sure to use the top 10 most important variables
train_features <- train_data[, top_10_vars, drop = FALSE]
test_features <- test_data[, top_10_vars, drop = FALSE]

# Define the parameter grid
param_grid <- expand.grid(
  cost = c(0.1, 1, 10, 100),
  gamma = c(0.01, 0.1, 1)
)

# Cross-validation function for SVM
cross_validation_svm <- function(data, y, k, param_grid) {
  folds <- createFolds(y, k = k, list = TRUE)
  results <- data.frame(cost = numeric(), gamma = numeric(), mse = numeric(), stringsAsFactors = FALSE)
  
  for(i in 1:nrow(param_grid)) {
    mse_values <- numeric(k)
    for(j in 1:k) {
      cv_train_indices <- unlist(folds[-j])
      cv_test_indices <- unlist(folds[j])
      
      cv_train_data <- data[cv_train_indices, ]
      cv_test_data <- data[cv_test_indices, ]
      cv_y_train <- y[cv_train_indices]
      cv_y_test <- y[cv_test_indices]
      
      model <- svm(x = cv_train_data, y = cv_y_train, 
                   cost = param_grid$cost[i], 
                   gamma = param_grid$gamma[i])
      
      pred <- predict(model, newdata = cv_test_data)
      mse_values[j] <- mean((pred - cv_y_test)^2)
    }
    avg_mse <- mean(mse_values)
    results <- rbind(results, data.frame(cost = param_grid$cost[i], 
                                         gamma = param_grid$gamma[i], 
                                         mse = avg_mse, 
                                         stringsAsFactors = FALSE))
  }
  
  return(results)
}

# Perform 5-fold cross-validation
cv_results <- cross_validation_svm(train_features, y_train, 5, param_grid)
best_params <- cv_results[which.min(cv_results$mse), ]
print(best_params)

# Train final model with the best parameters
final_model <- svm(x = train_features, y = y_train, 
                   cost = best_params$cost, 
                   gamma = best_params$gamma)

# Make predictions on the test set
preds <- predict(final_model, newdata = test_features)

# Calculate MSE and R²
mse <- mean((preds - y_test)^2)
response_variance <- var(y_test)
r2 <- 1 - mse / response_variance

cat("Final model MSE:", mse, "\n")
cat("Response variable variance:", response_variance, "\n")
cat("Final model R²:", r2, "\n")




