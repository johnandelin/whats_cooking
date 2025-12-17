library(tidyverse)
library(tidymodels)
library(textrecipes)
library(jsonlite)


# Load Data
train <- fromJSON("train.json", flatten = TRUE) |>
  as_tibble() |>
  mutate(ingredients = map_chr(ingredients, paste, collapse = " "))

test <- fromJSON("test.json", flatten = TRUE) |>
  as_tibble() |>
  mutate(ingredients = map_chr(ingredients, paste, collapse = " "))


# Train / Validation Split
split <- initial_split(train, prop = 0.8, strata = cuisine)

train_data <- training(split)
valid_data <- testing(split)


# Recipe (TF-IDF)
recipe_spec <- recipe(cuisine ~ ingredients, data = train_data) |>
  step_tokenize(ingredients) |>
  step_tokenfilter(ingredients, max_tokens = 6000) |>
  step_tfidf(ingredients)


# Tunable Model (glmnet)
model_spec <- multinom_reg(
  penalty = tune(),
  mixture = tune()
) |>
  set_engine("glmnet") |>
  set_mode("classification")


# Workflow
wf <- workflow() |>
  add_recipe(recipe_spec) |>
  add_model(model_spec)


# Cross-Validation
folds <- vfold_cv(train_data, v = 5, strata = cuisine)


# Tuning Grid
grid <- grid_regular(
  penalty(range = c(-4, -1)),   # 1e-4 to 1e-1
  mixture(range = c(0.5, 1)),
  levels = c(6, 4)
)


# Tune Model
tuned_results <- tune_grid(
  wf,
  resamples = folds,
  grid = grid,
  metrics = metric_set(accuracy),
  control = control_grid(save_pred = TRUE)
)


# Best Parameters
best_params <- tuned_results |>
  select_best("accuracy")


# Validate Performance
final_wf <- wf |>
  finalize_workflow(best_params)

final_fit <- final_wf |>
  fit(train_data)

validation_metrics <- final_fit |>
  predict(valid_data) |>
  bind_cols(valid_data) |>
  metrics(truth = cuisine, estimate = .pred_class)


# Train Final Model on Full Data
final_recipe <- recipe(cuisine ~ ingredients, data = train) |>
  step_tokenize(ingredients) |>
  step_tokenfilter(ingredients, max_tokens = 6000) |>
  step_tfidf(ingredients)

final_model <- multinom_reg(
  penalty = best_params$penalty,
  mixture = best_params$mixture
) |>
  set_engine("glmnet") |>
  set_mode("classification")

final_wf_full <- workflow() |>
  add_recipe(final_recipe) |>
  add_model(final_model)

final_fit_full <- final_wf_full |>
  fit(train)


# Predict Test + Submission
submission <- final_fit_full |>
  predict(test) |>
  bind_cols(test |> select(id)) |>
  rename(cuisine = .pred_class)

write_csv(submission, "submission.csv")
