library(tidyverse)
library(tidymodels)
library(textrecipes)
library(jsonlite)

train <- fromJSON("train.json", flatten = TRUE) |>
  as_tibble() |>
  mutate(ingredients = map_chr(ingredients, paste, collapse = " "))

test <- fromJSON("test.json", flatten = TRUE) |>
  as_tibble() |>
  mutate(ingredients = map_chr(ingredients, paste, collapse = " "))

# Defining my recipe 
rec <- recipe(cuisine ~ ingredients, data = train) |>
  step_tokenize(ingredients) |>
  step_tokenfilter(ingredients, max_tokens = 6000) |>
  step_tfidf(ingredients)

# Defining my model and workflow
mod <- multinom_reg(
  penalty = tune(),
  mixture = tune()
) |>
  set_engine("glmnet") |>
  set_mode("classification")

wf <- workflow() |>
  add_recipe(rec) |>
  add_model(mod)

# Cross-validation 
folds <- vfold_cv(train, v = 5, strata = cuisine)

grid <- grid_regular(
  penalty(range = c(-4, -1)),
  mixture(range = c(0.5, 1)),
  levels = 5
)

tuned <- tune_grid(
  wf,
  resamples = folds,
  grid = grid,
  metrics = metric_set(accuracy)
)

# Prediction
best_params <- tuned |>
  select_best("accuracy")

final_wf <- wf |>
  finalize_workflow(best_params)

final_fit <- final_wf |>
  fit(train)

submission <- final_fit |>
  predict(test) |>
  bind_cols(test |> select(id)) |>
  rename(cuisine = .pred_class)

write_csv(submission, "submission.csv")
