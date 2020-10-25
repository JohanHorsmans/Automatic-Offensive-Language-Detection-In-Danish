Automatic Offensive Language Detection In Danish
================
Aske Bredahl & Johan Horsmans

## Introduction:

This is the R-code-tutorial for the bachelor project of Aske Bredahl &
Johan Horsmans (link to assignemnt: \[INSERT LINK\]). In this tutorial
we will train the following models: Support Vector Machines, Naive
Bayes, Logistic Regression, LSTM Neural Network, Bi-directional LSTM
Neural Network and a Convoluted Neural Network. Furthermore, we will
design the ensemble support system described in the assignment. For the
Python-code follow this link: \[INSERT LINK\].

# PART 1: Support Vector Machine, Naive Bayes & Logistic Regression.

## Step 0: Loading packages

``` r
# Loading/installing the required packages:
library(pacman)
p_load(readr, tidyverse,rsample,recipes, textrecipes, parsnip, yardstick,workflows, discrim,kernlab,stringr, tm, ggplot2, GGally, e1071, caret,stopwords, stringi, SnowballC,fastmatch, parsnip, keras, tensorflow)
```

## Step 1: Loading data:

``` r
# Defining function to read the data:
loading_data <- function(path) {
  read_delim(path, "\t", escape_double = FALSE, trim_ws = TRUE)
}

# Loading the training- and testing data:
training <- loading_data("offenseval-da-training-v1.tsv") %>% 
          mutate(Id = id,label = factor(subtask_a),text=tweet) %>% 
          na.omit()
```

    ## Parsed with column specification:
    ## cols(
    ##   id = col_double(),
    ##   tweet = col_character(),
    ##   subtask_a = col_character()
    ## )

    ## Warning: 2 parsing failures.
    ##  row col               expected              actual                            file
    ## 2961  id no trailing characters     [deleted]   NOT 'offenseval-da-training-v1.tsv'
    ## 2961  -- 3 columns              1 columns           'offenseval-da-training-v1.tsv'

``` r
testing<-loading_data("offenseval-da-test-v1.tsv") %>% 
          mutate(Id = id,label = factor(subtask_a),text=tweet) %>% 
          na.omit()
```

    ## Parsed with column specification:
    ## cols(
    ##   id = col_double(),
    ##   tweet = col_character(),
    ##   subtask_a = col_character()
    ## )

## Step 2: Preprocessing:

``` r
# Make text lowercase
training$text<-tolower(training$text) 

# Remove stopwords:
stopwords_regex = paste(stopwords("da",source= "snowball"), collapse = '\\b|\\b')
stopwords_regex = paste0('\\b', stopwords_regex, '\\b')
training$text = stringr::str_replace_all(training$text, stopwords_regex, '')

# Eemove numbers:
training$text <-  removeNumbers(training$text)

# Stem words:
training$text <-  wordStem(training$text, language = "danish")

# remove punctuation
training$text<-removePunctuation(training$text)

#repeat same procedures for test data
testing$text<-tolower(testing$text) 
stopwords_regex = paste(stopwords("da",source= "snowball"), collapse = '\\b|\\b')
stopwords_regex = paste0('\\b', stopwords_regex, '\\b')
testing$text = stringr::str_replace_all(testing$text, stopwords_regex, '')
testing$text <-  removeNumbers(testing$text)
testing$text <-  wordStem(testing$text, language = "danish")
testing$text<-removePunctuation(testing$text)

# Remove original data columns:
training<-training[,4:6]
testing<-testing[,4:6]

#Inspect training data:
head(training)
```

    ## # A tibble: 6 x 3
    ##      Id label text                                                              
    ##   <dbl> <fct> <chr>                                                             
    ## 1  3131 NOT   " tror    dejlig køligt        svært såfremt personen  billedet  ~
    ## 2   711 NOT   "så kommer  nok   investere   ny cykelpumpe så landbetjenten kan ~
    ## 3  2500 OFF   "      ikeaaber   lavet spillet     gør  uspilleligt"             
    ## 4  2678 NOT   " varme emails   enige     sextilbud   indgående opkald lyder   m~
    ## 5   784 NOT   "desværre tyder    amerikanerne  helt ude  kontrol   kan stemme  ~
    ## 6  3191 NOT   "  fordi  danske børn  folkeskolerne     grund både  dårligere   ~

## Step 3: Naive Bayes, Logistic Regression and SVM using tidymodels

We start off creating the text treatment recipe:

``` r
# Create text recipe:
text_recipe_NB <- recipe(label ~ ., data = training) %>% 
  update_role(Id, new_role = "ID") %>% 
  step_tokenize(text, engine = "spacyr", token = "words") %>%
 ## step_stopwords(text) %>% 
  step_lemma(text) %>%
  step_tokenfilter(text, max_tokens = 100) %>%
  step_tfidf(text)

# Create text recipe:
text_recipe_SVM <- recipe(label ~ ., data = training) %>% 
  update_role(Id, new_role = "ID") %>% 
  step_tokenize(text, engine = "spacyr", token = "words") %>%
 ## step_stopwords(text) %>% 
  step_lemma(text) %>%
  step_tokenfilter(text, max_tokens = 100) %>%
  step_tfidf(text)

# Create text recipe:
text_recipe_LOG <- recipe(label ~ ., data = training) %>% 
  update_role(Id, new_role = "ID") %>% 
  step_tokenize(text, engine = "spacyr", token = "words") %>%
 ## step_stopwords(text) %>% 
  step_lemma(text) %>%
  step_tokenfilter(text, max_tokens = 100) %>%
  step_tfidf(text)

#

# NGRAM recipie
#text_recipe <- recipe(label ~ ., data = training) %>% 
 # update_role(Id, new_role = "ID") %>% 
  #step_tokenize(text) %>%
  #step_ngram(text, num_tokens = 5) %>%
 ## step_stopwords(text) %>% 
  ##step_lemma(text) %>%
  #step_tokenfilter(text, max_tokens = 100) %>%
  #step_tfidf(text)

#TRI GRAM:
#rec <- recipe(~ text, data = abc_tibble) %>%
 # step_tokenize(text) %>%
  #step_ngram(text, num_tokens = 3) %>%
  #step_tokenfilter(text) %>%
  #step_tf(text)
```

We now set the model specification (tidymodels framework) to make them
ready for
classification:

``` r
text_model_log_spec <- logistic_reg() %>% set_engine("glm") %>% set_mode("classification")
text_model_NB_spec <- naive_Bayes() %>% set_engine("naivebayes") %>% set_mode("classification")
text_model_svm_spec <- svm_poly("classification") %>% set_engine("kernlab")
```

We combine the model and the text processing recipe using
worksflows:

``` r
text_model_log_wf <- workflows::workflow() %>% add_recipe(text_recipe_LOG) %>% add_model(text_model_log_spec)
text_model_NB_wf <- workflows::workflow() %>% add_recipe(text_recipe_NB) %>% add_model(text_model_NB_spec)
text_model_svm_wf <- workflows::workflow() %>% add_recipe(text_recipe_SVM) %>% add_model(text_model_svm_spec)
```

## Step 4: Fitting:

``` r
#Fit the models on the training data:
fit_log_model <- fit(text_model_log_wf, training)
```

    ## Found 'spacy_condaenv'. spacyr will use this environment

    ## successfully initialized (spaCy Version: 2.3.2, language model: en_core_web_sm)

    ## (python options: type = "condaenv", value = "spacy_condaenv")

    ## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred

``` r
fit_NB_model <- fit(text_model_NB_wf, training)
fit_svm_model <- fit(text_model_svm_wf, training)
```

    ##  Setting default kernel parameters

## Step 5: Predictions:

We make the models predict the classes of the test data:

``` r
predictions_log <- predict(fit_log_model, testing) # Classifications
predictions_log$raw_log <- predict(fit_log_model, testing,type="prob") # Raw probabilities

predictions_NB <- predict(fit_NB_model, testing,type="class") # Classifications
predictions_NB$raw_NB <- predict(fit_NB_model, testing,type="prob") # Raw probabilities

predictions_SVM <- stats::predict(fit_svm_model, testing,type="class") # Classifications
predictions_SVM$raw_svm <- stats::predict(fit_svm_model, testing,type="prob") # Raw probabilities
```

## Step 6: Evaluate

### Logistic Regression:

``` r
bind_cols(testing,predictions_log) %>% accuracy(truth = label, estimate = .pred_class)
```

    ## # A tibble: 1 x 3
    ##   .metric  .estimator .estimate
    ##   <chr>    <chr>          <dbl>
    ## 1 accuracy binary         0.881

``` r
bind_cols(testing,predictions_log) %>% conf_mat(label, .pred_class) 
```

    ##           Truth
    ## Prediction NOT OFF
    ##        NOT 285  36
    ##        OFF   3   5

### Naive Bayes

``` r
bind_cols(testing,predictions_NB) %>% accuracy(truth = label, estimate = .pred_class)
```

    ## # A tibble: 1 x 3
    ##   .metric  .estimator .estimate
    ##   <chr>    <chr>          <dbl>
    ## 1 accuracy binary         0.815

``` r
bind_cols(testing,predictions_NB) %>% conf_mat(label, .pred_class) 
```

    ##           Truth
    ## Prediction NOT OFF
    ##        NOT 265  38
    ##        OFF  23   3

### SVM

``` r
bind_cols(testing,predictions_SVM) %>% accuracy(truth = label, estimate = .pred_class)
```

    ## # A tibble: 1 x 3
    ##   .metric  .estimator .estimate
    ##   <chr>    <chr>          <dbl>
    ## 1 accuracy binary         0.891

``` r
bind_cols(testing,predictions_SVM) %>% conf_mat(label, .pred_class) 
```

    ##           Truth
    ## Prediction NOT OFF
    ##        NOT 288  36
    ##        OFF   0   5

\#Part 2: Neural Networks.

## Step 1: Preprocessing:

``` r
training$label<-as.numeric(training$label)-1
testing$label<-as.numeric(testing$label)-1

# Inspect training data:
head(training)
```

    ## # A tibble: 6 x 3
    ##      Id label text                                                              
    ##   <dbl> <dbl> <chr>                                                             
    ## 1  3131     0 " tror    dejlig køligt        svært såfremt personen  billedet  ~
    ## 2   711     0 "så kommer  nok   investere   ny cykelpumpe så landbetjenten kan ~
    ## 3  2500     1 "      ikeaaber   lavet spillet     gør  uspilleligt"             
    ## 4  2678     0 " varme emails   enige     sextilbud   indgående opkald lyder   m~
    ## 5   784     0 "desværre tyder    amerikanerne  helt ude  kontrol   kan stemme  ~
    ## 6  3191     0 "  fordi  danske børn  folkeskolerne     grund både  dårligere   ~

## Format data for Keras framework:

bla bla bla

``` r
# Train
text <- training$text

max_features <- 1000
tokenizer <- text_tokenizer(num_words = max_features)

# Test
text_test <- testing$text
max_features_test <- 1000
tokenizer_test <- text_tokenizer(num_words = max_features)
```

bla bla bla

``` r
# Train
tokenizer %>% 
  fit_text_tokenizer(text)

# Test
tokenizer_test %>% 
  fit_text_tokenizer(text_test)
```

bla bla bla

``` r
# Train
tokenizer$document_count
```

    ## [1] 2960

``` r
# Test
tokenizer_test$document_count
```

    ## [1] 329

bla bla bla

``` r
# Train
tokenizer$word_index %>%
  head()
```

    ## $`sÃ¥`
    ## [1] 1
    ## 
    ## $kan
    ## [1] 2
    ## 
    ## $bare
    ## [1] 3
    ## 
    ## $godt
    ## [1] 4
    ## 
    ## $lige
    ## [1] 5
    ## 
    ## $ved
    ## [1] 6

``` r
# Test
tokenizer_test$word_index %>%
  head()
```

    ## $`sÃ¥`
    ## [1] 1
    ## 
    ## $kan
    ## [1] 2
    ## 
    ## $user
    ## [1] 3
    ## 
    ## $godt
    ## [1] 4
    ## 
    ## $danmark
    ## [1] 5
    ## 
    ## $bare
    ## [1] 6

bla bla bla

``` r
# Train
text_seqs <- texts_to_sequences(tokenizer, text)
text_seqs %>%
  head()
```

    ## [[1]]
    ## [1]  42 620 181 218 621
    ## 
    ## [[2]]
    ## [1]   1  22  10 254   1   2
    ## 
    ## [[3]]
    ## [1] 132 542  27
    ## 
    ## [[4]]
    ## [1] 480 894 255
    ## 
    ## [[5]]
    ## [1] 219 622  14 256   2 481   1  57
    ## 
    ## [[6]]
    ##  [1]  34  23 140 133 257 895   1  39 310 744

``` r
# Test
text_seqs_test <- texts_to_sequences(tokenizer_test, text_test)
text_seqs_test %>%
  head()
```

    ## [[1]]
    ## [1] 178 179 180 386 387 180 388 178 389
    ## 
    ## [[2]]
    ## [1]   5 390
    ## 
    ## [[3]]
    ##  [1]   7 391 109 392 393 181 394 395 182 396
    ## 
    ## [[4]]
    ##  [1]   2  71 397 110  24  13 398 399  49 400  20 401 179 402  25 403
    ## 
    ## [[5]]
    ## [1] 33
    ## 
    ## [[6]]
    ## [1] 404 405 406 407

## Step 4: Overall Model specifications (revisit):

``` r
maxlen <- 100
batch_size <- 32
embedding_dims <- 50
filters <- 64
kernel_size <- 3
hidden_dims <- 50
epochs <- 5
```

bla bla bla

``` r
# Train
x_train <- text_seqs %>%
  pad_sequences(maxlen = maxlen)
dim(x_train)
```

    ## [1] 2960  100

``` r
# Test
x_test <- text_seqs_test %>%
  pad_sequences(maxlen = maxlen)
dim(x_test)
```

    ## [1] 329 100

bla bla bla

``` r
# Train
y_train <- training$label
length(y_train)
```

    ## [1] 2960

``` r
# Test
y_test <- testing$label
length(y_test)
```

    ## [1] 329

``` r
# Bla bla
y_test<-y_test
y_train<-y_train
```

\#\#LSTM

## Step 1: Define model in Keras

``` r
model_lstm <- keras_model_sequential()
model_lstm %>%
  layer_embedding(input_dim = max_features, output_dim = 128) %>% 
  layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>% 
  layer_dense(units = 1, activation = 'sigmoid')

# Try using different optimizers and different optimizer configs
model_lstm %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy', "Precision", "Recall")
)
```

## Step 2: Fit model:

``` r
history_lstm <- model_lstm %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = 10,
  validation_data = list(x_test, y_test)
)

plot(history_lstm,method= "ggplot2", smooth = TRUE)
```

    ## `geom_smooth()` using formula 'y ~ x'

![](Automatic-Offensive-Language-Detection-In-Danish_files/figure-gfm/unnamed-chunk-18-1.png)<!-- -->

## Step 3: Evaluate:

``` r
neural_probs <- data.frame(predict_LSTM = predict_classes(model_lstm, x_test))
neural_probs$raw_probs_off_LSTM<-predict_proba(model_lstm, x_test)
neural_probs$raw_probs_not_LSTM<-1-neural_probs$raw_probs_off_LSTM
```

## CNN:

## Step 1: Set model specifications

Bla bla

``` r
# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

#2nd run
#max_features = 20000
#maxlen = 100
#embedding_size = 128
```

bla bla bla

``` r
# Convolution
kernel_size = 5
filters = 64
pool_size = 4

#2nd run
#kernel_size = 2
#filters = 32
#pool_size = 4
```

bla bla bla

``` r
# LSTM
lstm_output_size = 70

#2nd run
#lstm_output_size = 70
```

bla bla bla

``` r
# Training
batch_size = 30
epochs = 5

#2nd run
#batch_size = 15
#epochs = 5
```

``` r
# Defining Model ------------------------------------------------------
model_cnn <- keras_model_sequential()

model_cnn %>%
  layer_embedding(max_features, embedding_size, input_length = maxlen) %>%
  layer_dropout(0.25) %>%
  layer_conv_1d(
    filters, 
    kernel_size, 
    padding = "valid",
    activation = "relu",
    strides = 1
  ) %>%
  layer_max_pooling_1d(pool_size) %>%
  layer_lstm(lstm_output_size) %>%
  layer_dense(1) %>%
  layer_activation("sigmoid")

#Set learning rate (default 0.001)
optimizer_adam(
  lr = 0.001)
```

    ## <tensorflow.python.keras.optimizer_v2.adam.Adam>

``` r
model_cnn %>% compile(
  loss = "binary_crossentropy",
  optimizer = "adam",
  metrics = c("accuracy","Precision","Recall")
)
```

## Step 2: Fit model:

``` r
history_cnn <- model_cnn %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = 10,
  validation_data = list(x_test, y_test)
)

plot(history_cnn,method= "ggplot2", smooth = TRUE)
```

    ## `geom_smooth()` using formula 'y ~ x'

![](Automatic-Offensive-Language-Detection-In-Danish_files/figure-gfm/unnamed-chunk-25-1.png)<!-- -->

``` r
neural_probs$predict_CNN<-predict_classes(model_cnn, x_test) #Predict class
neural_probs$raw_probs_off_CNN<-predict_proba(model_cnn, x_test) #Predict raw probabilites (prob of tweet = OFF)
neural_probs$raw_probs_not_CNN<-1-neural_probs$raw_probs_off_CNN
```

## Bi-directional LSTM

## Step 1: Set model specifications:

``` r
# Define maximum number of input features
max_features <- 20000
```

``` r
# Cut texts after this number of words
# (among top max_features most common words)
maxlen <- 100
```

``` r
batch_size <- 32
```

``` r
#Initialize model
model_bilstm <- keras_model_sequential()
model_bilstm %>%
  # Creates dense embedding layer; outputs 3D tensor
  # with shape (batch_size, sequence_length, output_dim)
  layer_embedding(input_dim = max_features, 
                  output_dim = 128, 
                  input_length = maxlen) %>% 
  bidirectional(layer_lstm(units = 64)) %>%
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = 'sigmoid')
```

``` r
# Try using different optimizers and different optimizer configs
model_bilstm %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam',
  metrics = c('accuracy',"Recall","Precision")
)

#Change learning rate when something does not improve
callback_reduce_lr_on_plateau(
  monitor = "",
  factor = 0.1, #How much to reduce LR
  patience = 50, #How many epocs wiht no improvement
  verbose = 0,
  mode = c("auto", "min", "max"),
  min_delta = 1e-04,
  cooldown = 0,
  min_lr = 0
)
```

    ## <tensorflow.python.keras.callbacks.ReduceLROnPlateau>

## Step 2: Fit model:

``` r
# Train model over four epochs
history_bilstm <- model_bilstm %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = 10,
  validation_data = list(x_test, y_test)
)

plot(history_bilstm,method= "ggplot2", smooth = TRUE)
```

    ## `geom_smooth()` using formula 'y ~ x'

![](Automatic-Offensive-Language-Detection-In-Danish_files/figure-gfm/unnamed-chunk-32-1.png)<!-- -->

## Step 3: Evaluate

``` r
neural_probs$predict_BILSTM<-predict_classes(model_bilstm, x_test) #Predict class
neural_probs$raw_probs_off_BILSTM<-predict_proba(model_bilstm, x_test) #Predict raw probabilites (prob of tweet = OFF)
neural_probs$raw_probs_not_BILSTM<-1-neural_probs$raw_probs_off_BILSTM
```

## PART 3: Ensembling

Ensemble averaging pipeline:

We start of by loading the data from our BERT models (see following
markdown for Python script)

``` r
#Load BERT data:
Bert_results<-read_csv("OG_BERT_RESULTS.csv")
```

    ## Parsed with column specification:
    ## cols(
    ##   `0` = col_double(),
    ##   `1` = col_double()
    ## )

``` r
#testing$BERT_not_prob<-Bert_results$`0`
#testing$BERT_off_prob<-Bert_results$`1`

#Create ensemble with raw probs of other models:
Ensemble_probabilities<-predictions_SVM$raw_svm
Ensemble_probabilities$SVM<-predictions_SVM$raw_svm
Ensemble_probabilities<-Ensemble_probabilities[,3]
Ensemble_probabilities$NB<-predictions_NB$raw_NB
Ensemble_probabilities$log<-predictions_log$raw_log
Ensemble_probabilities$cnn_pred_off<-neural_probs$raw_probs_off_CNN
Ensemble_probabilities$cnn_pred_not<-neural_probs$raw_probs_not_CNN
Ensemble_probabilities$lstm_pred_off<-neural_probs$raw_probs_off_LSTM
Ensemble_probabilities$lstm_pred_not<-neural_probs$raw_probs_not_LSTM
Ensemble_probabilities$bilstm_pred_off<-neural_probs$raw_probs_off_BILSTM
Ensemble_probabilities$bilstm_pred_not<-neural_probs$raw_probs_not_BILSTM


#Define empty columns in ensemble probabilites set
Ensemble_probabilities$ensemble_plus_bert_probs_off<-1
Ensemble_probabilities$ensemble_plus_bert_probs_not<-1
Ensemble_probabilities$avg_preds_bert_plus_ensemble<-1
testing$support_system<-1
Ensemble_probabilities$bertpreds<-1 #This is just for inspection
Ensemble_probabilities$bert_off<-Bert_results$`1`
Ensemble_probabilities$bert_not<-Bert_results$`0`


#Calculating average OFF prob and class prediction
#for (i in 1:nrow(Ensemble_probabilities)){
#  Ensemble_probabilities$ensemble_probs_off[i] <- (Ensemble_probabilities$SVM$.pred_OFF[i] + Ensemble_probabilities$NB$.pred_OFF[i] + Ensemble_probabilities$log$.pred_OFF[i]) / 3
#}

#Calculating average NOT prob
#for (i in 1:nrow(Ensemble_probabilities)){
#  testing$ensemble_probs_not[i] <- (Ensemble_probabilities$SVM$.pred_NOT[i] + Ensemble_probabilities$NB$.pred_NOT[i] + Ensemble_probabilities$log$.pred_NOT[i]) / 3
#  testing$avg_preds[i] <- ifelse(testing$ensemble_probs_off[i] > 0.5, 1, 0)
#  }

#Calculating average OFF BERT+Ensemble prob
for (i in 1:nrow(Ensemble_probabilities)){
Ensemble_probabilities$ensemble_plus_bert_probs_off[i] <- (Ensemble_probabilities$SVM$.pred_OFF[i] + Ensemble_probabilities$NB$.pred_OFF[i] + Ensemble_probabilities$log$.pred_OFF[i] + Bert_results$`1`[i] + Ensemble_probabilities$cnn_pred_off[i] + Ensemble_probabilities$lstm_pred_off[i] + Ensemble_probabilities$bilstm_pred_off[i]) / 7
}

#Calculating average NOT BERT+Ensemble prob and making binary classification
for (i in 1:nrow(Ensemble_probabilities)){
Ensemble_probabilities$ensemble_plus_bert_probs_not[i] <- (Ensemble_probabilities$SVM$.pred_NOT[i] + Ensemble_probabilities$NB$.pred_NOT[i] + Ensemble_probabilities$log$.pred_NOT[i] + Bert_results$`0`[i] + Ensemble_probabilities$cnn_pred_not[i] + Ensemble_probabilities$lstm_pred_not[i] + Ensemble_probabilities$bilstm_pred_not[i]) / 7
}

for (i in 1:nrow(Ensemble_probabilities)){
Ensemble_probabilities$avg_preds_bert_plus_ensemble[i] <- ifelse(Ensemble_probabilities$ensemble_plus_bert_probs_off[i] > 0.5, 1, 0)
}


#Calculate binary BERT classification
for (i in 1:nrow(Ensemble_probabilities)){
  Ensemble_probabilities$bertpreds[i] <- ifelse(Ensemble_probabilities$bert_off[i] > 0.5, 1, 0)
}

##ENSEMBLE SUPPORT SYSTEM##

# Making support system integrating "unsure" offensive classifications (by BERT)
for (i in 1:nrow(testing)){
  testing$support_system[i]<-ifelse(Ensemble_probabilities$bert_off[i] > 0.5 & Ensemble_probabilities$bert_off[i] < 0.65, Ensemble_probabilities$avg_preds_bert_plus_ensemble[i], Ensemble_probabilities$bertpreds[i])
}

# Same but for NOT
for (i in 1:nrow(testing)){
  testing$support_system[i]<-ifelse(Ensemble_probabilities$bert_not[i] > 0.5 & Ensemble_probabilities$bert_not[i] < 0.65, Ensemble_probabilities$avg_preds_bert_plus_ensemble[i], Ensemble_probabilities$bertpreds[i])
}
  
# How many changed classifications:
length(Ensemble_probabilities$bertpreds[Ensemble_probabilities$bertpreds==1])
```

    ## [1] 36

``` r
length(testing$support_system[testing$support_system==1])
```

    ## [1] 36
