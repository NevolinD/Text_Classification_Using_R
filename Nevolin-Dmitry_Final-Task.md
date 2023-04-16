**General summary** (duplicates the end of the work): I can say that,
firs of all, the general hypothesis is approved: it is easy to
distinguish between campaign speeches of D. Trump and H. Clinton.
However, in some cases, they are differentiate only based on the
different length of regular words frequencies. This an issue of a texts
type: even if we conclude that these speeches are based not only
motivational phrases, but still, they contain a lot of regular words and
expressions. Also, the sample is a bit limmited: even with a fact that
all texts are long enough, there are only 136 observations, which can
cause such a good classification. However, I hope that it was not the
main reason and texts length levels this problem out.

An interesting finding is that not all topics that are clearly
identified with the use of frequency lists, are supported by important
words extracted from regression model results. For example, in case of
bigrams, we see that H. Clinton also widely used such topics as
militarism, immigrants and foreign policy.

As for the comparison between to analysis (unigrams and bigrams), model
works in pretty much the same way, with comparable metric scores at the
end. I stated that, probably, bigrams can bring more contextual
information in the analysis, especially in work with misclassifications,
and, to some extend, it is true. Therefore, I think that classification
with bigrams ia more appropriate here.

However, still, the most informative part, in my opinion, is always
produced by regular frequency lists and tf-idf score, which are, for
now, my favorite text analysis methods because of their simplicity,
clarity and, at the same time, efficiency.

``` r
library(haven)
library(tidyverse)
library(tidytext)
library(textstem)
library(stopwords)
library(sjmisc)
library(quanteda)
library(caret)
library(glmnet)

speech <- read_csv("Final task_Dataset.csv")
```

There is a wide spread opinion that campaign speeches are purely
motivational in nature. All candidates always use pretty much same
technique of electorate attraction and do not pay much attention on
specific topics (Rowbottom J. 2012). Therefore, there is, in fact, no
significant difference in the content of speeches done by different
candidates, with some rare exceptions. However, during the 2016 US
presidential elections, the situation was a bit different. Donald Trump
and Hilary Clinton operated with various images of America, emphasizing
the values of voters (Gunawan S. 2018).

In order to check, is this difference significant or not, I want to
apply text classification as a machine learning method of automatic
texts categorization. I would like to check, is it possible to correctly
classify speeches of D. Trump and H. Clinton.

In general, the research question is following: **Is there a significant
difference between campaign speeches of D. Trump and H. Clinton that can
be detected by text classification method?**

My main hypothesis is that this difference is significant because of the
great differences in discourses of both candidates.

This research is based on the following datasets, which were taken from
Kaggle: 1.
<https://www.kaggle.com/datasets/browndw/clintontrump-corpus> -
**Clinton/Trump Corpus** - a corpus of transcribed campaign speeches
delivered by Hillary Clinton and Donald Trump in the run up to the 2016
election. 2.
<https://www.kaggle.com/datasets/alandu20/2016-us-presidential-campaign-texts-and-polls> -
2016 U.S. Presidential Campaign Texts and Polls, from which transcripts
of speeches were taken and joined with the fist dataset.

Data was joined and a bit prepared manually: it was cleaned from the
metadata, which was included into transcripts with the use of
\*\*gsub(“\<.\*?\>“,”“, …)** command. Also I created an **id\*\*
variable, which contains a unique number for each speech.

``` r
speech_for_bigrams <- speech #Here I also create an additional original data set, which I will use in the work with bigrams.

head(speech)
```

    ## # A tibble: 6 × 3
    ##      id President text1                                                         
    ##   <dbl> <chr>     <chr>                                                         
    ## 1     1 Clinton   hello thank you whoa thank you all so much thank you  thank y…
    ## 2     2 Clinton   and after the convention was over starting friday morning tim…
    ## 3     3 Clinton   good morning i am so pleased to be here i want to thank you a…
    ## 4     4 Clinton   i am so happy to be back here and to have this chance to talk…
    ## 5     5 Clinton   thank you thank you so much i have to tell you i am thrilled …
    ## 6     6 Clinton   hello  wow hello everyone   thank you thank you all so much i…

As a result, the final dataset contains texts of speeches, information
about to which candidate it belongs and uniqu identificators for each
speech.

``` r
as.data.frame(lapply(speech, class))
```

    ##        id President     text1
    ## 1 numeric character character

“id” is recoded as numeric, “President” and “text1” as character,
because they contain information in string format.

Important to note that the sample is a bit unbalanced, however, it is
good enough for the test classification implementation:

``` r
nrow(speech) #136
```

    ## [1] 136

``` r
speech %>%
  group_by(President) %>%
  summarize(total_count = n())
```

    ## # A tibble: 2 × 2
    ##   President total_count
    ##   <chr>           <int>
    ## 1 Clinton            49
    ## 2 Trump              87

In general, there are 136 speeches, 87 of which belong to D. Trump, and
49 - to H. Clinton.

Another important characteristic is the length of texts:

``` r
speech_length <- str_split(speech$text1, " ")

speech$words_per_speech <- unlist(lapply(speech_length, length))

descr(speech$words_per_speech) %>%
  select(mean, sd, range)
```

    ## 
    ## ## Basic descriptive statistics
    ## 
    ##     mean      sd            range
    ##  4480.88 2084.37 9994 (852-10846)

The mean length of all 136 texts is about 4480 words per speech. There
are different speeches: their length varies from 852 to 10846 words,
however, 852 as a minimum value is enough for the text classification
implementation.

``` r
speech %>%
  arrange(desc(words_per_speech)) %>%
  select(id, President, words_per_speech) %>%
  top_n(5)
```

    ## Selecting by words_per_speech

    ## # A tibble: 5 × 3
    ##      id President words_per_speech
    ##   <dbl> <chr>                <int>
    ## 1    42 Trump                10846
    ## 2    40 Trump                 9820
    ## 3    46 Trump                 9474
    ## 4    44 Trump                 9466
    ## 5    39 Trump                 9308

Interesting, that the longest speeches belong to the D. Trump.

``` r
speech %>%
  filter(President == "Trump") %>%
  summarise(mean = mean(words_per_speech))
```

    ## # A tibble: 1 × 1
    ##    mean
    ##   <dbl>
    ## 1 5164.

The mean length of his speeches is more than 5000 words. As for H.
Clinton:

``` r
speech %>%
  filter(President == "Clinton") %>%
  summarise(mean = mean(words_per_speech))
```

    ## # A tibble: 1 × 1
    ##    mean
    ##   <dbl>
    ## 1 3268.

``` r
speech %>%
  filter(President == "Clinton") %>%
  arrange(desc(words_per_speech)) %>%
  select(id, President, words_per_speech) %>%
  top_n(5)
```

    ## # A tibble: 5 × 3
    ##      id President words_per_speech
    ##   <dbl> <chr>                <int>
    ## 1   135 Clinton               5966
    ## 2     5 Clinton               5947
    ## 3     3 Clinton               5686
    ## 4    13 Clinton               5551
    ## 5   121 Clinton               4817

Her speeches are also long enough: the mean length is more than 3000
words with the maximum length in about 6000 words.

As a conclusion, these texts are good enough for the text classification
algorithm: it will be easy to “train” model based on such long speeches.

Before the actual analysis, it is interesting to check the content of
used speeches. For doing this, I clean texts from all digits and
punctuation signs using **gsub** function:

``` r
speech$text1 <- str_to_lower(speech$text1)

speech$text1 <- gsub("[[:digit:]]", "", speech$text1)

speech$text1 <- gsub("[[:punct:]]", "", speech$text1)
```

Also, I tokenize all speeches by words in order to, first of all, create
frequency lists and then use them to build a text classification model:

``` r
speech_tokens <- speech %>%
  unnest_tokens(word, text1)

head(speech_tokens)
```

    ## # A tibble: 6 × 4
    ##      id President words_per_speech word 
    ##   <dbl> <chr>                <int> <chr>
    ## 1     1 Clinton               1625 hello
    ## 2     1 Clinton               1625 thank
    ## 3     1 Clinton               1625 you  
    ## 4     1 Clinton               1625 whoa 
    ## 5     1 Clinton               1625 thank
    ## 6     1 Clinton               1625 you

Moreover, I lemmatize words. It is necessary, because, for the
comparison, better to have normalized forms of words. I do it using
**textstem** package, because I do not need information about parts of
speech and, therefore, use udpipe package:

``` r
speech_lemma <- speech_tokens %>%
  mutate(lemma = lemmatize_words(word))

head(speech_lemma)
```

    ## # A tibble: 6 × 5
    ##      id President words_per_speech word  lemma
    ##   <dbl> <chr>                <int> <chr> <chr>
    ## 1     1 Clinton               1625 hello hello
    ## 2     1 Clinton               1625 thank thank
    ## 3     1 Clinton               1625 you   you  
    ## 4     1 Clinton               1625 whoa  whoa 
    ## 5     1 Clinton               1625 thank thank
    ## 6     1 Clinton               1625 you   you

I, once again, clean all lemmas, just in order to be sure that they do
not contain some numbers and punctuations. Also, I clean them from all
stop words, which are listed in the English dictionary of **stopwords**
package:

``` r
speech_lemma <- speech_lemma %>%
  filter(!(lemma %in% stopwords("en")),
         !(str_detect(lemma, "[[:digit:]]")),
         !(str_detect(lemma, "[[:punct:]]")))
```

Here I create frequency lists for both candidates:

``` r
speech_lemma %>%
  filter(President == "Trump") %>%
  count(President, lemma) %>%
  arrange(desc(n)) %>%
  top_n(10)
```

    ## Selecting by n

    ## # A tibble: 10 × 3
    ##    President lemma       n
    ##    <chr>     <chr>   <int>
    ##  1 Trump     go       6489
    ##  2 Trump     people   3234
    ##  3 Trump     say      3089
    ##  4 Trump     know     2624
    ##  5 Trump     get      2603
    ##  6 Trump     country  2261
    ##  7 Trump     good     2200
    ##  8 Trump     much     2071
    ##  9 Trump     great    1994
    ## 10 Trump     want     1904

As for D. Trump, for now it is hard to conclude something about
speeches’ consent based on this list. It is full of some general words,
especially verbs. It is because of the motivational side of the
speeches: candidates always try to motivate people to do something, or
they promise some activities.

As for H. Clinton:

``` r
speech_lemma %>%
  filter(President == "Clinton") %>%
  count(President, lemma) %>%
  arrange(desc(n)) %>%
  top_n(10)
```

    ## Selecting by n

    ## # A tibble: 10 × 3
    ##    President lemma      n
    ##    <chr>     <chr>  <int>
    ##  1 Clinton   go      1335
    ##  2 Clinton   get     1125
    ##  3 Clinton   people   978
    ##  4 Clinton   know     931
    ##  5 Clinton   say      877
    ##  6 Clinton   can      868
    ##  7 Clinton   good     823
    ##  8 Clinton   want     812
    ##  9 Clinton   make     808
    ## 10 Clinton   much     793

In general, situation is the same. Moreover, lists of both candidates
are really similar to each other.

To solve the problem of meaningless frequency lists, I will create them
based on the **tf-idf** score. I is useful, because this score not only
represents how frequent a particular term is, but also decreases the
most frequent terms weight and increases the less frequent.

First of all, I count tf-idf for the whole dataset:

``` r
data_tf_idf <- speech_lemma %>%
  count(President, lemma) %>%
  bind_tf_idf(lemma, President, n) %>%
  arrange(desc(tf_idf))
```

For now, I can filter data and check the content of speeches of both
candidates. As for D. Trump:

``` r
data_tf_idf %>%
  filter(President == "Trump") %>%
  select(lemma, tf_idf) %>%
  head(10)
```

    ## # A tibble: 10 × 2
    ##    lemma         tf_idf
    ##    <chr>          <dbl>
    ##  1 renegotiate 0.000339
    ##  2 boom        0.000209
    ##  3 hillarys    0.000200
    ##  4 libya       0.000197
    ##  5 server      0.000181
    ##  6 approve     0.000168
    ##  7 siege       0.000149
    ##  8 bleach      0.000143
    ##  9 hostage     0.000143
    ## 10 paris       0.000143

Now it is possible to recognize some general topics: the majority of
words here, in my opinion, are related to the international policy
issues (renegotiate (as we know, after being selected as a president,
Trump broke off many international agreements) Libya, siege (it can be
used as a description of some army activities and also as a sign of hard
times in general), hostage, Paris (probably about Paris climate
agreement)).

For sure, D. Trump in his speeches appealed to his opponent, therefore,
name of the second candidate is also in this list.

Also, there are some words, which are meaningless without context
(bleach, approve, server, boom).

As for H. Clinton:

``` r
data_tf_idf %>%
  filter(President == "Clinton") %>%
  select(lemma, tf_idf) %>%
  head(10)
```

    ## # A tibble: 10 × 2
    ##    lemma          tf_idf
    ##    <chr>           <dbl>
    ##  1 disability   0.000332
    ##  2 childrens    0.000305
    ##  3 khan         0.000296
    ##  4 womens       0.000296
    ##  5 fabric       0.000260
    ##  6 millionaire  0.000234
    ##  7 iwillvotecom 0.000207
    ##  8 casino       0.000189
    ##  9 tuition      0.000171
    ## 10 educator     0.000162

Here it is also possible to identify some topics. For example, there are
words related to the human rights issues (disability, children, women),
economy problems (fabric, millionaire, casino), education (tuition,
educator).

Also Clinton advertised this elections, that is why “iwillvotecom” is
also in this list.

“Khan” is a surname of a soldier, who died in Iraq in 2004, when he ran
toward a suicide bomber. He used this story as a argument against
Trump\`s foreign policy plans.

In general, we can see, even based on such lists it can be concluded
that during 2016 US president elections candidates paid huge attention
to certain values. **However, was this difference really significant?**
To answer this question, I need to apply text classification method.

As it is said above, I use a **machine learning-based** system. More
precisely, a **logistic regression** system with Ridge regularization.
It allows us to model the conditional probability and, therefore,
classify texts based on the dependent binary variable, which is names of
candidates. In order to do so, I need to create document term matrix,
where are listed all terms and their tf-idf weight for each document. I
will make furthe comparison and, therefore, classification possible:

``` r
speech_dtm <- speech_lemma %>%
  count(id, lemma) %>%
  cast_dfm(id, lemma, n) %>%
  dfm_tfidf()

speech_dtm
```

    ## Document-feature matrix of: 136 documents, 7,870 features (91.01% sparse) and 0 docvars.
    ##     features
    ## docs      able    across addiction administer administration   advance advocate
    ##    1 0.1940197 0.1467672  1.092146   2.133539      0.2527253 0.4707811 1.054358
    ##    2 0.1940197 0.4403015  0          0             0         0         0       
    ##    3 0.3880393 0.5870687  0          0             0.2527253 0         0       
    ##    4 0.1940197 0.4403015  0          0             0.2527253 0.4707811 0       
    ##    5 1.9401966 1.4676717  1.092146   0             0.2527253 1.4123432 1.054358
    ##    6 0.1940197 0.4403015  0          0             0         0         0       
    ##     features
    ## docs       ago     ahead    alarm
    ##    1 0.3370488 0.6412511 1.531479
    ##    2 0.1123496 0.3206256 0       
    ##    3 0.1123496 0         0       
    ##    4 0         0         0       
    ##    5 0.1123496 1.2825022 0       
    ##    6 0.2246992 0.3206256 0       
    ## [ reached max_ndoc ... 130 more documents, reached max_nfeat ... 7,860 more features ]

DTM is has a high score of sparsity, but it will no affect
classification process.

Moreover, for creating a logistic regression model, I need to divide my
sample into two random corpuses: train one, based on which model will
“learn”, and test corpus that contains documents which will be
classified. Train corpus includes 70 % of data (I asked you about 55 %,
but then I found that with 70 % (but no more) I still have something to
analyze), test one - 30 %. Train corpus should be bigger than test one
in order to have better prepared dictionary for further classification.
Important to nat that this sample slit is based on the index slicing.
They are saved as a single object before the division itself:

``` r
speech_ids <- speech_lemma %>%
  distinct(id, President) #saving indexes

set.seed(555)
split <- as.numeric(createDataPartition(y = speech_ids$id, p = 0.7, list = F))

train_dtm <- speech_dtm[split, ]

test_dtm <- speech_dtm[-split, ]

train_dtm
```

    ## Document-feature matrix of: 96 documents, 7,870 features (91.15% sparse) and 0 docvars.
    ##     features
    ## docs      able    across addiction administer administration   advance advocate
    ##    1 0.1940197 0.1467672  1.092146   2.133539      0.2527253 0.4707811 1.054358
    ##    2 0.1940197 0.4403015  0          0             0         0         0       
    ##    4 0.1940197 0.4403015  0          0             0.2527253 0.4707811 0       
    ##    6 0.1940197 0.4403015  0          0             0         0         0       
    ##    7 0.3880393 0.7338359  3.276439   0             0         0.4707811 0       
    ##    8 0.5820590 0.5870687  1.092146   0             0.5054506 0.4707811 3.163073
    ##     features
    ## docs       ago     ahead    alarm
    ##    1 0.3370488 0.6412511 1.531479
    ##    2 0.1123496 0.3206256 0       
    ##    4 0         0         0       
    ##    6 0.2246992 0.3206256 0       
    ##    7 0         0         0       
    ##    8 0.2246992 0.9618767 0       
    ## [ reached max_ndoc ... 90 more documents, reached max_nfeat ... 7,860 more features ]

``` r
test_dtm
```

    ## Document-feature matrix of: 40 documents, 7,870 features (90.65% sparse) and 0 docvars.
    ##     features
    ## docs      able    across addiction administer administration   advance advocate
    ##   3  0.3880393 0.5870687  0                 0      0.2527253 0         0       
    ##   5  1.9401966 1.4676717  1.092146          0      0.2527253 1.4123432 1.054358
    ##   10 0.3880393 0.4403015  0                 0      0.2527253 0.4707811 0       
    ##   18 0.1940197 0.5870687  0                 0      0         0         0       
    ##   19 0.9700983 0.1467672  0                 0      0         0         2.108715
    ##   20 0.1940197 0.5870687  2.184292          0      0         0.9415622 0       
    ##     features
    ## docs       ago     ahead alarm
    ##   3  0.1123496 0             0
    ##   5  0.1123496 1.2825022     0
    ##   10 0.2246992 0.3206256     0
    ##   18 0.2246992 0.3206256     0
    ##   19 0.1123496 1.2825022     0
    ##   20 0.1123496 0.9618767     0
    ## [ reached max_ndoc ... 34 more documents, reached max_nfeat ... 7,860 more features ]

As we can see, train corpus contains contains 96 speeches, a test one -
40.

Also, I need to split indexes into the same corpuses, using already
existed split:

``` r
true_presidents <- as.factor(speech_ids$President)

train_presidents <- true_presidents[split]
test_presidents <- true_presidents[-split]
```

Before running the model, I need to evaluate it by **cross-validation**
-a method for evaluating an analytical model and its behavior on
training data:

``` r
set.seed(556)
lg_model <- cv.glmnet(x = train_dtm, y = train_presidents,
                      alpha = 0, family = "binomial", type.measure = "auc", nfolds = 5,
                      lambda = seq(0.001, 0.1, by = 0.001), standardize = F)
lg_model
```

    ## 
    ## Call:  cv.glmnet(x = train_dtm, y = train_presidents, lambda = seq(0.001,      0.1, by = 0.001), type.measure = "auc", nfolds = 5, alpha = 0,      family = "binomial", standardize = F) 
    ## 
    ## Measure: AUC 
    ## 
    ##     Lambda Index Measure      SE Nonzero
    ## min    0.1     1  0.9868 0.01328    6906
    ## 1se    0.1     1  0.9868 0.01328    6906

**Lambda min** here is the value of lambda-sequence that gives **minimum
mean cross-validated error**, while **lambda 1se** is the value of
lambda-sequence that gives the most regularized model such that the
cross-validated error is within one standard error of the minimum. Min
lambda is equal 0.1. therefore, 0.1 will be used as a lambda value in
the regression model:

``` r
lg_model_best <- glmnet(x = train_dtm, y = train_presidents,
                        alpha = 0, family = "binomial",
                        lambda = lg_model$lambda.min, standardize = F)

lg_model_best
```

    ## 
    ## Call:  glmnet(x = train_dtm, y = train_presidents, family = "binomial",      alpha = 0, lambda = lg_model$lambda.min, standardize = F) 
    ## 
    ##     Df  %Dev Lambda
    ## 1 6906 97.68    0.1

In this model, I use test data and minimal lambda parameter. As a
result, we have the number of nonzero coefficients (Df = 6906), the
percent of deviance explained (%Dev = 97.68, which means that model
works really good) and the chosen value of lambda.

Now I can implenent classification process and measure it:

``` r
lg_pred <- as.factor(predict(lg_model_best, test_dtm, type = "class"))

lg_conf_matrix <- confusionMatrix(data = lg_pred, reference = test_presidents,
                                  positive = "Clinton", mode = "prec_recall")

lg_conf_matrix
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Clinton Trump
    ##    Clinton      12     3
    ##    Trump         1    24
    ##                                           
    ##                Accuracy : 0.9             
    ##                  95% CI : (0.7634, 0.9721)
    ##     No Information Rate : 0.675           
    ##     P-Value [Acc > NIR] : 0.0009239       
    ##                                           
    ##                   Kappa : 0.7808          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.6170751       
    ##                                           
    ##               Precision : 0.8000          
    ##                  Recall : 0.9231          
    ##                      F1 : 0.8571          
    ##              Prevalence : 0.3250          
    ##          Detection Rate : 0.3000          
    ##    Detection Prevalence : 0.3750          
    ##       Balanced Accuracy : 0.9060          
    ##                                           
    ##        'Positive' Class : Clinton         
    ## 

As we can see, model works really good. 12 True positive (because I
identify “Clinton” as positive class) and 24 True negative, which are
correct classifications (also 3 False positive and only 1 false
negative). It means that model correctly classifies about 90 % of
speeches, as it also shown by **accuracy** metric (0.9, calculated as
TP + TN / all speeches in this corpus).

Important to not that **recall** score (TP / (TP + FN), here it equal
0.9231) is also high. It means that model correctly detect about 92 % of
positive cases among all positive instances.

Also, **precision** (TP / (TP + FP)) is equal 0.8, which is also a very
high score. It means that model correctly detect about 80 % of positive
cases among all predicted positive cases. So, the **type I error**
(accept that case is positive, but, in reality, it is negative) is more
frequent in this classification.

As for **F1** (the harmonic mean of precision and recall), it is equal
to 0.86, with means that about 86 % of predictions are correct.

In general, these results are good enough to call this model well
functioning.

Moreover, I can check **words that are the most significant** for each
corpus (Clinton and Trump). In most cases, exactly based on them
speeches were classified correctly. I do this by extracting model
coefficients:

``` r
# important words for positive class (H. Clinton)
coef(lg_model_best) %>%
  as.matrix() %>%
  as.data.frame() %>%
  tibble::rownames_to_column() %>%
  arrange(-s0) %>%
  filter(rowname != "(Intercept)") %>%
  head(10)
```

    ##       rowname         s0
    ## 1     hillary 0.13037470
    ## 2     clinton 0.09811500
    ## 3      border 0.06970631
    ## 4       trade 0.05666698
    ## 5     jackson 0.05637204
    ## 6      bishop 0.05410479
    ## 7  incredible 0.05239548
    ## 8      legion 0.04971211
    ## 9     illegal 0.04686811
    ## 10      donor 0.04591282

``` r
# important words for negative class (D. Trump)
coef(lg_model_best) %>%
  as.matrix() %>%
  as.data.frame() %>%
  tibble::rownames_to_column() %>%
  arrange(s0) %>%
  filter(rowname != "(Intercept)") %>%
  head(10)
```

    ##        rowname          s0
    ## 1         khan -0.07673324
    ## 2      college -0.07396681
    ## 3       insult -0.05848521
    ## 4   convention -0.04748220
    ## 5      machine -0.04565721
    ## 6       excite -0.04432651
    ## 7     scranton -0.04400490
    ## 8          gun -0.04344051
    ## 9       vision -0.04275572
    ## 10 middleclass -0.04197009

As we can see, words that are significant for Clinton\`s speeches are
positive. On the other hand, Trump words coefficients are negative. That
is because H. Clinton was identified as a positive class in the model.

In general, for Clinton\`s speeches, we can observe words that are
related to different topics. First of all, for the model it was easy to
identify her speeches based on her name: “hillary” and “clinton” are the
most significant here, She use them a lot, probably, because of
self-promotion. Also, there are some words related to the foreign policy
(border, trade, legion, nafta (which is a North American Free Trade
Agreemen)). This is kind of interesting finding, because, as it was
observed using frequency lists, D. Trump spoke about international
politics more than Clintom. Moreover, there are words dedicated to the
human rights issue (illegal (probably), donor, africanamerican). This
finding is supported by what was observed with frequencu lists. There
are some surnames of the US politicians (jackson, who was the first
president from democratic party in the US history; bishop (it can be
translated as a clergyman, but I think it is a surname of a republican
politician)). Also, there are some words that are meaningless without
context (incredible, gonna, percent, uh)

As for D. Trump, here is hard to identify any clear topics. For the
model it was easy to classified his texts as Trump\`s by words, some of
which were related to the domestic policy issues, including human rights
and economy (college, gun, middleclass, kid, stake, register (maybe it
is about immigrants)). Also, there are some words that are related to a
particular place or person (khan (a soldier in Iraq), scranton (the city
in Pennsylvania)). All other words are a kind of meaningless without
context.

Because I have some misclassifications, I would be useful to check,
which words in both corpuses (Trump and Clinton) cause this confusion.

Now I explore Clinton\`s speeches that were incorrectly classified as
made by Trump and check main words, which led to this misclassification:

``` r
error_df <- speech_ids[-split, ]

misclassified_Clinton <- error_df[which(lg_pred == "Trump" & test_presidents == "Clinton"), ] %>%
  left_join(speech_lemma, by = c("id", "President"))

top_misclassified_lemmas <- misclassified_Clinton %>%
  count(lemma, sort = T) %>%
  top_n(10)

top_misclassified_lemmas
```

    ## # A tibble: 10 × 2
    ##    lemma       n
    ##    <chr>   <int>
    ##  1 go        115
    ##  2 get        55
    ##  3 people     54
    ##  4 say        53
    ##  5 great      38
    ##  6 know       35
    ##  7 much       31
    ##  8 think      30
    ##  9 country    28
    ## 10 theyre     28

In general, these words are meaningless without context. Probably, Trump
just used them too frequently and they became significant, and, in one
document, these words were also very frequently used by H. Clinton. I
also check model coefficients of these words:

``` r
coef(lg_model_best) %>%
  as.matrix() %>%
  as.data.frame() %>%
  tibble::rownames_to_column() %>%
  filter(rowname %in% top_misclassified_lemmas$lemma) %>%
  arrange(s0)
```

    ##    rowname            s0
    ## 1    think -1.014257e-03
    ## 2      get -2.604428e-06
    ## 3  country  0.000000e+00
    ## 4       go  0.000000e+00
    ## 5     know  0.000000e+00
    ## 6     much  0.000000e+00
    ## 7   people  0.000000e+00
    ## 8    great  2.067839e-03
    ## 9      say  2.589599e-03
    ## 10  theyre  4.175444e-02

As we see, just two of them really belong to Trump discourse: “think”
and “get”, because of their negative coefficients. Values of other words
are equal or bigger than 0 and, therefore, still more related to the
Clinton\`s discourse, but, maybe, were also frequently used by Trump in
some speeches (which is obvious, because all are just regular terms).

In general, it can be concluded that it was easy for a model to tag
speeches as Clinton\`s. All errors were produced just because of regular
words frequencies.

As for Trump speeches that were classified as Clinton\`s:

``` r
error_df <- speech_ids[-split, ]

misclassified_Trump <- error_df[which(lg_pred == "Clinton" & test_presidents == "Trump"), ] %>%
  left_join(speech_lemma, by = c("id", "President"))

top_misclassified_lemmas <- misclassified_Trump %>%
  count(lemma, sort = T) %>%
  head(10)

top_misclassified_lemmas
```

    ## # A tibble: 10 × 2
    ##    lemma       n
    ##    <chr>   <int>
    ##  1 people     65
    ##  2 much       58
    ##  3 country    54
    ##  4 say        52
    ##  5 can        44
    ##  6 make       40
    ##  7 go         39
    ##  8 know       39
    ##  9 story      36
    ## 10 see        34

In general, the situation is kind of the same as it was in previous
misclassification analysis. Probably, Clinton just used them too
frequently and they became significant, and, in one document, these
words were also very frequently used by D. Trump. I also check model
coefficients of these words:

``` r
coef(lg_model_best) %>%
  as.matrix() %>%
  as.data.frame() %>%
  tibble::rownames_to_column() %>%
  filter(rowname %in% top_misclassified_lemmas$lemma) %>%
  arrange(desc(s0))
```

    ##    rowname            s0
    ## 1      say  0.0025895988
    ## 2      see  0.0011683731
    ## 3      can  0.0000000000
    ## 4  country  0.0000000000
    ## 5       go  0.0000000000
    ## 6     know  0.0000000000
    ## 7     much  0.0000000000
    ## 8   people  0.0000000000
    ## 9     make -0.0001485283
    ## 10   story -0.0058750637

The situation is the same: only two words really belong to Trump
discourse: “say” and “see”, because of their positive coefficients.
Values of other words are equal or less than 0 and, therefore, still
more related to the Trump\`s discourse, but, maybe, were also frequently
used by Clinton in some speeches (which is obvious, because all are just
regular terms).

As a general conclusion, with this sample, classification logistic
regression model works brilliant. There are not so many False negative
and False positive, and all of them were classified so just based on the
different frequencies of regular words in both discourses. The main
reason for it, probably, is that speeches of candidates are too
different in case of some meaningful terms and topic, which, actually,
supports above mentioned hypothesis.

However, I guess that we can obtain another result if instead of
unigrams as a base of analysis I will use bigrams. The reason for it is
kind of simple: as we see, there are a lot of regular terms, with are
meaningful only in context, with can be provided by bigrams.

The majority of preprocessing steps for this analysis will be the same
as in case of unigrams and previous steps, therefore, I will, for the
most part, interpret precisely the semantic side of the analysis, rather
than repeat the specifics of all functions and metrics.

Here I use the original dataset that was additionally stored at the
beginning of the work:

``` r
head(speech_for_bigrams)
```

    ## # A tibble: 6 × 3
    ##      id President text1                                                         
    ##   <dbl> <chr>     <chr>                                                         
    ## 1     1 Clinton   hello thank you whoa thank you all so much thank you  thank y…
    ## 2     2 Clinton   and after the convention was over starting friday morning tim…
    ## 3     3 Clinton   good morning i am so pleased to be here i want to thank you a…
    ## 4     4 Clinton   i am so happy to be back here and to have this chance to talk…
    ## 5     5 Clinton   thank you thank you so much i have to tell you i am thrilled …
    ## 6     6 Clinton   hello  wow hello everyone   thank you thank you all so much i…

I also tokenize texts of all 136 speeches, but now, as a token, I use
bigram instead of unigram (word):

``` r
speech_bigrams <- speech_for_bigrams %>%
  unnest_tokens(bigram, text1, token = "ngrams", n = 2)
```

Cleaning for bigrams differs from the same steps with unigrams: here I
need to divide bigrams in a two separate columns and work with each word
separately. Also, I remove all stop words, numeric digits, punctuation
signs, words with 3 or more repeated letter (because these words are
more probable to be data mistakes or a problem of tokenization) and all
words that are just one letter. After text cleaning, I lemmatize these
words using text_stem package, unite them again into bigrams and create
a new variable, which contains the number of times a particular bigram
is included in speeches:

``` r
bigrams_speech <- speech_bigrams %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>% 
  filter(
    !word1 %in% stop_words$word,                 # remove stopwords from both words in bi-gram
    !word2 %in% stop_words$word,
    !str_detect(word1, pattern = "[[:digit:]]"), # removes any words with numeric digits
    !str_detect(word2, pattern = "[[:digit:]]"),
    !str_detect(word1, pattern = "[[:punct:]]"), # removes any remaining punctuations
    !str_detect(word2, pattern = "[[:punct:]]"),
    !str_detect(word1, pattern = "(.)\\1{2,}"),  # removes any words with 3 or more repeated letters
    !str_detect(word2, pattern = "(.)\\1{2,}"),
    !str_detect(word1, pattern = "\\b(.)\\b"),   # removes any remaining single letter words
    !str_detect(word1, pattern = "\\b(.)\\b")
    ) %>%
  mutate(word1 = lemmatize_words(word1)) %>%
  mutate(word2 = lemmatize_words(word2)) %>%
  unite("bigram", c(word1, word2), sep = " ") %>%
  count(bigram, id, President) %>% arrange(id)

head(bigrams_speech)
```

    ## # A tibble: 6 × 4
    ##   bigram                 id President     n
    ##   <chr>               <dbl> <chr>     <int>
    ## 1 advance manufacture     1 Clinton       1
    ## 2 alarm bell              1 Clinton       1
    ## 3 american people         1 Clinton       1
    ## 4 ann holten              1 Clinton       1
    ## 5 barack obama            1 Clinton       2
    ## 6 big investment          1 Clinton       1

It is important to understand that not all bigrams are meaningful for
the analysis. Therefore, their list can be cleaned. To do so, I check
the distribution of bigrams in order to decide, which frequency
threshold is appropriate here:

``` r
table(bigrams_speech$n)
```

    ## 
    ##     1     2     3     4     5     6     7     8     9    10    11    12    13 
    ## 41192  3377   789   270   114    82    40    33    21    10    12    10     7 
    ##    14    15    16    17    18    19    20    28    30    31 
    ##     4     8     6     6     2     3     1     1     1     1

As we can see, the majority of bigrams are occurred just once. I will
remove them and will analyze corpuses (which I also create here,
according to their petition status) based on other bigrams:

``` r
data_speech <- bigrams_speech %>%
  filter(n > 1)
```

In order to create frequency lists of bigrams, I immediately use tf-df
metric:

``` r
bigram_tf_idf <- 
  data_speech %>%
  bind_tf_idf(bigram, President, n)
```

    ## Warning: A value for tf_idf is negative:
    ##  Input should have exactly one row per document-term combination.

``` r
head(bigram_tf_idf)
```

    ## # A tibble: 6 × 7
    ##   bigram              id President     n       tf    idf    tf_idf
    ##   <chr>            <dbl> <chr>     <int>    <dbl>  <dbl>     <dbl>
    ## 1 barack obama         1 Clinton       2 0.000838 -1.50  -0.00126 
    ## 2 donald trump         1 Clinton       4 0.00168  -3.66  -0.00614 
    ## 3 im excite            1 Clinton       3 0.00126  -0.693 -0.000872
    ## 4 trump talk           1 Clinton       2 0.000838  0.693  0.000581
    ## 5 donald trump         2 Clinton       3 0.00126  -3.66  -0.00461 
    ## 6 financial crisis     2 Clinton       2 0.000838  0.693  0.000581

As for Trump\`s discourse:

``` r
bigram_tf_idf %>%
  filter(President == "Trump") %>%
  arrange(desc(tf_idf)) %>%
  select(bigram, tf_idf) %>%
  head(10)
```

    ## # A tibble: 10 × 2
    ##    bigram            tf_idf
    ##    <chr>              <dbl>
    ##  1 national guard  0.000652
    ##  2 rubber hose     0.000522
    ##  3 carpet bomb     0.000456
    ##  4 change maker    0.000391
    ##  5 missile defense 0.000391
    ##  6 voter fraud     0.000391
    ##  7 obama care      0.000391
    ##  8 jeff session    0.000326
    ##  9 gonna pay       0.000326
    ## 10 andrew jackson  0.000326

Here we can easily identify a couple of topics, as it also was in case
of unigrams. For example, there are some words related to the
international politics and military issues (national guard, carpet bomb,
missile defense). Also, there are some names of US politicians (jeff
session - one of the most conservative senators in the US history;
andrew jackson - the fist US president from democratic party). Moreover,
there are some words dedicated to the domestic policy issues (obama
care - an informal term for a federal law intended to improve access to
health insurance (as I know, it was highly criticized by republicans
back in that years (and not only by them)), change maker, voter fraud).
Other word (gonna pay, rubber hose) can be used just as regular
expressions.

In general, there are some clear topics, that were also identified
during analysis with unigrams.

As for H. Clinton\`s speeches:

``` r
bigram_tf_idf %>%
  filter(President == "Clinton") %>%
  arrange(desc(tf_idf)) %>%
  select(bigram, tf_idf) %>%
  head(10)
```

    ## # A tibble: 10 × 2
    ##    bigram               tf_idf
    ##    <chr>                 <dbl>
    ##  1 wake tech           0.00349
    ##  2 national service    0.00261
    ##  3 baptist church      0.00174
    ##  4 director comey      0.00145
    ##  5 foster care         0.00145
    ##  6 hes forget          0.00116
    ##  7 legal service       0.00116
    ##  8 john marshall       0.00116
    ##  9 white supremacist   0.00116
    ## 10 american leadership 0.00116

Here w also e can easily identify a couple of topics. For example, there
are some words, related to the policy problems, including human rights
issues (national service, foster care, legal service, american
leadership). Also, there are some names of US politicians (director
comey - an American lawyer who was the seventh director of the FBI, john
marshall - one of the Founding Fathers). Moreover, Clinton pays
attention on social values of her voters (baptist church, white
supremacist - with high probability, it is a “compliment” for a D. Trump
(: )). Also, there is a name of one US university (wake tech).

In general, all topic are identifiable here, but, in case of unigrams
list for Clinton, it was much more representable, in my opinion.

As it was done before, I create a document term matrix, but on the base
of bigrams:

``` r
bigrams_dtm <- data_speech %>%
  count(id, bigram) %>%
  cast_dfm(id, bigram, n) %>%
  dfm_tfidf()

bigrams_dtm
```

    ## Document-feature matrix of: 136 documents, 2,151 features (98.36% sparse) and 0 docvars.
    ##     features
    ## docs barack obama donald trump im excite trump talk financial crisis hes forget
    ##    1     1.179296    0.2414443  1.531479   2.133539         0          0       
    ##    2     0           0.2414443  0          0                2.133539   2.133539
    ##    3     0           0.2414443  0          0                0          0       
    ##    4     0           0.2414443  0          0                0          0       
    ##    5     0           0.2414443  0          0                0          0       
    ##    6     0           0.2414443  0          0                0          0       
    ##     features
    ## docs single day white house african american bipartisan bill
    ##    1   0          0                0                0       
    ##    2   1.355388   0.6283889        0                0       
    ##    3   0          0.6283889        0.8325089        2.133539
    ##    4   0          0                0                0       
    ##    5   1.355388   0.6283889        0                0       
    ##    6   0          0                0                0       
    ## [ reached max_ndoc ... 130 more documents, reached max_nfeat ... 2,141 more features ]

This DTM is even more sparse than is was in case of unigrams, however,
for the text classification it is not a big problem.

As always, I need to divide my sample into two random corpuses: train
one and test corpus. Following the same logic, train corpus includes 70
% of data, test one - 30 %.

``` r
new_speech_ids <- data_speech %>%
  distinct(id, President)

set.seed(555)
new_split <- as.numeric(createDataPartition(y = new_speech_ids$id, p = 0.7, list = F))

new_train_dtm <- bigrams_dtm[new_split, ]

new_test_dtm <- bigrams_dtm[-new_split, ]

new_train_dtm
```

    ## Document-feature matrix of: 96 documents, 2,151 features (98.42% sparse) and 0 docvars.
    ##     features
    ## docs barack obama donald trump im excite trump talk financial crisis hes forget
    ##    1     1.179296    0.2414443  1.531479   2.133539         0          0       
    ##    2     0           0.2414443  0          0                2.133539   2.133539
    ##    4     0           0.2414443  0          0                0          0       
    ##    6     0           0.2414443  0          0                0          0       
    ##    7     0           0.2414443  0          0                0          0       
    ##    8     0           0.2414443  0          0                0          0       
    ##     features
    ## docs single day white house african american bipartisan bill
    ##    1   0          0                        0               0
    ##    2   1.355388   0.6283889                0               0
    ##    4   0          0                        0               0
    ##    6   0          0                        0               0
    ##    7   0          0                        0               0
    ##    8   0          0                        0               0
    ## [ reached max_ndoc ... 90 more documents, reached max_nfeat ... 2,141 more features ]

``` r
new_test_dtm
```

    ## Document-feature matrix of: 40 documents, 2,151 features (98.23% sparse) and 0 docvars.
    ##     features
    ## docs barack obama donald trump im excite trump talk financial crisis hes forget
    ##   3             0    0.2414443         0          0                0          0
    ##   5             0    0.2414443         0          0                0          0
    ##   10            0    0                 0          0                0          0
    ##   18            0    0                 0          0                0          0
    ##   19            0    0.2414443         0          0                0          0
    ##   20            0    0.2414443         0          0                0          0
    ##     features
    ## docs single day white house african american bipartisan bill
    ##   3    0          0.6283889        0.8325089        2.133539
    ##   5    1.355388   0.6283889        0                0       
    ##   10   0          0                0                0       
    ##   18   1.355388   0                0                0       
    ##   19   0          0                0                0       
    ##   20   0          0                0                0       
    ## [ reached max_ndoc ... 34 more documents, reached max_nfeat ... 2,141 more features ]

Train corpus contains contains 96 speeches, a test one - 40.

Also, I need to split indexes into the same corpuses, using already
existed split:

``` r
new_true_presidents <- as.factor(new_speech_ids$President)

new_train_presidents <- new_true_presidents[new_split]
new_test_presidents <- new_true_presidents[-new_split]
```

Before running the model, I need to evaluate it by **cross-validation**:

``` r
set.seed(556)
new_lg_model <- cv.glmnet(x = new_train_dtm, y = new_train_presidents,
                      alpha = 0, family = "binomial", type.measure = "auc", nfolds = 5,
                      lambda = seq(0.001, 0.1, by = 0.001), standardize = F)

new_lg_model
```

    ## 
    ## Call:  cv.glmnet(x = new_train_dtm, y = new_train_presidents, lambda = seq(0.001,      0.1, by = 0.001), type.measure = "auc", nfolds = 5, alpha = 0,      family = "binomial", standardize = F) 
    ## 
    ## Measure: AUC 
    ## 
    ##     Lambda Index Measure      SE Nonzero
    ## min    0.1     1  0.9934 0.00664    1689
    ## 1se    0.1     1  0.9934 0.00664    1689

**Min lambda** is equal 0.1. therefore, 0.1 will be used as a lambda
value in the regression model:

``` r
new_lg_model_best <- glmnet(x = new_train_dtm, y = new_train_presidents,
                        alpha = 0, family = "binomial",
                        lambda = new_lg_model$lambda.min, standardize = F)

new_lg_model_best
```

    ## 
    ## Call:  glmnet(x = new_train_dtm, y = new_train_presidents, family = "binomial",      alpha = 0, lambda = new_lg_model$lambda.min, standardize = F) 
    ## 
    ##     Df %Dev Lambda
    ## 1 1689 74.6    0.1

In this model, I use test data and minimal lambda parameter. As a
result, we have the number of nonzero coefficients (Df = 1689), the
percent of deviance explained (%Dev = 74.6, much less than in case of
unigrams, but still good) and the chosen value of lambda.

Now I can implenent classification process and measure it:

``` r
# predict class (party)
new_lg_pred <- as.factor(predict(new_lg_model_best, new_test_dtm, type = "class"))

# create matrix with main model metrics
new_lg_conf_matrix <- confusionMatrix(data = new_lg_pred, reference = new_test_presidents,
                                  # the choice of positive class affects the metrics calculation
                                  positive = "Clinton", mode = "prec_recall")

new_lg_conf_matrix
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction Clinton Trump
    ##    Clinton      12     5
    ##    Trump         1    22
    ##                                           
    ##                Accuracy : 0.85            
    ##                  95% CI : (0.7016, 0.9429)
    ##     No Information Rate : 0.675           
    ##     P-Value [Acc > NIR] : 0.01056         
    ##                                           
    ##                   Kappa : 0.6834          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.22067         
    ##                                           
    ##               Precision : 0.7059          
    ##                  Recall : 0.9231          
    ##                      F1 : 0.8000          
    ##              Prevalence : 0.3250          
    ##          Detection Rate : 0.3000          
    ##    Detection Prevalence : 0.4250          
    ##       Balanced Accuracy : 0.8689          
    ##                                           
    ##        'Positive' Class : Clinton         
    ## 

In general, model\`s fit indicators are comparable to what we had with
unigrams.

This model also works really good. 12 True positive (because I once
again identify “Clinton” as positive class) and 22 True negative, which
are correct classifications (also 5 False positive (with unigrams were
only 3) and 1 false negative). It means that model correctly classifies
about 85 % of speeches, as it also shown by **accuracy** metric (0.8).

Important to not that **recall** score (0.9231) is also high. It means
that model correctly detect about 92 % of positive cases among all
positive instances.

Also, **precision** is equal 0.7 (much less incomparison to what was
with unigrams), which is still a good score. It means that model
correctly detect about 70 % of positive cases among all predicted
positive cases. So, the **type I error** is more frequent in this
classification.

As for **F1**, it is equal to 0.8, with means that about 80 % of
predictions are correct.

In general, these results are good enough to once again call this model
well functioning.

Moreover, I can check **words that are the most significant** for each
corpus (Clinton and Trump):

``` r
# important words for positive class (H. Clinton)
coef(new_lg_model_best) %>%
  as.matrix() %>%
  as.data.frame() %>%
  tibble::rownames_to_column() %>%
  arrange(-s0) %>%
  filter(rowname != "(Intercept)") %>%
  head(10)
```

    ##                     rowname        s0
    ## 1          hillary clintons 0.1654681
    ## 2             trade deficit 0.1439367
    ## 3         incredible people 0.1400887
    ## 4                   job job 0.1395613
    ## 5           hillary clinton 0.1380865
    ## 6         illegal immigrant 0.1379713
    ## 7             border patrol 0.1362940
    ## 8              whats happen 0.1330881
    ## 9  transpacific partnership 0.1307922
    ## 10        replace obamacare 0.1296806

``` r
# important words for negative class (D. Trump)
coef(new_lg_model_best) %>%
  as.matrix() %>%
  as.data.frame() %>%
  tibble::rownames_to_column() %>%
  arrange(s0) %>%
  filter(rowname != "(Intercept)") %>%
  head(10)
```

    ##             rowname         s0
    ## 1      captain khan -0.1391769
    ## 2      middle class -0.1390598
    ## 3         equal pay -0.1380119
    ## 4      minimum wage -0.1364896
    ## 5      gun violence -0.1362734
    ## 6    american dream -0.1216967
    ## 7        single day -0.1157331
    ## 8         im excite -0.1148265
    ## 9  renewable energy -0.1146877
    ## 10 health insurance -0.1049460

Words that are significant for Clinton\`s speeches are positive. On the
other hand, Trump words coefficients are negative.

In general, for Clinton\`s speeches, we can observe words that are
related to different topics. First of all, for the model it was easy to
identify her speeches based on her name: “hillary clinton” or “hillary
clintons” are the most significant here, which is similar to the
previous analysis. She use them a lot, probably, because of
self-promotion. Also, there are some words related to the foreign
policy, including international economy (transpacific partnership, trade
deficit) and to the domestic policy (job job). There are some just
regular expressions (incredible people, whats happen).

However, all other bigrams here a a kind of surprise for me, because,
from the first look, they are more related to the D. Trump discourse:
illegal immigrant and border patrol - a hot topic for all republicans;
replace obamacare - as i was discussed, Trump was against this project.
But these bigrams are included in Clinton\`s discource and, moreover,
have high coefficients.

As for D. Trump, for the model it was easy to classified his texts as
Trump\`s by words, some of which were related to the domestic policy
issues, including human rights and economy (middle class, equal pay,
minimum wage, gun violence, renewable energy, health insurance).

Also, there are some words that are related to a particular person
(captain khan (a soldier in Iraq) and some regular expressions (single
day, american dream, im excite)

In general, many of these Trump\`s words are also, in theory, more
related to the H. Clinton discourse.

Because I have some misclassifications, I would be useful to check,
which bigrams in both corpuses (Trump and Clinton) cause this confusion.

Now I explore Clinton\`s speeches that were incorrectly classified as
made by Trump and check main bigrams, which led to this
misclassification:

``` r
new_error_df <- new_speech_ids[-new_split, ]

new_misclassified_Clinton <- error_df[which(new_lg_pred == "Trump" & new_test_presidents == "Clinton"), ] %>%
  left_join(data_speech, by = c("id", "President"))

new_top_misclassified_bigrams <- new_misclassified_Clinton %>%
  count(bigram, sort = T) %>%
  head(10)

new_top_misclassified_bigrams
```

    ## # A tibble: 10 × 2
    ##    bigram                  n
    ##    <chr>               <int>
    ##  1 air force               1
    ##  2 ballot box              1
    ##  3 beautiful beautiful     1
    ##  4 big crowd               1
    ##  5 billy graham            1
    ##  6 camp lejeune            1
    ##  7 clean coal              1
    ##  8 coal clean              1
    ##  9 couldnt stand           1
    ## 10 crook hillary           1

Compare to what was with unigrams, here it is much more understandble,
why this one document was misclassified. I also check model coefficients
of these words:

``` r
coef(new_lg_model_best) %>%
  as.matrix() %>%
  as.data.frame() %>%
  tibble::rownames_to_column() %>%
  filter(rowname %in% new_top_misclassified_bigrams$bigram) %>%
  arrange(s0)
```

    ##                rowname          s0
    ## 1            air force -0.01852869
    ## 2           ballot box  0.01595827
    ## 3  beautiful beautiful  0.01595833
    ## 4           coal clean  0.01595838
    ## 5        couldnt stand  0.01595843
    ## 6         billy graham  0.02521643
    ## 7            big crowd  0.02563307
    ## 8         camp lejeune  0.02657102
    ## 9           clean coal  0.05221372
    ## 10       crook hillary  0.10024631

However, according to coefficients, just one bigram is related to the
Trump\`s discourse in reality: “air force”, which is dedicated to a
military topic, widely used by D. Trump. Other bigrams, even “crook
hillary”, “camp lejeune” (US military base) and billy graham (famous
religious activist) are more related to H. Clinton because of their
positive coefficients.

As for Trump speeches that were classified as Clinton\`s:

``` r
new_error_df <- new_speech_ids[-new_split, ]

new_misclassified_Trump <- error_df[which(new_lg_pred == "Clinton" & new_test_presidents == "Trump"), ] %>%
  left_join(data_speech, by = c("id", "President"))

new_top_misclassified_bigrams <- new_misclassified_Trump %>%
  count(bigram, sort = T) %>%
  head(10)

new_top_misclassified_bigrams
```

    ## # A tibble: 10 × 2
    ##    bigram              n
    ##    <chr>           <int>
    ##  1 donald trump        2
    ##  2 hillary clinton     2
    ##  3 american people     1
    ##  4 american troop      1
    ##  5 american worker     1
    ##  6 average cost        1
    ##  7 barack obama        1
    ##  8 big story           1
    ##  9 brother robert      1
    ## 10 bruce blair         1

Compare to what was with unigrams, here it is much more understandble,
why this one document was misclassified. I also check model coefficients
of these words:

``` r
coef(new_lg_model_best) %>%
  as.matrix() %>%
  as.data.frame() %>%
  tibble::rownames_to_column() %>%
  filter(rowname %in% new_top_misclassified_bigrams$bigram) %>%
  arrange(desc(s0))
```

    ##            rowname           s0
    ## 1  hillary clinton  0.138086478
    ## 2  american worker  0.110585745
    ## 3  american people  0.087102283
    ## 4     barack obama  0.008979997
    ## 5     average cost  0.000000000
    ## 6        big story  0.000000000
    ## 7   brother robert  0.000000000
    ## 8   american troop -0.014700639
    ## 9      bruce blair -0.076416329
    ## 10    donald trump -0.079652556

However, according to coefficients, just four bigrams are related to the
Clinton\`s discourse in reality: “hillary clinton”, “american worker”,
which is probably dedicated to the human rights issue in economic
context (this topic is really widely used by H. Clinton), “american
people” (a more regular phrase), “barac obama”, who was supported by
Clinton and criticized by Trump. Other bigrams, including “american
troop” and “bruce blair” (nuclear weapone scientist) are more related to
D. Trump because of their negative coefficients.

So, with this sample, classification logistic regression model works
brilliant even in case of bigrams. There are not so many False negative
and False positive. However, with bigrams, we can say that The main
reason for it, probably, is that speeches of candidates are too
different in case of some meaningful terms and topic, which, actually,
supports above mentioned hypothesis.

As a conclusion, I can say that, firs of all, the general hypothesis is
approved: it is easy to distinguish between campaign speeches of D.
Trump and H. Clinton. However, in some cases, they are differentiate
only based on the different length of regular words frequencies. This an
issue of a texts type: even if we conclude that these speeches are based
not only motivational phrases, but still, they contain a lot of regular
words and expressions. Also, the sample is a bit limmited: even with a
fact that all texts are long enough, there are only 136 observations,
which can cause such a good classification. However, I hope that it was
not the main reason and texts length levels this problem out.

An interesting finding is that not all topics that are clearly
identified with the use of frequency lists, are supported by important
words extracted from regression model results. For example, in case of
bigrams, we see that H. Clinton also widely used such topics as
militarism, immigrants and foreign policy.

As for the comparison between to analysis (unigrams and bigrams), model
works in pretty much the same way, with comparable metric scores at the
end. I stated that, probably, bigrams can bring more contextual
information in the analysis, especially in work with misclassifications,
and, to some extend, it is true. Therefore, I think that classification
with bigrams ia more appropriate here.

However, still, the most informative part, in my opinion, is always
produced by regular frequency lists and tf-idf score, which are, for
now, my favorite text analysis methods because of their simplicity,
clarity and, at the same time, efficiency.

**List of Sources**

1.  Rowbottom, J. (2012). Lies, manipulation and elections—Controlling
    false campaign statements. Oxford Journal of Legal Studies, 32(3),
    507-535.
2.  Gunawan, S. (2018). Donald Trump s versus Hillary Clinton s Campaign
    Rhetoric in their Presidential Nomination Acceptance Speeches
    (Doctoral dissertation, Petra Christian University).
