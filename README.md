# Fine-tuning LLMs

This repo documents my understanding of fine-tuning LLMs, which is mostly based on the chapters:
- Chapter 1: Understanding large language models
- Chapter 2: Working with text data, and
- Chapter 6: Fine-tuning for classification

of the book **<a href="https://www.manning.com/books/build-a-large-language-model-from-scratch">Build a Large Language Model (From Scratch)</a>** by Sebastian Raschka. 

# Table of content

1. [Intro](#1)
2. [Data preparation to make them LLM-friendly](#2)
   1. [Balancing the classes, a bit of cleaning, and splitting](#3)
   2. [Creating PyTorch Dataloaders, performing Padding or Truncation](#4)
3. [Re-architecting the LLM](#5)
   1. [Model initialization with pre-trained weights](#6)
   2. [Adding the classification head to the pre-trained model](#7)
   3. [Loss calculation](#8)
6. [Fine-tuning LLM](#9)
7. [Model evaluation](#10)
8. [Making inference](#11)
  
<a name="1"></a> 
## Intro
Here, I will focus on **classification fine-tunning** as a technique to get LLM working as an alternative to traditional classifiers. To fine-tune LLM, we need a pipeline to perform:
- Data preparation to make them LLM-friendly
- Re-architecting the LLM
- Fine-tuning LLM
- Model evaluation
- Model deployment

<a name="2"></a>
## Data preparation to make them LLM-friendly

<a name="3"></a>
### Balancing the classes, a bit of cleaning, and splitting

Here, I will work with **<a href="https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip">Spam/Ham email dataset</a>**. This data is imbalanced:

    Label
    ham     4825
    spam     747
    Name: count, dtype: int64

Balancing a dataset is a requirement whenever we are dealing with a classification task, there are a lot of different ways of doing this, but here I will go ahead with the undersampling approach and simply doing a random sampling from the majority class to match the number of minority class:

    Label
    ham     747
    spam    747
    Name: count, dtype: int64

Then we need to convert the class labels from string to integer:

    Label
    0    747
    1    747
    Name: count, dtype: int64


I split the dataset into training, validation, and test sets with a 70/10/20 ratio.

<a name="4"></a>
### Creating PyTorch Dataloaders, performing Padding or Truncation

While working with text data of different lengths, we need to make them consistent with either padding (pad all data to the length of the longest in the dataset or batch) or truncation (truncate all data to the length of the shortest in the dataset or batch). I choose padding to make sure I do not lose any useful information, down the road we can revisit this decision after having a chance to go over the results and see how computationally expensive this could be. This should be consistently done in training, validation, and test datasets. 

<a name="5"></a>
## Re-architecting the LLM

<a name="6"></a>
### Model initialization with pre-trained weights

To build a fine-tuned LLM classifier, there is a variety of options as the pre-trained/foundation model. I chose the **gpt2-small (124M)** model to make the implementation less expensive. Here is a quick summary of the model configuration:

|**Embedding dimensions** | **Number of transformer layers (blocks)**|**Number of attention heads per transformer layer** | **Number of parameters** | **Context window**| **Vocabulary size**| 
| -- | --|  -- | -- | -- | --|
|768 | 12 |12 | 124 M | 1024 tokens  | 50257|

<a name="7"></a>
### Adding the classification head to the pre-trained model

I need to modify the pre-trained LLM architecture to make it fit for a classification task. To do so, I need to replace the original output layer, which maps the hidden representation to a vocabulary of 50,257 (the total number of unique tokens the model was trained to recognize and generate during pretraining), with a smaller output layer that maps to two classes: 0 (negative class) and 1 (positive class), as shown below:

![](https://github.com/DanialArab/images/blob/main/llm_from_scratch/re-architecture.jpg)

This modification is necessary because, instead of performing text generation, which is the primary task of the pre-trained model, we now need the model to predict 2 classes as the output of the fine-tuned model. 

<a name="8"></a>
### Loss calculation

In the generative tasks, LLM generates the token ID of the next token through converting the 50,257 outputs into probabilities via the softmax function and then returning the position of the highest probability via the argmax function. Here in the classification task, the same approach is taken to calculate whether the model outputs a positive class or negative class prediction for a given input. The only difference is that in the classification task we work with 2-dimensional instead of 50,257-dimensional outputs. Before fine-tuning the prediction accuracies are as follows:

      Training accuracy: 46.25%
      Validation accuracy: 45.00%
      Test accuracy: 48.75%

As shown above, the accuracies are almost near a random classifier. We try to increase them through fine-tuning. 

<a name="9"></a>
## Fine-tuning LLM

The next step is to fine-tune the model on the supervised data. I did the training for 5 epochs, loss and accuracy for the training and validations sets are presented below: 

![](https://github.com/DanialArab/images/blob/main/llm_from_scratch/loss_and_accuracy_fine_tune_llm_classifier.png)

And the accuracies for the fine-tuned model are:

      Training accuracy: 97.21%
      Validation accuracy: 97.32%
      Test accuracy: 95.67%

<a name="10"></a>
## Model evaluation

Here are the classification report along with the confusion matrix:

                  precision    recall  f1-score   support
      
                 0       0.98      0.81      0.88       149
                 1       0.84      0.98      0.90       151
      
          accuracy                           0.89       300
         macro avg       0.91      0.89      0.89       300
      weighted avg       0.91      0.89      0.89       300

![](https://github.com/DanialArab/images/blob/main/llm_from_scratch/ham_vs_spam_cm.png)

<a name="11"></a>
## Making inference

Here are some inferences using the fine-tuned GPT-based spam classifier: 

      text_1 = (
          "You are a winner you have been specially"
          " selected to receive $1000 cash or a $2000 award."
      )

Model prediction:

      spam

another example:

      text_2 = (
          "Hey, just wanted to check if we're still on"
          " for dinner tonight? Let me know!"
      )

Model prediction:

      ham
