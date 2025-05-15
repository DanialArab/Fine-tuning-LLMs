# Fine-tuning LLMs

This repo documents my understanding of fine-tuning LLMs, which is mostly based on chapters 1, 2, and 6 of the book **<a href="https://www.manning.com/books/build-a-large-language-model-from-scratch">Build a Large Language Model (From Scratch)</a>** by Sebastian Raschka. 

# Table of content

1. [Intro](#1)
2. [Data preparation to make them LLM-friendly](#2)
   1. [Balancing the classes, a bit of cleaning, and splitting](#3)
   2. [Creating PyTorch Dataloaders, performing Padding or Truncation](#4)
4. [Re-architecting the LLM](#3)
5. [Fine-tuning LLM](#4)
6. [Model evaluation](#10)
7. [Model deployment](#6)
  
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



