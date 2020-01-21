---
layout: post
mathjax: true
title: Categorising Scientific Papers
---

I recently finished `Scholarly`, a long-standing side project of mine, which consists of predicting the category of a given title and abstract of a scientific paper. More precisely, I am predicting the ~150 subject classification categories from the [arXiv](https://arxiv.org/) preprint server, and have trained the model on _all_ papers on the arXiv up to and including year 2019. Test out the model here:

<center>
  <a href="https://saattrupdan.pythonanywhere.com/scholarly">
    saattrupdan.pythonanywhere.com/scholarly
  </a>
</center>

All the source code can be found in the [Github repo](https://github.com/saattrupdan/scholarly) and the model and all the data can be found in the [pCloud repo](https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data).

Now, with all the practicalities out of the way, in this blog post I'd like to talk about the process of the project, the data I've been using and the model I ended up using. Let's start from the beginning.

<center>
  <figure>
    <img src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/arxiv_example.jpg" alt="Example of an arXiv entry" style="width:50%;">
    <figcaption>
      An example of arXiv categories in action
    </figcaption>
  </figure>
</center>

## Scraping all of arXiv
Turns out that ArXiv has an [API](https://arxiv.org/help/api/index). There are a couple of quirks and limitations, however. Firstly, it requires you to query something, you cannot just put in a blank query. To get around that it turns out that it's completely fine if you put in a blank query _for a particular category_. So by looping through all the categories I would be able to get all the papers. Great!

Next up is that the API can only return 10,000 papers at a time, and if you query it repeatedly it blocks you out temporarily. The solution? Write a scraping script that queries in batches, and which takes breaks if the arXiv decides that it's break time. Scraping with breaks is also considered [good practice](https://info.scrapinghub.com/web-scraping-guide/web-scraping-best-practices#1), and note that the actual scraping is taking place on [export.arxiv.org](https://export.arxiv.org), which is copy of the arXiv meant for programmable purposes, updated daily. Now, we have a scraper that's scraping away, a new problem arises: it turns out that you can't really tell the difference between being blocked out and there being no results left in that particular category. My hacky solution to that was to set an upper bound on the amount of tries (=amount of patience) the scraping script should attempt before moving on to the next category. 

As my laptop is low on memory I had to come up with a way to store all this data in an efficient manner. I tried `tsv` files and `json` files, but both of them had the annoying feature of needing to at some point store the entire file in memory (unless I'm missing some neat trick here). So instead, I dived into SQL.

<center>
  <figure>
    <img src="https://imgs.xkcd.com/comics/exploits_of_a_mom.png" alt="XKCD comic about SQL" style="width:50%;">
    <figcaption>
      There's nothing like a dry SQL joke
    </figcaption>
  </figure>
</center>

 I ended up going with a local SQLite database, which is a standalone SQL database in a single file. The `sqlalchemy` package provides a nice Python interface to work with SQL databases, which can both work with the databases in a [purely object-oriented manner](https://en.wikipedia.org/wiki/Object-relational_mapping) or by simply providing a way to query SQL statements; I chose the latter, as that turned out to be a lot faster. This process also taught me a lot about how to work with SQL databases in an efficient manner. For instance, I found that it's _way_ more efficient to have _really_ long and few queries: inserting 1,000 entries into my database in 1,000 queries took hours, but doing it all in a single query took seconds!

As I was suddenly working with a database, I wanted to take advantage of the relational structure and created separate tables for the categories, papers and authors, and linked these together. This also yields a highly structured dataset of scientific papers and authors, which also might be of independent interest. With all of these components the script successfully downloaded all the titles, abstracts, dates and authors from the arXiv in less than three days.

## Prepping the data: from SQL to tsv
From my database I created a `tsv` file for my particular classification application, containing the titles, abstracts and all the categories that the given paper belonged to. Even only working with this smaller (~2GB) dataset was causing my laptop to give up on life. For testing purposes on my laptop I therefore wrote a script that utilises [Numpy memory maps](https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html) to be able to create smaller randomly sampled versions of the dataset, without needing to load the large one into memory.

The amount of preprocessing was quite minimal: I replaced LaTeX equations like $\sqrt{1}{5}$ with `-EQN-` and used the tokens `-TITLE_START-`, `-TITLE_END-`, `-ABSTRACT_START-` and `-ABSTRACT_END-` to separate the titles and abstracts, allowing the model to distinguish between the two. To tokenise the titles and abstracts I used the [spaCy](https://spacy.io/) `en-core-web-sm` NLP model, which is nice and fast (the [pipe](https://spacy.io/api/tokenizer#pipe) method came in handy here).

## So, why not train our own word vectors?
Since I'm dealing with a massive dataset I decided to train my own word vectors from scratch, which would both allow the model to work with "scientific language" as well as having neat vector encodings of the special `-EQN-`, `-TITLE_START-`, `-TITLE_END-`, `-ABSTRACT-START-` and `-ABSTRACT_END-` tokens. I trained bigram [FastText](https://fasttext.cc/) 100d vectors on the corpus in an unsupervised manner. The vectors live up to their name: it only took a couple of hours to train the vectors on the entire dataset! The resulting model and vectors can be found in the above-mentioned [pCloud repo](https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data).

<center>
  <figure>
    <img src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/vector_comparison.png" alt="Comparison of model performance when trained on the homemade FastText vectors and the pre-trained GloVe vectors. FastText wins massively." style="width:50%;">
    <figcaption>
      A comparison of my homemade FastText vectors and the pre-trained GloVe vectors.
    </figcaption>
  </figure>
</center>

The above plot is a comparison of the homemade FastText vectors and pre-trained [6B GloVe vectors](https://nlp.stanford.edu/projects/glove/), trained on Wikipedia (both are 100-dimensional). As the plot shows, it *can* be worth it to train your own word vectors from scratch on your own corpus. Note that this is despite the fact that the pre-trained ones have been trained on a much larger corpus!

## Self-attention and all that jazz
The model that I ended up using after much trial and error was a simpler version of the recent [SHA-RNN architecture](https://arxiv.org/abs/1911.11423). More precisely, here's what's going on:

  1. The word vectors are plugged into a bi-directional [Gated Recurrent Unit](https://en.wikipedia.org/wiki/Gated_recurrent_unit) with 2x256 hidden units
  2. We apply a [scaled dot product self-attention](https://arxiv.org/abs/1706.03762) on the 3d outputs from the GRU, to allow the model to attend to the important tokens in the text. After taking the corresponding weighted sum we end up with a 2d tensor.
  3. Next, we project down to dimension 148, the number of categories we're trying to predict
  4. We then apply yet another self-attention on the 2d outputs, with the idea being to spot similarities between the logits for every category
  5. Lastly, we apply a **Boom** layer: two fully connected layers, blowing the (32, 148)-shaped input up to (32, 512) and then projecting it back down to dimension (32, 148).

There are layer normalisations happening at nearly every layer, and the [GELU activation](https://arxiv.org/abs/1606.08415) is used everywhere. All of this ends up having ~800,000 trainable parameters (the embedding weights are frozen), which is not too shabby at all with a ~1.3M dataset. I used no regularisation for the same reason. I tried adding 10% and 20% dropout, but both just resulted in poorer performance.

## Nested Binary Cross Entropy Loss
Since it's basically impossible to have perfect predictions on these ~150 categories, I wanted to ensure that the model would at least get close when it's wrong. I decided to do this by grouping the categories together in the six official "master categories" of arXiv: Mathematics, Physics, Computer Science, Statistics, Quantitative Biology and Quantitative Finance.

I didn't want to train a separate model on these master categories; I didn't even want the master categories to be labelled in my datasets. So what I did instead was to simply describe which categories go in what master categories, and from that I could define a custom loss functions which also penalised the model for getting the master categories wrong. Here's what the loss function is doing:

  1. Duplicate the category logits 6 times, for each of the six master categories
  2. Mask each copy corresponding to the master category
  3. "Project" the masked copies down to logits for the master categories
  4. Apply weighted binary cross entropy on both the category logits and the master category logits
  5. Take a weighted sum of the two losses

The projection in step 3 works by "mixing the top2 logits within each master category". To understand what I mean by that, let's do an example. Say you roll two dice: what's the probability of there being at least one of them hitting 6? This would be $1 - \left(\tfrac{5}{6}\right)^2 \sim 31%$, as there's a $\tfrac{5}{6}$ chance of it not being 6. This would be an instance of "mixing" the two probabilities $\tfrac{1}{6}\sim 17%$ into $31%$. The problem with this is that it's using the probabilities instead of the logits, but I need the logits to allow utilising class weights. One naive solution to this would be to translate the logits to probabilities, perform the mixing and then translate back, but this causes rounding issues: some probabilities would be rounded to 100%, which are translated back to infinite logits, yielding NaN loss. Woohoo. Instead, some simple algebra gives us that we can perform the mixing directly on the logits by the following formula:

$$ \text{mix}(x, y) := x + y + \log(1 + e^{-x} + e^{-y}). $$

So, to recap, in step 3 we are taking the top2 logits from each master category, mixing them in the above sense, and then using those values to compute the master category (weighted) binary cross-entropy loss. This means that if, for instance, the top2 probabilities within a master category are both 30% then the mixed probabilities will be ~51%, thus yielding a positive prediction for the master category even though the category predictions are all false.

Lastly there's a question of ratio in step 5: how much priority should the model give to the master category loss over the category loss? I performed a handful of experiments and found that giving a 0.1 weight to the master category loss and a 0.9 weight to the category loss performed the best. A ratio of 0 meant that the master category score suffered drastically, and likewise for a ratio near 1 for the category score. I also tried starting with a large ratio and reducing it exponentially, but that turned out to not make any notable difference.

## Results
The score that I was using was the *sample-average F1 score*, which means that for every sample I'm computing the F1 score of the predictions of the sample (note that we are in a [multilabel](https://en.wikipedia.org/wiki/Multi-label_classification) setup), and averaging that over all the samples. If this was a [multiclass](https://en.wikipedia.org/wiki/Multiclass_classification) setup (in particular binary classification) then this would simply correspond to accuracy. The difference is that in a multilabel setup the model can be *partially* correct, if it correctly predicts some of the categories.

<center>
  <figure>
    <img src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/master_cats.png" alt="A plot of the sample-average F1 score of the master categories on the training- and validation set. The training score converges to ~95% and the validation score to ~93%." style="width:50%;"><img src="https://filedn.com/lRBwPhPxgV74tO0rDoe8SpH/scholarly_data/all_cats.png" alt="A plot of the sample-average F1 score of all the categories on the training- and validation set. The training score converges to ~68% and the validation score to ~64%." style="width:50%;">
    <figcaption>
      The final scores.
    </figcaption>
  </figure>
</center>

The model ended up achieving a ~93% and ~65% validation sample-average F1 score on the master categories and all the categories, respectively. Training the model requires ~17GB memory and it takes roughly a day to train on an Nvidia P100 GPU. This was trained on the [BlueCrystal Phase 4 compute cluster](https://www.acrc.bris.ac.uk/acrc/phase4.htm) at University of Bristol, UK.

## Monitoring progress
A shout out also goes out to the people at [Weights & Biases](https://www.wandb.com/), which made it incredibly easy for me to compare my models' performance, even though some of them were trained on the compute cluster, some of them in [Colab notebooks](https://colab.research.google.com/), some on my laptop and some on my office computer. Highly recommended, and it's even free. You can check out my training runs at my WandB project here:

<center>
  <a href="https://app.wandb.ai/saattrupdan/scholarly/runs/3kv495v2/overview">
    https://app.wandb.ai/saattrupdan/scholarly/runs/3kv495v2/overview 
  </a>
</center>

