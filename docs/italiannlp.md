
Italian Word Embeddings
=======================

We release two sets of word embeddings trained starting from two different corpora. These word embeddings were used for our participation at EVALITA 2018 edition [^1]

1.  itWaC: billion word corpus constructed from the Web limiting the crawl to the .it domain and using medium-frequency words from the Repubblica corpus and basic Italian vocabulary lists as seeds. [^2]
2.  Twitter: 46.935.207 tweets.

The word embeddings are 128-sized and were generated with word2vec with the following command:

```sh
word2vec -train corpus.doc -output vec128.bin -size 128 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 1 -cbow 1
```

Preprocessing
-------------

The corpora were not linguistically analyzed (forms were used to generate the embeddings). Tokenization was the only process performed on both the corpora. The corpora were tokenized according to standard Italian tokenization rules.
For example the sentence:

> Questo è l’apostrofo, con una virgola ed un punto e virgola;

is tokenized as follows:

> Questo / è / l’ / apostrofo / , / con / una / virgola / ed / un / punto / e / virgola / ;

Try the [LinguA](http://linguistic-annotation-tool.italianlp.it/) tool to obtain more tokenization examples.

Both corpora were preprocessed with the aim of reducing the vocabulary size.
To obtain the embedding of a word, you should use the same normalization process we adopted in the preprocessing step.
More precisely:

Numbers:
1. Integer numbers between 0 and 2100 were kept as original
2. Each integer number greater than 2100 is mapped in a string which represents the number of digits needed to store the number (ex: 10000 -> DIGLEN\_5)
3. Each digit in a string that is not convertible to a number must be converted with the following char: @Dg. This is an example of replacement (ex: 10,234 -> @Dg@Dg,@Dg@Dg@Dg)

Words:
1.  A string starting with lower case character must be lowercased (e.g.: (“aNtoNio” -> “antonio”), (“cane” -> “cane”))
2.  A string starting with an upcased character must be capitalized (e.g.: (“CANE” -> “Cane”, “Antonio”-> “Antonio”))

This preprocessing step is implemented in a script that you can download [here](http://www.italianlp.it/we-distributed/norm_script.py)

File format
-----------

We converted the generated word embedding model files in a sqlite file with the following schema:

```
Key      text PRIMARY KEY, 
dim0     REAL,
dim1     REAL,
 ...
dim127   REAL, 
ranking  INT
```

Where *text* is the word, *dim0*,…,*dim127* are the components of each vector, and *ranking* represents the rank (the lower is the value, the higher is the frequency in the corpus).
You can easily convert the sqlite file in a tabbed format (one line for each word) file with [this](http://www.italianlp.it/we-distributed/conv_script.py) script.

Download
--------

Click [here](http://www.italianlp.it/download-italian-twitter-embeddings) to download the Italian Twitter embeddings

Click [here](http://www.italianlp.it/download-itwac-word-embeddings) to download the itWac embeddings

References
----------

[^1]: Cimino A., De Mattei L., Dell’Orletta F. (2018) "[*Multi-task Learning in Deep Neural Networks at EVALITA 2018*](http://ceur-ws.org/Vol-2263/paper013.pdf)". In Proceedings of EVALITA ’18, Evaluation of NLP and Speech Tools for Italian, 12-13 December, Turin, Italy.

*(Please cite the paper above if you make use of these embeddings in your research)*

[^2]: http://wacky.sslmit.unibo.it/doku.php?id=corpora

