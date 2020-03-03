# QAAM (Question Answering Auto Model)

- **Automatic question answering from any text source**

`QAAM` is a question and answering engine for answering questions of any text documents or texts extracted from a website's URL (see below). The model leverages a fine-tuned representation on *SQuAD* from the `Transformers` library while the context treated by proper tokenization techniques for online text.

## Installation

- To use the model install the required dependencies:

```bash
numpy==1.17.3
scipy==1.4.1
cupy==7.0.0
transformers==2.3.0
spacy==2.2.3
david==0.0.1
```

```bash
pip install -r requirements.txt
pip install .
```

## Usage

Extract text content from a website's blog or article:

> First, import the `QAAM` class. Below are some of the default parameters that can be configured to accommodate a document's environment.

- You can add texts with one of following instance methods:
  - `self.texts_from_str(str:Sequence[str])`
  - `self.texts_from_doc(doc:List[str])`
  - `self.texts_from_url(url:str)`

```python
from qaam import QAAM
qaam = QAAM(0.2, mode='tfidf', metric='cosine')

blog_url = ("https://medium.com/analytics-vidhya/semantic-"
            "similarity-in-sentences-and-bert-e8d34f5a4677")

# texts are processed automatically
qaam.texts_from_url(blog_url)
qaam.common_entities(10, lower=True, lemma=True)
...
[('bert', 14),
 ('nlp', 8),
 ('two', 5),
 ('google', 3),
 ('one', 3),
 ('2018', 2),
 ('glove', 2),
 ('universal sentence encoder', 2),
 ('1', 2),
 ('2', 2)]
```

How to query questions from a website's text content:

```python
from pprint import pprint

question = "Why is it good to use pre-trained sentence encoders?"
prediction = qaam.answer(question)
pprint(prediction)
...
# predicted answer
{'answer': 'it takes much less time to train a fine-tuned model.',
# cosine-similary context
 'context': 'Using minimal task-specific fine-tuning efforts, researchers have '
            'been able to surpass multiple benchmarks by leveraging '
            'pre-trained models that can easily be implemented to produce '
            'state of the art results. It soon became common practice to '
            'download a pre-trained deep network and quickly retrain it or add '
            'additional layers on top for the new task. Pre-trained sentence '
            'encoders aim to play the same role as word2vec and GloVe play for '
            'words. Consequently, it takes much less time to train a '
            'fine-tuned model. Note that you will have to choose the correct '
            'path and pre-trained model name for BERT 3. It is therefore '
            'referred to as Multi-head Attention.',
 'end': 511,
# fetched input query/question
 'question': 'Why is it good to use pre-trained sentence encoders?',
 'score': 0.317362858170501,
 'start': 459}
```

> How does the `input-to-context-adjustment` technique yields faster and more accurate results from more complex methods? In short, `qaam` appropriately adjusts (fits) the input query to the environment's vocabulary.

In the query below, the output remains equivalent to the result above - regardless of incorrect spelling or grammar. Therefore, the adjustment is executed before computing the ***cosine-distance metric*** (to build the context) and transferring the question to the ***Transformers Auto-Model*** for question-answering.

A word like `food` is correct, but it is not correct in terms of the document's context. So the word is automatically adjusted to the most likely intention based on the surrounding words.

```python
question = "Why is it food to use pre-trained sentencr encoters?"
prediction = qaam.answer(question)
pprint(prediction)
...
{'answer': 'it takes much less time to train a fine-tuned model.', # predicted answer
 'context': 'Using minimal task-specific fine-tuning efforts, researchers have '
            'been able to surpass multiple benchmarks by leveraging '
            'pre-trained models that can easily be implemented to produce '
            'state of the art results. It soon became common practice to '
            'download a pre-trained deep network and quickly retrain it or add '
            'additional layers on top for the new task. Pre-trained sentence '
            'encoders aim to play the same role as word2vec and GloVe play for '
            'words. Consequently, it takes much less time to train a '
            'fine-tuned model. Note that you will have to choose the correct '
            'path and pre-trained model name for BERT 3. It is therefore '
            'referred to as Multi-head Attention.',
 'end': 511,
 'question': 'Why is it good to use pre-trained sentence encoders?',  # adjusted input
 'score': 0.317362858170501,
 'start': 459}
```

- Here `BERTO` is adjusted to correct context-term: `BERT`.

```python
prediction = qaam.answer("How was BERTO trained?")
pprint(prediction)
...
{'answer': 'on the Wiki corpus,',  # predicted answer
 'context': 'Like all models, BERT is not the perfect solution that fits all '
            'problem areas and multiple models may need to be evaluated for '
            'performance depending on the task. Pre-trained representations '
            'can either be context-free or contextual, and contextual '
            'representations can further be unidirectional or bidirectional. '
            'However, there are easy wrapper services and implementations like '
            'the popular bert-as-a-service that can be used to that effect. In '
            'the case of BERT, having been trained on the Wiki corpus, the '
            'pre-trained model weights already encode a lot of information '
            'about our language. However, BERT represents “train” using both '
            'its previous and next context — This is a simple example of the '
            'popular bert-as-a-service. Therefore, BERT embeddings cannot be '
            'used directly to apply cosine distance to measure similarity.',
 'end': 500,
 'question': 'How was BERT trained?', # adjusted input
 'score': 0.5697720375227675,
 'start': 481}
```

## Cosine Similarity

Scores based on `TfIdf`

```python
matrix_a = qaam.to_matrix("context into a sequence")
matrix_b = qaam.to_matrix("context based on a sequence")
qaam.cosine_similarity(matrix_a, matrix_b)
0.75
```

Scores based on spaCy's `en_core_web_sm` model

```python
tensor_a = qaam.to_tensor("context into a sequence")
tensor_b = qaam.to_tensor("context based on a sequence")
qaam.cosine_similarity(tensor_a, tensor_b)
...
0.8587002754211426
```

## Asking Questions to YouTube Comments

```python
from pprint import pprint
from tqdm import tqdm
from qaam import QAAM
from david.youtube import YTCommentScraper

qaam = QAAM()
scraper = YTCommentScraper()
video_url = "https://www.youtube.com/watch?v=EYIKy_FM9x0&t=2189s"
iterbatch = scraper.scrape_comments(video_url=video_url)

# available keys: {text, time, author, cid}
comments = [comment["text"] for comment in iterbatch]
...
comments: 150 [00:08, 17.88/s]
```

```python
# add the comments to the model
qaam.texts_from_doc(comments)
qaam.common_entities(10, lower=True, lemma=True)
...
[('lex', 13),
 ('jordan', 11),
 ('agi', 7),
 ('first', 7),
 ('michael', 4),
 ('elon musk', 4),
 ('@ataraxia', 4),
 ('elon', 3),
 ('michael jordan', 3),
 ('two', 3)]
```

```python
pprint(qaam.answer("What is an interesting topic of the video?"))
...
  Converting examples to features: 100%|██████████| 1/1 [00:00<00:00, 145.69it/s]

  {'answer': 'what would the creation of AI',
   'context': 'Many interesting topics - 1:34:36 LF: what would the creation of '
              'AI take and how that would be different from human '
              'intelligence?   The lack of knowledge about functioning of the '
              'brain and the analogy to the smoke above a city was great as '
              'well. - Is the world deterministic?   - What is statistics?   the '
              'difference?   What is intelligence?   What is the real '
              'intelligence?',
   'end': 67,
   'score': 0.10144105513980461,
   'start': 38}
```

```python
pprint(qaam.answer("What is the name of this straight forward person?"))
...
  Converting examples to features: 100%|██████████| 1/1 [00:00<00:00, 272.16it/s]

  {'answer': 'Michael Jordan',
   'context': '- 1:34:36 LF: what would the creation of AI take and how that '
              'would be different from human intelligence? - Is the world '
              'deterministic?   Lex this is quickly becoming one of the best '
              'podcasts on the interwebs.   - What is statistics?   - What is '
              'intelligence?   Wow, this guy is the Michael Jordan of AI.   then '
              'what’s the difference?  ',
   'end': 297,
   'score': 0.9536337651297018,
   'start': 283}
```

```python
pprint(qaam.answer("What do people love about Lex and this video"))
...
  Converting examples to features: 100%|██████████| 1/1 [00:00<00:00, 214.49it/s]

  {'answer': 'different perception and perspective about AI.',
   'context': "I've heard Elon Musk talk about Neurallace, and he knows how much "
              "we don't know about the brain. One too many times I've heard this "
              'same talk about how Musk won\'t do something and "Young people '
              'get into this field and think its all done because we have '
              'tensorflow" Don\'t do that Love his different perception and '
              'perspective about AI. What about singularity ?!   and I love '
              "that! you'll see that Skill is what it is all about.",
   'end': 334,
   'score': 0.05500556350229058,
   'start': 288}
```

```python
pprint(qaam.answer("why do people love michael jordan?"))

  Converting examples to features: 100%|██████████| 1/1 [00:00<00:00, 309.66it/s]


  {'answer': 'Dr Jordan doesn’t allow popular terms that in reality don’t exist.',
   'context': 'Michael Jordan naturally speaks at 1.25x speed  DNA? Dr Jordan '
              'doesn’t allow popular terms that in reality don’t exist. @Ian '
              'Drake why dont you read some automl papers and see for yourself? '
              "Does science progress by ideas or personalities?   Don't do that "
              'Where did these thoughts originate from? Does he always though?',
   'end': 119,
   'score': 0.1176816181595175,
   'start': 53}
```
