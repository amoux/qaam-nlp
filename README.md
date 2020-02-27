# QAAM (Question Answering Auto Model)

- **Automatic question answering from any text source**

`QAAM` is a question and answering engine for answering questions of any text documents or texts extracted from a website's URL (see below). The model leverages a fine-tuned representation on *SQuAD* from the `Transformers` library while the context treated by proper tokenization techniques for online text.

## Notes

- document similarity experimentation:
  - fast sequencing intersection via skip-pointers

```python
def intersect_with_skips(p1, p2):
  '''
  doc_id<Dict[int, str]> -> int:
  has_skip<Callable[bool: Any]> -> bool: For an intermediate result
    in a complex query, the call has_skip(p) will always return false.
  '''
  answer = []
  while (p1 and p2) is not None:
    if doc_id(p1) == doc_id(p2):
      answer.append(doc_id(p1))
    elif doc_id(p1) < doc_id(p2):
      if has_skip(p1) and (doc_id(skip(p1)) <= doc_id(p2)):
        while has_skip(p1) and (doc_id(skip(p1)) <= doc_id(p2)):
          p1 = skip(p1)
        else:
          p1 = next(p1)
      elif has_skip(p2) and (doc(skip(p2)) <= doc_id(p1)):
        while has_skip(p2) and (doc_id(skip(p2)) <= doc_id(p1)):
          p2 = skip(p2)
        else:
          p2 = next(p2)
  return answer
```

## Installation

- To use the model install the required dependencies:

```bash
autocorrect==0.4.4
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

qaam = QAAM(threshold=0.2, summarize=True)
wiki_url = "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)"
qaam.texts_from_url(wiki_url) # texts are processed automatically
qaam.common_entities(top_k=10)
...
[('Transformer', 6),
 ('Transformers', 5),
 ('NLP', 3),
 ('first', 3),
 ('BERT', 2),
 ('GPT-2', 2),
 ('two', 2),
 ('2017', 1),
 ('three', 1),
 ('XLNet', 1)]
```

How to query questions from a website's text content:

```python
from pprint import pprint

question = "What is the purpose of Transformers models?"
answer = qaam.answer(question)
pprint(answer)
...
{'answer': 'to use a set of encodings to incorporate context into a '
           'sequence.[11]',
 'context': 'Pretraining is typically done on a much larger dataset than '
            'fine-tuning, due to the restricted availability of labeled '
            'training data. The Transformer is a deep machine learning model '
            'introduced in 2017, used primarily in the field of natural '
            'language processing ( So, if the data in question is natural '
            'language, the Transformer does not need to process the beginning '
            'of a sentence before it processes the end. ] The purpose of an '
            'attention mechanism is to use a set of encodings to incorporate '
            'context into a sequence.[11]',
 'end': 522,
 'score': 0.4509814318797609,
 'start': 454}
```

How does the model improve results better than other more complex methods? Simply, by adjusting the input (question) to the context of it's environment (website's texts). In short, the model will properly accommodate any query to the environment's vocabulary.

```python
question = "why it doesn't need to process the beginning of the sentence?"
prediction = qaam.answer(question)
print(prediction["answer"])
...
'the output sequence must be partially masked to prevent this reverse information flow.[1]'
```

```python
# the threshold and top_k parameters control how much the model can "see" from all texts
qaam.answer("what prevents the information flow?", top_k=20)
...
{'answer': 'a cell state which only passes through linear operations',
 'context': 'LSTMs make use of a cell state which only passes through linear'
            'operations in the recurrent portion, allowing information to pass'
            'through relatively unchanged with each iteration.[6]..'}
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
comments = []   # for this demo we ony get the texts from the comment items
for comment in tqdm(interbranch, desc="comments", unit=""):
    comments.append(comment["text"])  # available keys: {text, time, author, cid}
...
comments: 150 [00:08, 17.88/s]
```

```python
# add the comments to the model
qaam.texts_from_doc(comments)
qaam.common_entities(10, lower=True, lemmatize=True)
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
from pprint import pprint
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
