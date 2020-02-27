# qaam (Question Answering Auto Model)

- **Automatic question answering from any text source**

`qaam` is a question and answering engine for answering questions from any text document or text from a given website URL. The model leverages a fine-tuned model on SQuAD from the `transformers` library. While the context is handled by special tokenization techniques for online text and `Tfidf` vectorization scoring.

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
