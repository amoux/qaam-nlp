# qaam-nlp

`qaam-nlp` is an question and answering api for any text content that can be extracted from the web.

**usage:**

> Extract texts form a website e.g., blog or article and loading preprocessing all the text content by simply using the `david.text_from_url()` instance method:

- First, install the package: `pip install .` and import the `QAAM` class. Below are some of the default parameters that can be configured to accommodate a document's environment. The `server_url` parameter is the only parameter that you need to configure - which is the url endpoint where the model is being served.

```python
from qaam import QAAM
qaam = QAAM(top_k=20, feature="tfidf", ngram=(1,3), server_url="http://server.com")
qaam.texts_from_url("https://www.entrepreneur.com/article/241026")

# All the content has been loaded.
qaam.common_entities()
...
[('first', 2),
 ('One', 2),
 ('the day', 1),
 ('one', 1),
 ('third', 1),
 ('Cover Entrepreneur Media, Inc.', 1),
 ('Privacy Policy', 1)]
```

> How to query questions from a website's blog text content:

```python
from pprint import pprint
question = "What is business property insurance?"
prediction = qaam.answer(question)
pprint(prediction)
...
{'answer': 'Whether a business owns or leases its space',
 'context': 'Whether a business owns or leases its space, property insurance '
            'is a must. Related: Business Interruption Insurance: What It Will '
            "-- and Won't Unfortunately, homeowner’s policies don’t cover "
            'home-based businesses in the way commercial property insurance '
            'does. However, mass-destruction events like floods and '
            'earthquakes are generally not covered under standard property '
            'insurance policies.'}
```

> How does the model improve results better than other more complex methods? Simply, by adjusting the input (question) to the context (paragraph) of the environment (website's text content). In short, the model will properly accommodate any query to the environment's vocabulary.

```python
# Let's see how the incorrect question is fixed.
question = "Why there's no one-siez-fits-all polucy?"
prediction = qaam.answer(question)

# What the QA model actually 'sees'.
print(qaam.history["spelling"])
...
["Why there's no one-size-fits-all policy?"]

# What cosine similarly actually 'sees'
print(qaam.history["query"])
["Why there 's no one-size-fits-all policy"]
```
