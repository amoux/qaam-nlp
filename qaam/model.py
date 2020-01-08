import json
import os
import tarfile
from contextlib import closing
from typing import (IO, Callable, Dict, List, NoReturn, Optional, Tuple,
                    TypeVar, Union)

import cupy
import requests
import scipy
import spacy
from autocorrect import Speller
from autocorrect.word_count import count_words
from david import (SimilarDocuments, extract_text_from_url,
                   normalize_whitespace, remove_punctuation, unicode_to_ascii)
from nptyping import Array

Response = TypeVar('Response', Callable, requests.Response)

HISTORY = {'question': [], 'spelling': [], 'query': [], 'answer': [], 'context': []}


class QAAM(SimilarDocuments):
    spell_basepath = "context"
    spell_nlp_file = "en.qaam.tar.gz"
    spell_vocab_file = "vocab.txt"
    spell_words_file = "words.json"

    def __init__(self, top_k=10, ngram=(1, 3), threshold=0.1, feature="tfidf", model="en_core_web_sm", speller_lang="en"):
        super()
        self.top_k = top_k
        self.ngram = ngram
        self.feature = feature
        self.nlp = spacy.load(model)
        self.server_url = "http://localhost:5000/model/predict"
        self.threshold = threshold
        self.doc = None
        self.queries = []
        self.raw_doc = []
        self.history = HISTORY
        self.spell = Speller(lang=speller_lang)
        self._spell_vocab_fp = self._load_path(self.spell_vocab_file)
        self._spell_words_fp = self._load_path(self.spell_words_file)
        self._spell_nlp_fp = self._load_path(self.spell_nlp_file)

    def _load_path(self, filename: str) -> NoReturn:
        return os.path.join(self.spell_basepath, filename)

    def _save_spelling_context(self, lang: str = "en") -> IO:
        # Saves the necessary data sourcer for the Speller context.
        if not os.path.isdir(self.spell_basepath):
            os.makedirs(self.spell_basepath)
        with open(self._spell_vocab_fp, mode="w") as file:
            for word in self.vectorizer.vocabulary_:
                file.write("%s\n" % word)

        # save the speller data files to a temp file.
        count_words(self._spell_vocab_fp, lang, self._spell_words_fp)
        nlp_tarfile = tarfile.open(self._spell_nlp_fp, "w:gz")
        nlp_tarfile.add(self._spell_words_fp)
        nlp_tarfile.close()

    def _load_spelling_context(self, lang: str = "en") -> Dict[str, int]:
        self._save_spelling_context(lang=lang)
        with closing(tarfile.open(self._spell_nlp_fp, 'r:gz')) as tarf:
            with closing(tarf.extractfile(self._spell_words_fp)) as file:
                return json.load(file)

    def _build_paragraph(self, query: str) -> str:
        sim_doc = []
        for doc in self.iter_similar(self.top_k, query, True):
            if doc["sim"] > self.threshold:
                sim_doc.append(doc["text"])
        sim_texts = " ".join(sim_doc)
        paragraph = sim_texts.replace("\n\n", " ").replace("\n", " ").strip()
        return paragraph

    def _load_context(self, question: str) -> Tuple[str, str]:
        self.history["question"].append(question)

        # build and initialize the vocabulary from the url's texts.
        self.learn_vocab()
        self.spell.nlp_data.update(self._load_spelling_context())
        question = self.spell(question)
        question_as_query = remove_punctuation(question)
        paragraph = self._build_paragraph(question_as_query)

        self.history["spelling"].append(question)
        self.history["query"].append(question_as_query)
        self.history["context"].append(paragraph)

        # fetch the question and context paragraph to the maxq model.
        response = self.max_model(question, paragraph)
        answer = response.json()
        if "ok" in answer.values():
            answer = answer["predictions"][0][0]
            self.history["answer"].append(answer)
        return answer, paragraph

    def add_url(self, url: str) -> NoReturn:
        """Adds a the url where the model is served."""
        self.server_url = "{}/model/predict".format(url)

    def max_model(self, questions: Union[str, List[str]], context: str) -> Response:
        """Loads the question and context to the MAXQ Server Model.

        This method assumes both; the context has been properly formated
        and the server model's endpoint url is configured `server_url`.

        Parameters:
        ----------

        `questions` (Union[str, List[str]]):
            Pass a single string question or an iterable of multiple questions.

        `context` (str):
            The context the model will use to answer the question(s). Note the max
            number of characters is `~1000`.

        Usage:
            >>> context = ("Self is merely a conventional name "
                           "for the first argument of a method.")
            >>> question = "What is self?"
            >>> answer = qaam.max_model(question, context).json()
            >>> print(answer['predictions'][0][0])
            'a conventional name for the first argument of a method'

        Returns (Response): A response object, obtain the results with `response.json()`.
        """
        if isinstance(questions, str):
            questions = [questions]
        return requests.post(self.server_url, json={
            "paragraphs": [{"context": context, "questions": questions}]})

    def texts_from_url(self, url: str) -> NoReturn:
        """Extracts all available text from a website.

        Parameters:
        ----------

        `url` (str): A website's full url where the text will be extracted.

        Returns -> (None): The extracted text is preprocessed and converted to 
            sentences for the context in a session.
        """
        texts = extract_text_from_url(url)
        texts = unicode_to_ascii(texts)
        texts = normalize_whitespace(texts)
        self.doc = self.nlp(texts)
        sentences = []
        for sent in self.doc.sents:
            if sent.text:
                sentences.append(sent.text)
        self.raw_doc.extend(sentences)
        del sentences

    def answer(self, question: str, top_k: Optional[int] = None) -> Dict[str, str]:
        """Answers any question related to the content from the website."""
        self.top_k = top_k if top_k else self.top_k
        answer, context = self._load_context(question)
        return dict(answer=answer, context=context)

    def embedd_sequence(self, sequence: str) -> Array:
        """Returns an embedded (Array) from a string sequence of tensors."""
        tensor = self.nlp(sequence).tensor.sum(axis=0)
        embedd = cupy.asnumpy(tensor)
        return embedd

    def cosine_distance(self, embedd_a: Array, embedd_b: Array) -> float:
        """Returns the similarity defined as 1 - cosine distance between two arrays."""
        similarity = 1 - scipy.spatial.distance.cosine(embedd_a, embedd_b)
        return similarity
