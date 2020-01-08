import os
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import cupy
import requests
import scipy
import spacy
from autocorrect import Speller
from david import (SimilarDocuments, extract_text_from_url,
                   normalize_whitespace, preprocess_sequence, unicode_to_ascii)
from nptyping import Array

Response = TypeVar('Response', Callable, requests.Response)


class QAAM(SimilarDocuments):

    def __init__(self, top_k=10, ngram=(1, 3), feature="tfidf", model="en_core_web_sm"):
        super()
        self.top_k = top_k
        self.ngram = ngram
        self.feature = feature
        self.nlp = spacy.load(model)
        self.server_url = "http://localhost:5000/model/predict"
        self.threshold = 0.1
        self.spell = Speller(lang="en")
        self.doc = None
        self.queries = []
        self.raw_doc = []

    def _build_paragraph(self, query: str) -> str:
        self.learn_vocab()

        sim_doc = []
        for doc in self.iter_similar(self.top_k, query, True):
            if doc["sim"] > self.threshold:
                sim_doc.append(doc["text"])

        sim_texts = " ".join(sim_doc)
        paragraph = sim_texts.replace("\n\n", " ").replace("\n", " ").strip()
        return paragraph

    def _load_context(self, question: str) -> Tuple[str, str]:
        # fix user's spelling before preprocessing the question.
        question = self.spell(question)
        query = preprocess_sequence(
            question, lemmatize=False, rm_stopwords=False)
        paragraph = self._build_paragraph(query)
        response = self.max_model(question, paragraph)
        answer = response.json()
        if "ok" in answer.values():
            answer = answer["predictions"][0][0]
        return answer, paragraph

    def add_url(self, url: str) -> None:
        """Adds a the url where the model is served."""
        self.server_url = "{}/model/predict".format(url)

    def max_model(self, questions: Union[str, List[str]], context: str) -> Response:
        """Loads the question and context to the MAXQ Server Model.

        This method assumes both; the context has been properly formated
        and the server model's endpoint url is configured `server_url`.

        Parameters:
        ----------

        `questions` (Union[str, List[str]]):
            Pass a single (`str`) question or a (`List[str]`) of multiple questions.

        `context` (str):
            The context the model will use to answer the question(s).
            Note the max number of characters is `~1000`.

        Usage:
            >>> context = ("Self is merely a conventional name "
                           "for the first argument of a method.")
            >>> question = "What is self?"
            >>> answer = qaam.max_model(question, context).json()
            >>> print(answer['predictions'][0][0])
            'a conventional name for the first argument of a method'

        Returns -> (Response):
            A response object, obtain the results with `response.json()`.

        """
        if isinstance(questions, str):
            questions = [questions]
        return requests.post(self.server_url, json={
            "paragraphs": [{"context": context, "questions": questions}]})

    def texts_from_url(self, url: str) -> None:
        """Extracts all available text from a website.

        Parameters:
        ----------

        `url` (str): A website's full url where the text will be extracted.

        Returns -> (None): The extracted text is preprocessed and converted
            to sentences for the context in a session.

        """
        texts = extract_text_from_url(url)
        texts = unicode_to_ascii(texts)
        texts = normalize_whitespace(texts)
        self.doc = self.nlp(texts)
        # sentences is used by the SimilarDocuments class.
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
        similarity = 1 - scipy.spatial.distance(embedd_a, embedd_b)
        return similarity
