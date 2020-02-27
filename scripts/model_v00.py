from typing import Any, Dict, List, Optional, TypeVar, Union

import requests
import spacy

from david import (
    SimilarDocuments,
    extract_text_from_url,
    normalize_whitespace,
    preprocess_sequence,
    unicode_to_ascii,
)

Response = TypeVar("Response", object, requests.Response)


class MAXQAModel(object):
    CONTEXT = None

    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url
        if server_url is not None:
            self.add_url(server_url)

    def add_url(self, url: str) -> None:
        server_endpoint = "{}/model/predict"
        self.server_url = server_endpoint.format(url.strip())

    def load_context(
        self,
        questions: Optional[Union[str, List[str]]],
        context: Optional[str] = None,
        response_only: bool = False,
    ) -> Union[Response, Dict[str, str]]:
        """Loads the question and context to the MAXQ Server Model.

        This method assumes both; the context and server url have been
        configured. You can load/access the context parameter from CONTEXT
        or passing the string context to the `context` argument.

        Arguments:
            question {Optional[Union[str, List[str]]]}:
                Pass a single question or a list of multiple questions.

            context {Optional[str]} (default: {None}):
                If context left as None, then it will try self.CONTEXT

            response_only {bool} (default: {None}):
                If true, return only the response, you will have to call
                `response.json()` to obtain the values from the server.
                Otherwise Both the answer and context will be returned.

        Returns -> {Union[Response, Dict[str, str]]}:
        """
        if isinstance(questions, str):
            questions = list(questions)
        context = context if context else self.CONTEXT
        response = requests.post(
            self.server_url,
            json={"paragraphs": [{"context": context, "questions": questions}]},
        )
        if response_only:
            return response
        answer = response.json()
        if "ok" in answer.values():
            answer = answer["predictions"][0][0]
        return answer, context


class QAAM(MAXQAModel, SimilarDocuments):
    def __init__(self, *args, **kwargs):
        super(QAAM, self).__init__()
        self.top_k = 10
        self.ngram = (1, 3)
        self.feature = "tfidf"
        self.spacy_doc = None
        self.threshold = 0.1
        self.queries = []

    def _build_paragraph(self, query: str) -> str:
        self.learn_vocab()
        # finally, build the paragraph from the most similar documents.
        simdocs = []
        for doc in self.iter_similar(self.top_k, query, True):
            if doc["sim"] > self.threshold:
                simdocs.append(doc["text"])
        simtexts = " ".join(simdocs)
        paragraph = simtexts.replace("\n\n", " ").replace("\n", " ").strip()
        return paragraph

    def load_url_texts(self, url: str) -> None:
        texts = extract_text_from_url(url)
        texts = unicode_to_ascii(texts)
        texts = normalize_whitespace(texts)
        nlp = spacy.load("en_core_web_sm")
        self.spacy_doc = nlp(texts)
        raw_doc = []
        for i in self.spacy_doc.sents:
            if i.text:
                raw_doc.append(i.text)
        self.raw_doc = list()
        self.raw_doc.extend(raw_doc)
        del raw_doc

    def answer(self, question: str, top_k: int = None):
        """Request for an answer to the MAXQ model.
        """
        self.top_k = top_k if top_k else self.top_k
        query = preprocess_sequence(question, lemmatize=False, rm_stopwords=False)
        context = self._build_paragraph(query)
        response = self.load_context(question, context, response_only=True)
        answer = response.json()
        if "ok" in answer.values():
            answer = answer["predictions"][0][0]
        try:
            context = context.replace(answer, f"<Answer( {answer} )>")
        except TypeError:
            message = "Try asking questions related to the content."
            return dict(answer=message, context="<NOT A VALID QUESTION>")
        return dict(answer=answer, context=context)


def extract_sentence(target, text):
    if not isinstance(text, str) or len(text.strip()) < 5:
        return text
    for sent in text.split("."):
        if target in sent:
            x = sent.find(target)
        return f"{sent[:x]} {sent[x:]}".strip()
