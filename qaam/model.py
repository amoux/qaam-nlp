# -*- coding: utf-8 -*-

from __future__ import division, unicode_literals

from typing import Dict, List, NamedTuple, Tuple, Union

import cupy
import numpy as np
import spacy
from david.lang import Speller
from david.lang.en_lexicon import DAVID_STOP_WORDS
from david.text import extract_text_from_url, normalize_whitespace, unicode_to_ascii
from david.text.summarization import summarizer as text_summarizer
from david.tokenizers import Tokenizer
from scipy import spatial
from transformers import pipeline


class AutoModelPrediction(NamedTuple):
    """Dictionary with the model's prediction."""

    answer: str
    context: str
    score: float
    start: int
    end: int
    question: str


class QAAM:
    """Question Answering Auto Model."""

    def __init__(
        self,
        threshold=0.1,
        summarize=True,
        lemmatize=True,
        metric="cosine",
        model_name="question-answering",
        spacy_model="en_core_web_sm",
    ):
        """Initialize a question answering instance.

        `threshold`: The threshold confidence for document similarity.
        `summarize`: Whether to summarize long phrases (usefull if using high top_K).
        `lemmatize`: Whether to use lemmatization for the vocabulary (this does not
            affect the self.document instance property - used only for tokenization).
            It is recommened to use the default settings for more accurate results.
        `metric`: The distance metric to use for building the context.
        """
        self.threshold = threshold
        self.summarize = summarize
        self.lemmatize = lemmatize
        self.metric = metric
        self.qa_model = pipeline(model_name)
        self.nlp_model = spacy.load(spacy_model)
        self.tokenizer = None
        self.speller = None
        self.document: List[str] = []
        self.vocab_matrix = None
        self._is_enviroment_vocabulary_ready = False

    def _lemmatize_document(self, document: List[str]) -> List[str]:
        # used to improve document similarity scores. the tokenizer
        # uses/maintains the lemmatized tokens internally which means
        # the self.document property and the model's answers and context
        # are not returned in lemmatized form.
        lemma_doc = []
        for doc in self.nlp_model.pipe(document):
            sent = " ".join([token.lemma_ for token in doc])
            lemma_doc.append(sent)
        return lemma_doc

    def _build_enviroment_vocabulary(self):
        # create an instance of the document's vocabulary.
        document = self.document

        # construct a speller with the document instance
        # adding missing stop-words for spell correction
        speller = Speller(document=document)
        for word in set(DAVID_STOP_WORDS):
            word = word.lower()
            if len(word) > 3 and word not in speller.word_count:
                speller.word_count[word] = 1
            else:
                speller.word_count[word] += 1

        if self.lemmatize:
            document = self._lemmatize_document(document)

        # setup the tokenizer and the vocabulary
        tokenizer = Tokenizer(document=document)
        tokenizer.fit_vocabulary(mincount=1)
        sequences = tokenizer.document_to_sequences(document=document)

        self.vocab_matrix = tokenizer.sequences_to_matrix(sequences, "tfidf")
        self.tokenizer = tokenizer
        self.speller = speller
        self._is_enviroment_vocabulary_ready = True

    def similar_documents(self, query: str, top_k: int, return_score=True):
        """Compute distance between each pair of the two collections of inputs."""
        if self.lemmatize:
            query = self._lemmatize_document([query])[0]

        query_sequence = self.tokenizer.convert_string_to_ids(query)
        query_matrix = self.tokenizer.sequences_to_matrix([query_sequence], "tfidf")

        similar: List[Union[Tuple[str, float], str]] = []
        for q, qx in zip([query], query_matrix):
            distances = spatial.distance.cdist([qx], self.vocab_matrix, self.metric)
            distances = zip(range(len(distances[0])), distances[0])
            distances = sorted(distances, key=lambda k: k[1])
            for index, distance in distances[0:top_k]:
                k = 1 - distance
                if k >= self.threshold:
                    doc = self.document[index]
                    if return_score:
                        similar.append((doc, k))
                    else:
                        similar.append(doc)

        return similar

    def _build_answer(self, question: str, top_k: int) -> AutoModelPrediction:
        # fix (if needed) the question's grammar in relation to the context
        question = self.speller.correct_string(question)

        # convert the question to query form and build the context
        query = " ".join(self.speller.tokenize(question))
        documents = self.similar_documents(query, top_k, return_score=False)
        paragraph = " ".join(documents)
        context = paragraph.replace("\n\n", " ").replace("\n", " ").strip()

        # summarize the context if larger than N characters (including spaces)
        if self.summarize and len(context) >= 80:
            context = text_summarizer(context)

        # pass the corrected question and the context to the Transformers model
        prediction = self.qa_model({"question": question, "context": context})
        prediction["context"] = context
        prediction["question"] = question
        return prediction

    def _preprocess_texts(self, texts: str):
        texts = normalize_whitespace(unicode_to_ascii(texts))
        document = []
        doc = self.nlp_model(texts)
        for sent in doc.sents:
            if sent.text is not None:
                document.append(sent.text)

        self.document = document
        self._is_enviroment_vocabulary_ready = False

    def texts_from_url(self, url: str):
        """Extract texts from an URL link."""
        self._preprocess_texts(extract_text_from_url(url))

    def texts_from_str(self, sequences: str):
        """Load texts from an string sequences."""
        self._preprocess_texts(sequences)

    def texts_from_doc(self, document: List[str]):
        """Load texts from an iterable list of string sequences."""
        sentences = []
        for doc in self.nlp_model.pipe(document):
            for sent in doc.sents:
                text = normalize_whitespace(unicode_to_ascii(sent.text))
                if text is not None:
                    sentences.append(text)

        self.document = sentences
        self._is_enviroment_vocabulary_ready = False

    def common_entities(
        self, k: int = None, lower=False, lemma=False) -> List[Tuple[str, int]]:
        """Return the most common entities from the document."""
        document = self.document
        if lemma:
            document = self._lemmatize_document(document)

        common: Dict[str, int] = {}
        for doc in self.nlp_model.pipe(document):
            for ent in doc.ents:
                ent = ent.text if not lower else ent.text.lower()
                if ent not in common:
                    common[ent] = 1
                else:
                    common[ent] += 1

        entities = sorted(common.items(), key=lambda k: k[1], reverse=True)
        if k is not None:
            return entities[:k]
        return entities

    def answer(self, question: str, top_k=10) -> AutoModelPrediction:
        """Return an answer based on the question to the text context."""
        if not self._is_enviroment_vocabulary_ready:
            self._build_enviroment_vocabulary()
        return self._build_answer(question, top_k=top_k)

    def to_tensor(self, sequence: str) -> float:
        """Convert a string sequence to a tensor based on spaCy's nlp model vocab.

        Usage:
            >>> tensor_a = to_tensor("words are not in vocabulary")
            >>> tensor_b = to_tensor("words are not in vocabulary")
            >>> self.cosine_similarity(tensor_a, tensor_b)
                1.0
        """
        tensor = self.nlp_model(sequence).tensor.sum(axis=0)
        return cupy.asnumpy(tensor)

    def to_matrix(self, sequence: str) -> float:
        """Convert a string sequence to a matrix based on the vocab from the context.
        
        Usage:
            >>> matrix_a = to_matrix("words are in the vocabulary")
            >>> matrix_b = to_matrix("words are in the vocabulary")
            >>> self.cosine_similarity(matrix_a, matrix_b)
                1.0
        """
        if self.lemmatize:
            sequence = self._lemmatize_document([sequence])[0]

        embedd = self.tokenizer.convert_string_to_ids(sequence)
        vector = self.tokenizer.sequences_to_matrix([embedd], mode="tfidf")
        return vector.sum(axis=0)

    def cosine_similarity(self, array_a, array_b):
        """Compute the cosine distance of two arrays.

        arrays can be obtained from either `self.to_matrix` or `self.to_tensor`
        """
        return 1 - spatial.distance.cosine(array_a, array_b)
