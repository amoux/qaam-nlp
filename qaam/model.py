from typing import List

import cupy
import numpy as np
import scipy
import spacy
from autocorrect import Speller
from david.text import (extract_text_from_url, normalize_whitespace,
                        unicode_to_ascii)
from david.text.summarization import summarizer as text_summarizer
from david.tokenizers import Tokenizer
from transformers import pipeline


class QAAM:
    """Question Answering Auto Model."""

    def __init__(
        self,
        threshold=0.1,
        summarize=True,
        lemmatize=True,
        model_name="question-answering",
        spacy_model="en_core_web_sm",
    ):
        """Initialize a question answering instance.

        `threshold`: The threshold confidence for document similarity.
        `summarize`: Whether to summarize long phrases (usefull if using high top_K).
        `lemmatize`: Whether to use lemmatization for the vocabulary (this does not
            affect the self.document instance property - used only for tokenization).
            It is recommened to use the default settings for more accurate results.
        """
        self.threshold = threshold
        self.summarize = summarize
        self.lemmatize = lemmatize
        self.qa_model = pipeline(model_name)
        self.nlp_model = spacy.load(spacy_model)
        self.tokenizer = None
        self.speller = None
        self.document = None
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
        document = self.document
        if self.lemmatize:
            document = self._lemmatize_document(document)

        tokenizer = Tokenizer(document=document)
        tokenizer.fit_vocabulary(mincount=1)
        sequences = tokenizer.document_to_sequences(document=document)
        self.vocab_matrix = tokenizer.sequences_to_matrix(sequences, "tfidf")

        speller = Speller(lang="en")
        for token, count in tokenizer.index_vocab.items():
            if token not in speller.nlp_data:
                speller.nlp_data[token] = count

        self.tokenizer = tokenizer
        self.speller = speller
        self._is_enviroment_vocabulary_ready = True

    def _build_context_paragraph(self, query: str, top_k: int):
        if self.lemmatize:
            query = self._lemmatize_document([query])[0]
        # encode the query as a vector and compute its dot product with the vocab-matrix
        embedd_query = self.tokenizer.convert_string_to_ids(query)
        matrix_query = self.tokenizer.sequences_to_matrix([embedd_query], "tfidf")

        matrix = self.vocab_matrix
        score = np.sum(matrix * matrix_query, axis=1) / np.linalg.norm(matrix, axis=1)
        top_indices = np.argsort(score)[::-1][:top_k]

        sim_doc = []
        for index in top_indices:
            k = score[index]
            if k >= self.threshold:
                text = self.document[index]
                sim_doc.append(text)

        sim_texts = " ".join(sim_doc)
        paragraph = sim_texts.replace("\n\n", " ").replace("\n", " ").strip()
        return paragraph

    def _build_answer(self, question: str, top_k: int):
        # clean and fix spelling (if any) and fetch the top similar to the question.
        question = self.speller(normalize_whitespace(unicode_to_ascii(question)))
        context = self._build_context_paragraph(question, top_k=top_k)

        if self.summarize and len(context) >= 100:
            context = text_summarizer(context)

        answer = self.qa_model({"question": question, "context": context})
        answer["context"] = context  # add the context to the answer dict.
        return answer

    def _preprocess_texts(self, texts: str):
        texts = normalize_whitespace(unicode_to_ascii(texts))
        document = []
        doc = self.nlp_model(texts)
        for sent in doc.sents:
            if sent.text is not None:
                document.append(sent.text)
        self.document = document

    def texts_from_url(self, url: str):
        """Extract texts from an URL link."""
        self._preprocess_texts(extract_text_from_url(url))

    def texts_from_str(self, sequences: str):
        """Load texts from an string sequences."""
        self._preprocess_texts(sequences)

    def texts_from_doc(self, document: List[str]):
        """Load texts from an iterable list of string sequences."""
        document = []
        for doc in self.nlp_model.pipe(document):
            for sent in doc.sents:
                text = normalize_whitespace(unicode_to_ascii(sent.text))
                if text is None:
                    continue
                document.append(text)
        self.document = document

    def common_entities(self, k: int = None, lower=False, lemmatize=False):
        """Return the most common entities from the document."""
        document = self.document
        if lemmatize:
            document = self._lemmatize_document(document)

        entities = {}
        for doc in self.nlp_model.pipe(document):
            for ent in doc.ents:
                ent = ent.text if not lower else ent.text.lower()
                if ent not in entities:
                    entities[ent] = 1
                else:
                    entities[ent] += 1

        entities = sorted(entities.items(), key=lambda k: k[1], reverse=True)
        if k is not None:
            return entities[:k]
        return entities

    def answer(self, question: str, top_k=10):
        """Return an answer based on the question to the text context."""
        if not self._is_enviroment_vocabulary_ready:
            self._build_enviroment_vocabulary()
        return self._build_answer(question, top_k=top_k)

    def to_tensor(self, sequence: str):
        """Convert a string sequence to a tensor based on spaCy's nlp model vocab.

        Usage:
            >>> tensor_a = to_tensor("words are not in vocabulary")
            >>> tensor_b = to_tensor("words are not in vocabulary")
            >>> self.cosine_similarity(tensor_a, tensor_b)
                1.0
        """
        tensor = self.nlp_model(sequence).tensor.sum(axis=0)
        return cupy.asnumpy(tensor)

    def to_matrix(self, sequence: str):
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
        return 1 - scipy.spatial.distance.cosine(array_a, array_b)
