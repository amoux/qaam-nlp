import json
import os
import tarfile
from collections import Counter
from contextlib import closing
from typing import (IO, Callable, Dict, List, NewType, NoReturn, Optional,
                    Tuple, TypeVar, Union)

import cupy
import requests
import scipy
import spacy
from autocorrect import Speller
from autocorrect.word_count import count_words
from nptyping import Array
from spacy.tokens import Doc

from david.cosine import SimilarDocuments
from david.text.prep import (normalize_whitespace, remove_punctuation,
                             unicode_to_ascii)
from david.text.summarization import summarizer as text_summarizer
from david.text.utils import extract_text_from_url

HISTORY: Dict[str, List[str]] = {
    "question": [],
    "spelling": [],
    "query": [],
    "answer": [],
    "context": [],
}


class QAAM(SimilarDocuments):
    spell_basepath: str = "context"
    spell_nlp_file: str = "en.qaam.tar.gz"
    spell_vocab_file: str = "vocab.txt"
    spell_words_file: str = "words.json"

    def __init__(
        self,
        top_k: int = 10,
        ngram: Tuple[int, int] = (1, 3),
        threshold: float = 0.1,
        feature: str = "tfidf",
        model: str = "en_core_web_sm",
        speller_lang: str = "en",
        summarize: bool = False,
        server_url: str = "http://localhost:5000",
    ):
        super()
        self.top_k = top_k
        self.ngram = ngram
        self.feature = feature
        self.nlp = spacy.load(model)
        self.summarize = summarize
        self.server_url = f"{server_url}/model/predict".strip()
        self.threshold = threshold
        self.doc: Doc = None
        self.queries: List[str] = []
        self.raw_doc: List[str] = []
        self.history: Dict[str, List[str]] = HISTORY
        self.spell = Speller(lang=speller_lang)
        self._spell_vocab_fp: str = self._load_path(self.spell_vocab_file)
        self._spell_words_fp: str = self._load_path(self.spell_words_file)
        self._spell_nlp_fp: str = self._load_path(self.spell_nlp_file)
        self._is_env_context_ready: bool = False

    def _load_path(self, filename: str) -> str:
        return os.path.join(self.spell_basepath, filename)

    def _save_spelling_context(self, lang: str = "en") -> None:
        if not os.path.isdir(self.spell_basepath):
            os.makedirs(self.spell_basepath)

        # Saves the necessary data sourcer for the Speller context.
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
        with closing(tarfile.open(self._spell_nlp_fp, "r:gz")) as tarf:
            with closing(tarf.extractfile(self._spell_words_fp)) as file:
                return json.load(file)

    def _build_paragraph(self, query: str) -> str:
        # Finds all similar sentences given the query.
        sim_doc: List[str] = []
        for doc in self.iter_similar(self.top_k, query, True):
            if doc["sim"] > self.threshold:
                sim_doc.append(doc["text"])

        # Format all similar sentences to a "paragraph" format.
        sim_texts = " ".join(sim_doc)
        paragraph = sim_texts.replace("\n\n", " ").replace("\n", " ").strip()
        return paragraph

    def _load_env_context(self) -> None:
        # Initialize the vocabulary and context dependacies.
        self.learn_vocab()
        env_context_vocab = self._load_spelling_context()
        self.spell.nlp_data.update(env_context_vocab)
        self._is_env_context_ready = True

    def _build_answer(self, question: str) -> Tuple[str, str]:
        # Builds answer and context (paragraph) given the query.
        spelling = self.spell(question)
        query = remove_punctuation(spelling)
        context = self._build_paragraph(query)
        # len("word word word word") => 19 (not a positive result)
        if len(context) < 20:
            raise Exception(
                f"Invalid paragraph, got size {len(context)} "
                "most likely caused by the question (Not related to context)."
            )
        if self.summarize and len(context) > 100:
            paragraph = text_summarizer(context)

        self.history["question"].append(question)
        self.history["spelling"].append(spelling)
        self.history["query"].append(query)
        self.history["context"].append(context)

        # Fetch the question and context paragraph to the maxq model.
        response = self.max_model(spelling, context)
        if response.status_code == 200:
            answer = response.json()
            answer = answer["predictions"][0][0]
            self.history["answer"].append(answer)

        elif response.status_code == 400:
            pass
        else:
            raise Exception("Failed to communicate with the Server endpoint.")
        return answer, context

    def add_server_url(self, url: str) -> None:
        """Adds a the url where the model is served."""
        self.server_url = "{url}/model/predict".strip()

    def max_model(
        self, questions: Union[str, List[str]], context: str
    ) -> requests.Response:
        """Loads the question and context to the MAXQ Server Model."""
        if isinstance(questions, str):
            questions = [questions]
        return requests.post(
            self.server_url,
            json={"paragraphs": [{"context": context, "questions": questions}]},
        )

    def texts_from_url(self, url: str) -> None:
        """Extracts all available text from a website."""
        texts = extract_text_from_url(url)
        texts = unicode_to_ascii(texts)
        texts = normalize_whitespace(texts)

        self.doc = self.nlp(texts)
        sentences: List[str] = []
        for sent in self.doc.sents:
            if sent.text:
                sentences.append(sent.text)

        self.raw_doc.extend(sentences)
        self._is_env_context_ready = False
        del sentences

    def texts_from_string(self, sequences, preprocess: bool = False) -> None:
        """Loads texts from a string object of string sequences."""
        if preprocess:
            sequences = unicode_to_ascii(sequences)
            sequences = normalize_whitespace(sequences)

        self.doc = self.nlp(sequences)
        sentences: List[str] = []
        for sent in self.doc.sents:
            if sent.text:
                sentences.append(sent.text)

        self.raw_doc.extend(sentences)
        self._is_env_context_ready = False
        del sentences

    def texts_from_doc(self, document: List[str], preprocess: bool = False) -> None:
        """Loads text documents of string sequences from a list type object."""
        if preprocess:
            texts: List[str] = []
            for line in document:
                texts.append(normalize_whitespace(unicode_to_ascii(line)))
            self.doc = self.nlp(" ".join(texts))
            del texts
        else:
            self.doc = self.nlp(" ".join(document))

        sentences: List[str] = []
        for sent in self.doc.sents:
            if sent.text:
                sentences.append(sent.text)

        self.raw_doc.extend(sentences)
        self._is_env_context_ready = False
        del sentences

    def answer(self, question: str, top_k: Optional[int] = None) -> Dict[str, str]:
        """Answers any question related to the content from the website."""
        if not isinstance(question, str):
            raise Exception(
                f"Invalid type, question needs be of string type not {type(question)}."
            )
        if len(question.split()) < 4:
            raise Exception(
                f"{len(question.split())} words is not decent question size"
            )

        self.top_k = top_k if top_k else self.top_k
        # Builds the context and vocabulary only if it hasn't been initialized.
        if not self._is_env_context_ready:
            self._load_env_context()

        answer, context = self._build_answer(question)
        return dict(answer=answer, context=context)

    def common_entities(
        self, top_k: int = 10, lower: bool = False
    ) -> List[Tuple[str, int]]:
        """Returns the most common entities from the document."""
        entities: Dict[str, int] = dict()
        for ent in self.doc.ents:
            ent = ent.text if not lower else ent.text.lower()
            if ent not in entities:
                entities[ent] = 1
            else:
                entities[ent] += 1
        return Counter(entities).most_common(top_k)

    def embedd_sequence(self, sequence: str) -> Array:
        """Returns an embedded (Array) from a string sequence of tensors."""
        tensor = self.nlp(sequence).tensor.sum(axis=0)
        embedd = cupy.asnumpy(tensor)
        return embedd

    def cosine_distance(self, embedd_a: Array, embedd_b: Array) -> float:
        """Returns the similarity defined as 1 - cosine distance between two arrays."""
        similarity = 1 - scipy.spatial.distance.cosine(embedd_a, embedd_b)
        return similarity
