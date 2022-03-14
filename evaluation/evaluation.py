import json
import time
import itertools
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from typing import Mapping, Any, List, Tuple

try:
    from bertopic import BERTopic
except ImportError:
    pass

try:
    from top2vec import Top2Vec
except ImportError:
    pass

try:
    from contextualized_topic_models.models.ctm import CombinedTM
    from contextualized_topic_models.utils.data_preparation import (
        TopicModelDataPreparation,
    )
    import nltk

    nltk.download("stopwords")
    from nltk.corpus import stopwords
except ImportError:
    pass

from octis.models.ETM import ETM
from octis.models.LDA import LDA
from octis.models.NMF import NMF
from octis.models.CTM import CTM
from octis.dataset.dataset import Dataset
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

import gensim
import gensim.corpora as corpora
from gensim.models import ldaseqmodel


class Trainer:
    """Train and evaluate a topic model

    Arguments:
        dataset: The dataset to be used, should be a string and either a
                 dataset found in OCTIS or a custom dataset
        model_name: The name of the topic model to be used:
                        * BERTopic
                        * Top2Vec
                        * CTM_CUSTOM (original package)
                        * ETM (OCTIS)
                        * LDA (OCTIS)
                        * CTM (OCTIS)
                        * NMF (OCTIS)
        params: The parameters of the model to be used
        topk: The top n words in each topic to include
        custom_dataset: Whether a custom dataset is used
        bt_embeddings: Pre-trained embeddings used in BERTopic to speed
                       up training.
        bt_timestamps: Timestamps used in BERTopic for dynamic
                       topic modeling
        bt_nr_bins: Number of bins to create from timestamps in BERTopic
        custom_model: A custom BERTopic or Top2Vec class
        verbose: Control the verbosity of the trainer

    Usage:

    ```python
    from evaluation import Trainer
    dataset, custom = "20NewsGroup", False
    params = {"num_topics": [(i+1)*10 for i in range(5)], "random_state": 42}

    trainer = Trainer(dataset=dataset,
                      model_name="LDA",
                      params=params,
                      custom_dataset=custom,
                      verbose=True)
    ```

    Note that we need to specify whether a custom OCTIS dataset is used.
    Since we use a preprocessed dataset from OCTIS [here](https://github.com/MIND-Lab/OCTIS#available-datasets),
    no custom dataset is used.

    This trainer focused on iterating over all combinations of parameters in `params`.
    In the example above, we iterate over different number of topics.
    """

    def __init__(
        self,
        dataset: str,
        model_name: str,
        params: Mapping[str, Any],
        topk: int = 10,
        custom_dataset: bool = False,
        bt_embeddings: np.ndarray = None,
        bt_timestamps: List[str] = None,
        bt_nr_bins: int = None,
        custom_model=None,
        verbose: bool = True,
    ):
        self.dataset = dataset
        self.custom_dataset = custom_dataset
        self.model_name = model_name
        self.params = params
        self.topk = topk
        self.timestamps = bt_timestamps
        self.nr_bins = bt_nr_bins
        self.embeddings = bt_embeddings
        self.ctm_preprocessed_docs = None
        self.custom_model = custom_model
        self.verbose = verbose

        # Prepare data and metrics
        self.data = self.get_dataset()
        self.metrics = self.get_metrics()

        # CTM
        self.qt_ctm = None
        self.training_dataset_ctm = None

    def train(self, save: str = False) -> Mapping[str, Any]:
        """Train a topic model

        Arguments:
            save: The name of the file to save it to.
                  It will be saved as a .json in the current
                  working directory

        Usage:

        ```python
        from evaluation import Trainer
        dataset, custom = "20NewsGroup", False
        params = {"num_topics": [(i+1)*10 for i in range(5)], "random_state": 42}

        trainer = Trainer(dataset=dataset,
                        model_name="LDA",
                        params=params,
                        custom_dataset=custom,
                        verbose=True)
        results = trainer.train(save="LDA_results")
        ```
        """

        results = []

        # Loop over all parameters
        params_name = list(self.params.keys())
        params = {
            param: (value if type(value) == list else [value])
            for param, value in self.params.items()
        }
        new_params = list(itertools.product(*params.values()))
        for param_combo in new_params:

            # Train and evaluate model
            params_to_use = {
                param: value for param, value in zip(params_name, param_combo)
            }
            output, computation_time = self._train_tm_model(params_to_use)
            scores = self.evaluate(output)

            # Update results
            result = {
                "Dataset": self.dataset,
                "Dataset Size": len(self.data.get_corpus()),
                "Model": self.model_name,
                "Params": params_to_use,
                "Scores": scores,
                "Computation Time": computation_time,
            }
            results.append(result)

        if save:
            with open(f"{save}.json", "w") as f:
                json.dump(results, f)

            try:
                from google.colab import files

                files.download(f"{save}.json")
            except ImportError:
                pass

        return results

    def _train_tm_model(
        self, params: Mapping[str, Any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Select and train the Topic Model"""
        # Train custom CTM
        if self.model_name == "CTM_CUSTOM":
            if self.qt_ctm is None:
                self._preprocess_ctm()
            return self._train_ctm(params)

        # Train BERTopic
        elif self.model_name == "BERTopic":
            return self._train_bertopic(params)

        # Train Top2Vec
        elif self.model_name == "Top2Vec":
            return self._train_top2vec(params)

        # Train LDAseq
        elif self.model_name == "LDAseq":
            return self._train_ldaseq(params)

        # Train OCTIS model
        octis_models = ["ETM", "LDA", "CTM", "NMF"]
        if self.model_name in octis_models:
            return self._train_octis_model(params)

    def _train_ldaseq(
        self, params: Mapping[str, any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Train LDA seq model"""
        data = self.data.get_corpus()
        docs = [" ".join(words) for words in data]

        df = pd.DataFrame({"Doc": docs, "Timestamp": self.timestamps}).sort_values(
            "Timestamp"
        )
        df["Bins"] = pd.cut(df.Timestamp, bins=params["nr_bins"])
        df["Timestamp"] = df.apply(lambda row: row.Bins.left, 1)
        timestamps = df.groupby("Bins").count().Timestamp.values
        docs = df.Doc.values

        data_words = list(sent_to_words(docs))
        id2word = corpora.Dictionary(data_words)
        corpus = [id2word.doc2bow(text) for text in data_words]

        print(len(corpus), len(self.timestamps), timestamps)

        params["corpus"] = corpus
        params["id2word"] = id2word
        params["time_slice"] = timestamps
        del params["nr_bins"]

        start = time.time()
        ldaseq = ldaseqmodel.LdaSeqModel(**params)
        end = time.time()
        computation_time = end - start

        all_topics = {}
        for i in range(len(timestamps)):
            topics = ldaseq.print_topics(time=i)
            topics = [[word for word, _ in topic][:5] for topic in topics]
            all_topics[i] = {"topics": topics}

        return all_topics, computation_time

    def _train_top2vec(
        self, params: Mapping[str, Any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Train Top2Vec"""
        nr_topics = None
        data = self.data.get_corpus()
        data = [" ".join(words) for words in data]
        params["documents"] = data

        if params.get("nr_topics"):
            nr_topics = params["nr_topics"]
            del params["nr_topics"]

        start = time.time()

        if self.custom_model is not None:
            model = self.custom_model(**params)
        else:
            model = Top2Vec(**params)

        if nr_topics:
            try:
                _ = model.hierarchical_topic_reduction(nr_topics)
                params["reduction"] = True
                params["nr_topics"] = nr_topics
            except:
                params["reduction"] = False
                nr_topics = False

        end = time.time()
        computation_time = float(end - start)

        if nr_topics:
            topic_words, _, _ = model.get_topics(reduced=True)
        else:
            topic_words, _, _ = model.get_topics(reduced=False)

        topics_old = [list(topic[:10]) for topic in topic_words]
        all_words = [word for words in self.data.get_corpus() for word in words]
        topics = []
        for topic in topics_old:
            words = []
            for word in topic:
                if word in all_words:
                    words.append(word)
                else:
                    print(f"error: {word}")
                    words.append(all_words[0])
            topics.append(words)

        if not nr_topics:
            params["nr_topics"] = len(topics)
            params["reduction"] = False

        del params["documents"]
        output_tm = {
            "topics": topics,
        }
        return output_tm, computation_time

    def _train_ctm(self, params) -> Tuple[Mapping[str, Any], float]:
        """Train CTM"""
        params["bow_size"] = len(self.qt_ctm.vocab)
        ctm = CombinedTM(**params)

        start = time.time()
        ctm.fit(self.training_dataset_ctm)
        end = time.time()
        computation_time = float(end - start)

        topics = ctm.get_topics(10)
        topics = [topics[x] for x in topics]

        output_tm = {
            "topics": topics,
        }

        return output_tm, computation_time

    def _preprocess_ctm(self):
        """Preprocess data for CTM"""
        # Prepare docs
        data = self.data.get_corpus()
        docs = [" ".join(words) for words in data]

        # Remove stop words
        stop_words = stopwords.words("english")
        preprocessed_documents = [
            " ".join([x for x in doc.split(" ") if x not in stop_words]).strip()
            for doc in docs
        ]

        # Get vocabulary
        vectorizer = CountVectorizer(
            max_features=2000, token_pattern=r"\b[a-zA-Z]{2,}\b"
        )
        vectorizer.fit_transform(preprocessed_documents)
        vocabulary = set(vectorizer.get_feature_names())

        # Preprocess documents further
        preprocessed_documents = [
            " ".join([w for w in doc.split() if w in vocabulary]).strip()
            for doc in preprocessed_documents
        ]

        # Prepare CTM data
        qt = TopicModelDataPreparation("all-mpnet-base-v2")
        training_dataset = qt.fit(
            text_for_contextual=docs, text_for_bow=preprocessed_documents
        )

        self.qt_ctm = qt
        self.training_dataset_ctm = training_dataset

    def _train_octis_model(
        self, params: Mapping[str, any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Train OCTIS model"""

        if self.model_name == "ETM":
            model = ETM(**params)
            model.use_partitions = False
        elif self.model_name == "LDA":
            model = LDA(**params)
            model.use_partitions = False
        elif self.model_name == "CTM":
            model = CTM(**params)
            model.use_partitions = False
        elif self.model_name == "NMF":
            model = NMF(**params)
            model.use_partitions = False

        start = time.time()
        output_tm = model.train_model(self.data)
        end = time.time()
        computation_time = end - start
        return output_tm, computation_time

    def _train_bertopic(
        self, params: Mapping[str, any]
    ) -> Tuple[Mapping[str, Any], float]:
        """Train BERTopic model"""
        data = self.data.get_corpus()
        data = [" ".join(words) for words in data]
        params["calculate_probabilities"] = False

        if self.custom_model is not None:
            model = self.custom_model(**params)
        else:
            model = BERTopic(**params)

        start = time.time()
        topics, _ = model.fit_transform(data, self.embeddings)

        # Dynamic Topic Modeling
        if self.timestamps:
            topics_over_time = model.topics_over_time(
                data,
                topics,
                self.timestamps,
                nr_bins=self.nr_bins,
                evolution_tuning=False,
                global_tuning=False,
            )
            unique_timestamps = topics_over_time.Timestamp.unique()
            dtm_topics = {}
            for unique_timestamp in unique_timestamps:
                dtm_topic = topics_over_time.loc[
                    topics_over_time.Timestamp == unique_timestamp, :
                ].sort_values("Frequency", ascending=True)
                dtm_topic = dtm_topic.loc[dtm_topic.Topic != -1, :]
                dtm_topic = [topic.split(", ") for topic in dtm_topic.Words.values]
                dtm_topics[unique_timestamp] = {"topics": dtm_topic}

                all_words = [word for words in self.data.get_corpus() for word in words]

                updated_topics = []
                for topic in dtm_topic:
                    updated_topic = []
                    for word in topic:
                        if word not in all_words:
                            print(word)
                            updated_topic.append(all_words[0])
                        else:
                            updated_topic.append(word)
                    updated_topics.append(updated_topic)

                dtm_topics[unique_timestamp] = {"topics": updated_topics}

            output_tm = dtm_topics

        end = time.time()
        computation_time = float(end - start)

        if not self.timestamps:
            all_words = [word for words in self.data.get_corpus() for word in words]
            bertopic_topics = [
                [
                    vals[0] if vals[0] in all_words else all_words[0]
                    for vals in model.get_topic(i)[:10]
                ]
                for i in range(len(set(topics)) - 1)
            ]

            output_tm = {"topics": bertopic_topics}

        return output_tm, computation_time

    def evaluate(self, output_tm):
        """Using metrics and output of the topic model, evaluate the topic model"""
        if self.timestamps:
            results = {str(timestamp): {} for timestamp, _ in output_tm.items()}
            for timestamp, topics in output_tm.items():
                self.metrics = self.get_metrics()
                for scorers, _ in self.metrics:
                    for scorer, name in scorers:
                        score = scorer.score(topics)
                        results[str(timestamp)][name] = float(score)

        else:
            # Calculate results
            results = {}
            for scorers, _ in self.metrics:
                for scorer, name in scorers:
                    score = scorer.score(output_tm)
                    results[name] = float(score)

            # Print results
            if self.verbose:
                print("Results")
                print("============")
                for metric, score in results.items():
                    print(f"{metric}: {str(score)}")
                print(" ")

        return results

    def get_dataset(self):
        """Get dataset from OCTIS"""
        data = Dataset()

        if self.custom_dataset:
            data.load_custom_dataset_from_folder(self.dataset)
        else:
            data.fetch_dataset(self.dataset)
        return data

    def get_metrics(self):
        """Prepare evaluation measures using OCTIS"""
        npmi = Coherence(texts=self.data.get_corpus(), topk=self.topk, measure="c_npmi")
        topic_diversity = TopicDiversity(topk=self.topk)

        # Define methods
        coherence = [(npmi, "npmi")]
        diversity = [(topic_diversity, "diversity")]
        metrics = [(coherence, "Coherence"), (diversity, "Diversity")]

        return metrics


def sent_to_words(sentences):
    for sentence in sentences:
        yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))
