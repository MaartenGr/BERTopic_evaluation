import re
import nltk
import string
import pandas as pd

from typing import List, Tuple, Union
from octis.dataset.dataset import Dataset
from octis.preprocessing.preprocessing import Preprocessing

nltk.download("punkt")


class DataLoader:
    """Prepare and load custom data using OCTIS

    Arguments:
        dataset: The name of the dataset, default options:
                    * trump
                    * 20news

    Usage:

    **Trump** - Unprocessed

    ```python
    from evaluation import DataLoader
    dataloader = DataLoader(dataset="trump").prepare_docs(save="trump.txt").preprocess_octis(output_folder="trump")
    ```

    **20 Newsgroups** - Unprocessed

    ```python
    from evaluation import DataLoader
    dataloader = DataLoader(dataset="20news").prepare_docs(save="20news.txt").preprocess_octis(output_folder="20news")
    ```

    **Custom Data**

    Whenever you want to use a custom dataset (list of strings), make sure to use the loader like this:

    ```python
    from evaluation import DataLoader
    dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs).preprocess_octis(output_folder="my_docs")
    ```
    """

    def __init__(self, dataset: str):
        self.dataset = dataset
        self.docs = None
        self.timestamps = None
        self.octis_docs = None
        self.doc_path = None

    def load_docs(
        self, save: bool = False, docs: List[str] = None
    ) -> Tuple[List[str], Union[List[str], None]]:
        """Load in the documents

        ```python
        dataloader = DataLoader(dataset="trump")
        docs, timestamps = dataloader.load_docs()
        ```
        """
        if docs is not None:
            return self.docs, None

        if self.dataset == "trump":
            self.docs, self.timestamps = self._trump()
        elif self.dataset == "trump_dtm":
            self.docs, self.timestamps = self._trump_dtm()
        elif self.dataset == "un_dtm":
            self.docs, self.timestamps = self._un_dtm()
        elif self.dataset == "20news":
            self.docs, self.timestamps = self._20news()

        if save:
            self._save(self.docs, save)

        return self.docs, self.timestamps

    def load_octis(self, custom: bool = False) -> Dataset:
        """Get dataset from OCTIS

        Arguments:
            custom: Whether a custom dataset is used or one retrieved from
                    https://github.com/MIND-Lab/OCTIS#available-datasets

        Usage:

        ```python
        from evaluation import DataLoader
        dataloader = DataLoader(dataset="20news")
        data = dataloader.load_octis(custom=True)
        ```
        """
        data = Dataset()

        if custom:
            data.load_custom_dataset_from_folder(self.dataset)
        else:
            data.fetch_dataset(self.dataset)

        self.octis_docs = data
        return self.octis_docs

    def prepare_docs(self, save: bool = False, docs: List[str] = None):
        """Prepare documents

        Arguments:
            save: The path to save the model to, make sure it ends in .json
            docs: The documents you want to preprocess in OCTIS

        Usage:

        ```python
        from evaluation import DataLoader
        dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs)
        ```
        """
        self.load_docs(save, docs)
        return self

    def preprocess_octis(
        self,
        preprocessor: Preprocessing = None,
        documents_path: str = None,
        output_folder: str = "docs",
    ):
        """Preprocess the data using OCTIS

        Arguments:
            preprocessor: Custom OCTIS preprocessor
            documents_path: Path to the .txt file
            output_folder: Path to where you want to save the preprocessed data

        Usage:

        ```python
        from evaluation import DataLoader
        dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs)
        dataloader.preprocess_octis(output_folder="my_docs")
        ```

        If you want to use your custom preprocessor:

        ```python
        from evaluation import DataLoader
        from octis.preprocessing.preprocessing import Preprocessing

        preprocessor = Preprocessing(lowercase=False,
                                remove_punctuation=False,
                                punctuation=string.punctuation,
                                remove_numbers=False,
                                lemmatize=False,
                                language='english',
                                split=False,
                                verbose=True,
                                save_original_indexes=True,
                                remove_stopwords_spacy=False)

        dataloader = DataLoader(dataset="my_docs").prepare_docs(save="my_docs.txt", docs=my_docs)
        dataloader.preprocess_octis(preprocessor=preprocessor, output_folder="my_docs")
        ```
        """
        if preprocessor is None:
            preprocessor = Preprocessing(
                lowercase=False,
                remove_punctuation=False,
                punctuation=string.punctuation,
                remove_numbers=False,
                lemmatize=False,
                language="english",
                split=False,
                verbose=True,
                save_original_indexes=True,
                remove_stopwords_spacy=False,
            )
        if not documents_path:
            documents_path = self.doc_path
        dataset = preprocessor.preprocess_dataset(documents_path=documents_path)
        dataset.save(output_folder)

    def _trump(self) -> Tuple[List[str], List[str]]:
        """Prepare the trump dataset"""
        trump = pd.read_csv(
            "https://drive.google.com/uc?export=download&id=1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6"
        )
        trump = trump.loc[(trump.isRetweet == "f") & (trump.text != ""), :]
        timestamps = trump.date.to_list()
        docs = trump.text.to_list()
        docs = [doc.lower().replace("\n", " ") for doc in docs if len(doc) > 2]
        timestamps = [
            timestamp for timestamp, doc in zip(timestamps, docs) if len(doc) > 2
        ]
        return docs, timestamps

    def _trump_dtm(self) -> Tuple[List[str], List[str]]:
        """Prepare the trump dataset including timestamps"""
        trump = pd.read_csv(
            "https://drive.google.com/uc?export=download&id=1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6"
        )
        trump = trump.loc[(trump.isRetweet == "f") & (trump.text != ""), :]
        timestamps = trump.date.to_list()
        documents = trump.text.to_list()

        docs = []
        time = []
        for doc, timestamp in zip(documents, timestamps):
            if len(doc) > 2:
                docs.append(doc.lower().replace("\n", " "))
                time.append(timestamp)

        # Create bins
        nr_bins = 10
        df = pd.DataFrame({"Doc": docs, "Timestamp": time}).sort_values("Timestamp")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], infer_datetime_format=True)
        df["Bins"] = pd.cut(df.Timestamp, bins=nr_bins)
        df["Timestamp"] = df.apply(lambda row: row.Bins.left, 1)
        timestamps = df.Timestamp.tolist()
        documents = df.Doc.tolist()

        return docs, timestamps

    def _un_dtm(self) -> Tuple[List[str], List[str]]:
        """Prepare the UN dataset"""

        def create_paragraphs(text):
            text = text.replace("Mr.\n", "Mr. ")
            text = text.replace(".\n", " \p ")
            text = text.replace(". \n ", " \p ")
            text = text.replace(". \n", " \p ")
            text = text.replace("\n", " ")
            text = [x.strip().lower() for x in text.split("\p")]
            return text

        dataset = pd.read_csv(
            "https://runestone.academy/runestone/books/published/httlads/_static/un-general-debates.csv"
        )
        dataset["text"] = dataset.apply(lambda row: create_paragraphs(row.text), 1)
        dataset = dataset.explode("text").sort_values("year")
        dataset = dataset.loc[dataset.year > 2005, :]
        docs = dataset.text.tolist()
        timestamps = dataset.year.tolist()
        return docs, timestamps

    def _save(self, docs: List[str], save: str):
        """Save the documents"""
        with open(save, mode="wt", encoding="utf-8") as myfile:
            myfile.write("\n".join(docs))

        self.doc_path = save
