import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from typing import Mapping, List


class Results:
    def __init__(self, main_folder: str, combine_models: bool = False) -> None:
        self.main_folder = main_folder
        self.combine_models = combine_models

        self.basic_results = {}
        self.dtm_results = {}

        self._load_results()

    def get_data(self, name, dtm: bool = False, aggregated: bool = False):
        if dtm:
            if aggregated:
                return self.dtm_results[name].groupby("Model").mean()
            else:
                return self.dtm_results[name]
        else:
            if aggregated:
                return self.basic_results[name].groupby("Model").mean()
            else:
                return self.basic_results[name]

    def get_keys(self):
        return {
            "basic": list(self.basic_results.keys()),
            "dtm": list(self.dtm_results.keys()),
        }

    def visualize_table(self, dtm: bool = False, models: List[str] = None):
        if dtm:
            results = self.dtm_results["all"].copy()
        else:
            results = self.basic_results["all"].copy()
        
        datasets = set([column for column, _ in results.columns if column])

        if models:
            results = results.loc[results[""]["Model"].isin(models), :]

        return (
            results.style.apply(highlight_max)
            .format(
                formatter={
                    (dataset, method): "{:.3f}"
                    for dataset in datasets
                    for method in [
                        "npmi",
                        "diversity",
                    ]
                }
            )
            .set_properties(**{"width": "10em", "text-align": "center"})
            .set_table_styles([dict(selector="th", props=[("text-align", "center")])])
            .set_table_styles([dict(selector="th", props=[("text-align", "center")])])
            .set_table_styles(
                {
                    (dataset, "npmi"): [
                        {"selector": "td", "props": "border-left: 2px solid #000066"}
                    ]
                    for dataset in datasets
                },
                overwrite=False,
                axis=0,
            )
        )

    def visualize_table_tq(self, dtm: bool = False):
        if dtm:
            to_plot = self.dtm_results["all"].copy()
        else:
            to_plot = self.basic_results["all"].copy()

        datasets = list(set([column for column, _ in to_plot.columns if column]))
        models = to_plot[""]["Model"].values
        averaged_results = pd.DataFrame({"Model": models})

        for dataset in datasets:
            averaged_results[dataset] = (
                to_plot[dataset]["npmi"] * to_plot[dataset]["diversity"]
            )
            averaged_results[dataset] = averaged_results[dataset].astype(float).round(3)

        return (
            averaged_results.style.apply(highlight_max)
            .format(
                formatter={
                    (dataset): "{:.3f}"
                    for dataset in datasets
                    for method in ["npmi", "diversity"]
                }
            )
            .set_properties(**{"width": "10em", "text-align": "center"})
            .set_table_styles([dict(selector="th", props=[("text-align", "center")])])
        )

    def _load_results(self):
        folders = os.listdir(self.main_folder)

        if "Basic" in folders:
            for folder in os.listdir(self.main_folder + "/Basic"):
                self._results_per_folder(f"{self.main_folder}Basic/{folder}")
            self._load_all_results()

        if "Dynamic Topic Modeling" in folders:
            for folder in os.listdir(self.main_folder + "Dynamic Topic Modeling"):
                self._load_dtm_results(
                    f"{self.main_folder}Dynamic Topic Modeling/{folder}"
                )
            self._load_all_results(dtm=True)

        if "Computation" in folders:
            self._load_computation_results()

    def _load_all_results(self, dtm=False):
        # Load data
        if dtm:
            data = {
                dataset: (
                    dataframe.groupby("Model")
                    .mean()
                    .loc[
                        :,
                        [
                            "npmi",
                            "diversity",
                        ],
                    ]
                    .reset_index()
                )
                for dataset, dataframe in self.dtm_results.items()
            }
        else:
            data = {
                dataset: (
                    dataframe.groupby("Model")
                    .mean()
                    .loc[
                        :,
                        [
                            "npmi",
                            "diversity",
                        ],
                    ]
                    .reset_index()
                )
                for dataset, dataframe in self.basic_results.items()
            }

        # Sort by model before concatenating
        order = data[list(data.keys())[-2]].sort_values("npmi")["Model"].tolist()
        models = pd.DataFrame({"Model": order})

        for dataset in data.keys():
            data[dataset] = (
                data[dataset]
                .set_index("Model")
                .loc[order]
                .reset_index()
                .drop("Model", axis=1)
            )

        # MultiIndex
        models.columns = pd.MultiIndex.from_product([[""], models.columns])

        for dataset in data.keys():
            data[dataset].columns = pd.MultiIndex.from_product(
                [[dataset], data[dataset].columns]
            )

        results = (
            pd.concat([models] + [data[dataset] for dataset in data.keys()], axis=1)
            .round(3)
            .astype(object)
        )

        if dtm:
            self.dtm_results["all"] = results
        else:
            self.basic_results["all"] = results

    def _results_per_folder(self, folder) -> pd.DataFrame:
        """Load the results for topic model evaluation

        Args:
            main_folder: The main folder from which to extract
                        the results. Make sure that they are
                        saved as .json and follow the evaluation
                        procedure.

        Returns:
            results: The results in a dataframe format where each
                    evaluation point is saved per row
        """

        # Extract all results from the folder
        data_path = [
            data_path
            for data_path in os.listdir(folder)
            if ".json" in data_path and "metadata" not in data_path
        ]

        # Initialize empty df for results
        columns = [
            "Dataset",
            "Model",
            "nr_topics",
            "npmi",
            "diversity",
            "params",
            "Dataset_Size",
            "ComputationTime",
        ]
        results = pd.DataFrame(columns=columns)

        # Extract results from each file
        for index, path in enumerate(data_path):

            # Load raw results
            with open(f"{folder}/{path}", "r") as f:
                data = json.load(f)

            # Write all results to `results`
            for row in data:

                # General info
                dataset = row["Dataset"]
                if self.combine_models:
                    model = row["Model"]
                else:
                    model = row["Model"] + f"_{index}"

                params = row["Params"]
                dataset_size = row["Dataset Size"]
                computation_time = row["Computation Time"]

                # Extract scores
                npmi = row["Scores"]["npmi"]
                diversity = row["Scores"]["diversity"]

                # Get the number of topics depending on how they are
                # defined in the model
                if row["Params"].get("nr_topics"):
                    nr_topics = row["Params"]["nr_topics"]
                elif row["Params"].get("num_topics"):
                    nr_topics = row["Params"]["num_topics"]
                elif row["Params"].get("n_components"):
                    nr_topics = row["Params"]["n_components"]
                else:
                    nr_topics = None

                results.loc[len(results), :] = [
                    dataset,
                    model,
                    nr_topics,
                    npmi,
                    diversity,
                    params,
                    dataset_size,
                    computation_time,
                ]

        # Making sure they have the correct type
        for column in ["npmi", "diversity"]:
            results[column] = results[column].astype(float)

        self.basic_results[dataset] = results

    def _load_dtm_results(self, folder):
        datasets = os.listdir(folder)

        # Initialize empty df for results
        columns = [
            "Dataset",
            "Model",
            "time_slice",
            "nr_topics",
            "npmi",
            "diversity",
            "params",
            "Dataset_Size",
            "ComputationTime",
        ]
        results = pd.DataFrame(columns=columns)

        for data_name in datasets:
            with open(f"{folder}/{data_name}", "r") as f:
                data = json.load(f)

            model = data_name.split(".json")[0][:-1]

            for row in data:
                for time_slice, score in row["Scores"].items():

                    # General info
                    dataset = row["Dataset"]
                    params = row["Params"]
                    dataset_size = row["Dataset Size"]
                    computation_time = row["Computation Time"]

                    # Extract scores
                    npmi = score["npmi"]
                    diversity = score["diversity"]

                    # Get the number of topics depending on how they are
                    # defined in the model
                    if row["Params"].get("nr_topics"):
                        nr_topics = row["Params"]["nr_topics"]
                    elif row["Params"].get("num_topics"):
                        nr_topics = row["Params"]["num_topics"]
                    elif row["Params"].get("n_components"):
                        nr_topics = row["Params"]["n_components"]
                    else:
                        nr_topics = None

                    results.loc[len(results), :] = [
                        dataset,
                        model,
                        time_slice,
                        nr_topics,
                        npmi,
                        diversity,
                        params,
                        dataset_size,
                        computation_time,
                    ]

        for column in ["npmi", "diversity"]:
            results[column] = results[column].astype(float)

        self.dtm_results[dataset] = results

    def _load_computation_results(self):
        path = "../results/Computation/"
        files = os.listdir(path)

        computation = pd.read_csv(path + files[0])
        computation["model"] = files[0]

        for file in files[1:]:
            df_to_add = pd.read_csv(path + file)
            df_to_add["model"] = file
            computation = computation.append(df_to_add)

        self.computation = computation

    def plot_results(
        self,
        dataset: pd.DataFrame = None,
        title: str = None,
        x: str = "nr_topics",
        y: str = "npmi",
        xlabel: str = None,
        ylabel: str = None,
        figsize: tuple = (10, 5),
        confidence_interval: bool = False,
    ):

        results = self.basic_results[dataset].copy()

        fig, ax = plt.subplots(figsize=figsize)

        for model in results.Model.unique():
            selection = results.loc[results.Model == model, :]

            if confidence_interval:
                # Define variables to plot
                y_mean = selection.groupby("nr_topics").mean()[y]
                x_vals = y_mean.index

                # Compute upper and lower bounds using chosen uncertainty measure: here
                # it is a fraction of the standard deviation of measurements at each
                # time point based on the unbiased sample variance
                y_std = selection.groupby("nr_topics").std()[y]
                error = 0.5 * y_std
                lower = y_mean - error
                upper = y_mean + error

                ax.plot(x_vals, y_mean, label=model)
                ax.fill_between(x_vals, lower, upper, alpha=0.2)
            else:
                ax.plot(selection[x], selection[y], label=model)

        if xlabel:
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel(x)

        if ylabel:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel(y)

        plt.grid(axis="x", color=".7", which="major", linestyle="dashed")
        plt.grid(axis="y", color=".7", which="major", linestyle="dashed")

        ax.tick_params(axis="y", direction="in", length=8, which="major")
        ax.tick_params(axis="y", direction="in", length=4, which="minor")
        ax.tick_params(axis="y", direction="in", length=4, which="minor", right=True)

        ax.tick_params(axis="x", direction="in", length=8, which="major")
        ax.tick_params(axis="x", direction="in", length=4, which="minor")
        ax.tick_params(axis="x", direction="in", length=4, which="minor", top=True)

        ax.tick_params(right=True, top=True)

        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
        ax.xaxis.set_minor_locator(tck.AutoMinorLocator())

        plt.xticks(np.arange(min(results[x]), max(results[x]) + 1, 10))

        plt.title(title)
        plt.legend()

        return fig

    def plot_computation(
        self,
        labels: Mapping = None,
        title: str = None,
        xlabel: str = "Vocabulary size",
        ylabel: str = "Wall time (s)",
        figsize: tuple = (10, 5),
        with_ctm: bool = True,
    ):
        fig, ax = plt.subplots(figsize=figsize)
        models = list(self.computation.model.unique())
        linestyles = {"BERTopic": "solid", "Classic": "solid", "Top2Vec": "solid"}

        if not with_ctm:
            models.remove("ctm.csv")

        for model in models:

            selection = self.computation.loc[self.computation.model == model, :]

            if "bertopic" in model:
                linestyle = linestyles["BERTopic"]
            elif "top2vec" in model:
                linestyle = linestyles["Top2Vec"]
            else:
                linestyle = linestyles["Classic"]

            if labels:
                ax.plot(
                    selection.vocab_size,
                    selection.time,
                    label=labels[model],
                    linestyle=linestyle,
                )
            else:
                ax.plot(
                    selection.vocab_size,
                    selection.time,
                    label=model,
                    linestyle=linestyle,
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        plt.grid(axis="x", color=".7", which="major", linestyle="dashed")
        plt.grid(axis="y", color=".7", which="major", linestyle="dashed")

        ax.tick_params(axis="y", direction="in", length=8, which="major")
        ax.tick_params(axis="y", direction="in", length=4, which="minor")
        ax.tick_params(axis="y", direction="in", length=4, which="minor", right=True)

        ax.tick_params(axis="x", direction="in", length=8, which="major")
        ax.tick_params(axis="x", direction="in", length=4, which="minor")
        ax.tick_params(axis="x", direction="in", length=4, which="minor", top=True)

        ax.tick_params(right=True, top=True)

        ax.yaxis.set_minor_locator(tck.AutoMinorLocator())
        ax.xaxis.set_minor_locator(tck.AutoMinorLocator())

        plt.xticks(np.arange(2500, max(self.computation.vocab_size), 2500))

        plt.title(title)
        plt.legend()

        return fig


def highlight_max(data):
    """
    highlight the maximum in a Series or DataFrame
    """
    colors = ["#64db00", "#76FF03", "#e1ffc7"]
    attr = f"background-color: {colors[0]}"
    lighter_attr = f"background-color: {colors[1]}"
    lightest_attr = f"background-color: {colors[2]}"
    if (
        data.ndim == 1 and "BERTopic" not in data
    ):  # Series from .apply(axis=0) or axis=1

        maximum = data.max()
        second_to_max = max([val for val in data if val != maximum])
        third_to_max = max([val for val in data if val not in [maximum, second_to_max]])
        to_return = []
        for value in data:
            if value == maximum and type(value) != str:
                to_return.append(attr)
            elif value == second_to_max and type(value) != str:
                to_return.append(lighter_attr)
            elif value == third_to_max and type(value) != str:
                to_return.append(lightest_attr)
            else:
                to_return.append("")
        return to_return
    else:
        return data
