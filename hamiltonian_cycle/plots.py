import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_solution(ds: pd.DataFrame, *, title: str = "") -> None:
    if ds["x"].max() > ds["y"].max():
        width, height = 13.0, 5.0
    else:
        width, height = 5.0, 13.0

    fig, ax = plt.subplots()
    fig.set_size_inches(width, height)
    sns.scatterplot(
        data=ds,
        x="x",
        y="y",
        size="cost",
        legend=False,
        ax=ax,
    ).set_title(title)

    for i in range(-1, len(ds) - 1):
        plt.plot(
            [ds.iloc[i]["x"], ds.iloc[i + 1]["x"]],
            [ds.iloc[i]["y"], ds.iloc[i + 1]["y"]],
            color="k",
            linestyle="-",
        )

    fig.show()
