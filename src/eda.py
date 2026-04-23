import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os

os.makedirs("outputs/level2/plots", exist_ok=True)

def run_eda(df):

    plt.figure()
    df["resolved_in_72hrs"].value_counts().plot(kind="bar")
    plt.title("Resolved within 72 Hours")
    plt.tight_layout()
    plt.savefig("outputs/level2/plots/target.png")
    plt.close()

    plt.figure()
    df.groupby("borough")["resolved_in_72hrs"].mean().plot(kind="bar")
    plt.title("Resolution Rate by Borough")
    plt.tight_layout()
    plt.savefig("outputs/level2/plots/borough.png")
    plt.close()

    plt.figure()
    df.groupby("hour")["resolved_in_72hrs"].mean().plot()
    plt.title("Resolution Rate by Hour")
    plt.tight_layout()
    plt.savefig("outputs/level2/plots/hour.png")
    plt.close()