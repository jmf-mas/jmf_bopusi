import seaborn as sns
from matplotlib import pyplot as plt

palette ={"in": "blue", "out": "red", "synthetic": "green"}
outputs = "outputs/"

def plot_segmented_one_line(identifier, db, threshold, metric):
    sample_df = db.sample(n=1000)
    sns.scatterplot(data=sample_df, x="sample", y=metric, hue='class', palette=palette)
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.savefig(outputs + identifier+".pdf", dpi=300)
    plt.cla()

def plot_segmented(identifier, db, metric):
    sample_df = db.sample(n=1000)
    sns.scatterplot(data=sample_df, x="sample", y=metric, hue='class', palette=palette)
    plt.savefig(outputs + identifier+".pdf", dpi=300)
    plt.cla()

def plot_segmented_two_lines(identifier, db, threshold_min, threshold_max, metric):
    sample_df = db.sample(n=1000)
    sns.scatterplot(data=sample_df, x="sample", y=metric, hue='class', palette=palette)
    plt.axhline(y=threshold_min, color='r', linestyle='-')
    plt.axhline(y=threshold_max, color='r', linestyle='-')
    plt.savefig(outputs + identifier+".pdf", dpi=300)
    plt.cla()