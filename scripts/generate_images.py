import pandas as pd
from matplotlib import pyplot as plt

COLORS = ["dodgerblue", "green", "red", "orange"]
NAMES = ["ResNet50-old", "ResNet50-new", "ResNeXt50-old","ResNeXt50-new" ]
REPORTS = ["report_e200_v-19.800.csv", "report_e200_v-15.850.csv", "report_e200_v-18.800.csv","report_e200_v-39.550.csv" ]


# COLORS = ["dodgerblue", "orange"]
# ResNeXt
# NAMES = ["ResNeXt50-old","ResNeXt50-new" ]
# REPORTS = ["report_e200_v-18.800.csv","report_e200_v-39.550.csv" ]

# ResNet
# NAMES = ["ResNet50-old", "ResNet50-new"]
# REPORTS = ["report_e200_v-19.800.csv", "report_e200_v-15.850.csv"]


COLORS = ["dodgerblue", "orange"]
NAMES = ["ResNet50-old", "ResNeXt50-old" ]
REPORTS = ["report_e200_v-19.800.csv", "report_e200_v-18.800.csv" ]

def build_chart(dataframes, column, title, filename):
    global COLORS
    global NAMES
    global REPORTS
    for i, df in enumerate(dataframes):
        plt.plot(df[column],color=COLORS[i], label=NAMES[i].replace("-old"," Original DA").replace("-new", " Proposed DA"))
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def main():
    global NAMES
    global REPORTS
    dataframes = []
    for i, name in enumerate(NAMES):
        try:
            dataframes.append(pd.read_csv(f"../data/trained/{name}/{REPORTS[i]}"))
        except FileNotFoundError:
            print("file not found")
    
    build_chart(dataframes, "train_loss", "Loss curve for all architectures with the original data augmentation", "loss_curve-archs.png")
    

if __name__ == "__main__":
    main()