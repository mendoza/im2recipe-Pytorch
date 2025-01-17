import os
import sys
import shutil
import numpy as np
import pandas as pd
sys.path.append("..")
from args import get_parser
from matplotlib import pyplot as plt
from humanfriendly import format_timespan

parser = get_parser()
opts = parser.parse_args()

def generate_line_chart_image(title, x, label_x, filename):
    plt.title(title)
    plt.plot(x, label=label_x)
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def remove_minus_ones(numpy_list):
    return numpy_list[~np.isin(numpy_list, [-1])]

def main():
    report = pd.read_csv(opts.report_path)

    batch_size_col = report["epoch_time"].to_numpy()
    train_loss_col = report["train_loss"].to_numpy()
    medR_col = remove_minus_ones(report["medR"].to_numpy())
    R_at_1_col = remove_minus_ones(report["R@1"].to_numpy())
    R_at_5_col = remove_minus_ones(report["R@5"].to_numpy())
    R_at_10_col = remove_minus_ones(report["R@10"].to_numpy())

    if os.path.isdir('../report'):
        shutil.rmtree("../report")
        
    os.mkdir("../report")

    with open("../report/info.txt","w+") as f:
        total_train = batch_size_col.sum()
        average_epoch_time = total_train / len(batch_size_col)
        f.write(f"Total training took: {format_timespan(total_train)} \n")
        f.write(f"Average epoch took: {format_timespan(average_epoch_time)} \n")
        f.write(f"Highest train loss: {train_loss_col.max()} \n")
        f.write(f"Lowest train loss: {train_loss_col.min()} \n")
        f.write(f"medR: {medR_col[-1]} \n")
        f.write(f"R@1: {R_at_1_col[-1]} \n")
        f.write(f"R@5: {R_at_5_col[-1]} \n")
        f.write(f"R@10: {R_at_10_col[-1]} \n")
        f.write(f"Average Loss: {train_loss_col[-1]} \n")
        

    generate_line_chart_image("Loss Curve", train_loss_col.tolist(), "Train Loss", "../report/loss_curve.png")
    generate_line_chart_image("medR over the validation", medR_col.tolist(), "Median Rank", "../report/medR.png")
    generate_line_chart_image("R@1 over the validation", R_at_1_col.tolist(), "Recall at top 1", "../report/r_at_1.png")
    generate_line_chart_image("R@5 over the validation", R_at_5_col.tolist(), "Recall at top 5", "../report/r_at_5.png")
    generate_line_chart_image("R@10 over the validation", R_at_10_col.tolist(), "Recall at top 10", "../report/r_at_10.png")
    

if __name__ == "__main__":
    main()