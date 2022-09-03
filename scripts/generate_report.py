import os
import sys
import shutil
import pandas as pd
sys.path.append("..")
from args import get_parser
from matplotlib import pyplot as plt
from humanfriendly import format_timespan

parser = get_parser()
opts = parser.parse_args()

def build_line_chart(title, x, label_x, filename):
    plt.title(title)
    plt.plot(x, label=label_x)
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def main():
    report = pd.read_csv(opts.report_filename)

    batch_size_col = report["batch_time"].to_numpy()
    train_loss_col = report["train_loss"].to_numpy()
    medR_col = report["medR"].to_numpy()
    R_at_1_col = report["R@1"].to_numpy()
    R_at_5_col = report["R@5"].to_numpy()
    R_at_10_col = report["R@10"].to_numpy()

    # TODO: Buscar la libreria para imprimir bomnito el delta
    if os.path.isdir('../report'):
        shutil.rmtree('../report')

    with open("../report/info.txt","w") as f:
        total_train = batch_size_col.sum()
        average_epoch_time = total_train / len(batch_size_col)
        f.write(f"Total training took: {format_timespan(total_train)} \n")
        f.write(f"Average epoch took: {format_timespan(average_epoch_time)} \n")
        f.write(f"Highest train loss: {train_loss_col.max()} \n")
        f.write(f"Lowest train loss: {train_loss_col.min()} \n")

    build_line_chart("Loss Curve", train_loss_col.tolist(), "Train Loss", "../report/loss_curve.png")
    build_line_chart("medR over the validation", medR_col.tolist(), "Median Rank", "../report/medR.png")
    build_line_chart("R@1 over the validation", R_at_1_col.tolist(), "Recall at top 1", "../report/r_at_1.png")
    build_line_chart("R@5 over the validation", R_at_5_col.tolist(), "Recall at top 5", "../report/r_at_5.png")
    build_line_chart("R@10 over the validation", R_at_10_col.tolist(), "Recall at top 10", "../report/r_at_10.png")
    

    

    
if __name__ == "__main__":
    main()