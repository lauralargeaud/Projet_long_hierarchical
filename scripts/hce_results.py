import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

def show_results_from_csv_summary(filename):
    matplotlib.use('TkAgg')

    data = pd.read_csv(filename)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(data['epoch'], data['train_loss'], label='Train Loss', color='blue', linestyle='--')
    ax1.plot(data['epoch'], data['eval_loss'], label='Eval Loss', color='red')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_title("Training and Evaluation Loss")
    ax1.grid()

    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data['epoch'], data['eval_top1'], label='Eval Top-1 Accuracy', color='green')
    ax2.plot(data['epoch'], data['eval_top5'], label='Eval Top-5 Accuracy', color='purple')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="lower right")
    ax2.set_title("Evaluation Accuracy")
    ax2.grid()

    plt.show()
    plt.imsave()

def show_results_from_csv_summary_cce_hce(filename_cce, filename_hce, folder="output/img"):
    matplotlib.use('TkAgg')

    data_cce = pd.read_csv(filename_cce)
    data_hce = pd.read_csv(filename_hce)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(data_cce['epoch'], data_cce['train_loss'], label='CCE Train Loss', color='red', linestyle='--')
    ax1.plot(data_cce['epoch'], data_cce['eval_loss'], label='CCE Eval Loss', color='orange')
    ax1.plot(data_hce['epoch'], data_hce['train_loss'], label='HCE Train Loss', color='green', linestyle='--')
    ax1.plot(data_hce['epoch'], data_hce['eval_loss'], label='HCE Eval Loss', color='lime')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_title("Training and Evaluation Loss")
    ax1.grid()
    fig.savefig(os.path.join(folder, f"loss_summary_cce_hce"))

    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data_cce['epoch'], data_cce['eval_top1'], label='CCE Eval Top-1 Accuracy', color='red')
    ax2.plot(data_cce['epoch'], data_cce['eval_top5'], label='CCE Eval Top-5 Accuracy', color='orange')
    ax2.plot(data_hce['epoch'], data_hce['eval_top1'], label='HCE Eval Top-1 Accuracy', color='green')
    ax2.plot(data_hce['epoch'], data_hce['eval_top5'], label='HCE Eval Top-5 Accuracy', color='lime')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="lower right")
    ax2.set_title("Evaluation Accuracy")
    ax2.grid()
    fig.savefig(os.path.join(folder, f"acc_summary_cce_hce"))

    plt.show()

def show_results_from_csv_summary_cce_hce_alpha(filename_cce, filename_hce_0_1, filename_hce_0_5, folder="output/img"):
    matplotlib.use('TkAgg')

    data_cce = pd.read_csv(filename_cce)
    data_hce_0_1 = pd.read_csv(filename_hce_0_1)
    data_hce_0_5 = pd.read_csv(filename_hce_0_5)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(data_cce['epoch'], data_cce['train_loss'], label='CCE Train Loss', color='red', linestyle='--')
    ax1.plot(data_cce['epoch'], data_cce['eval_loss'], label='CCE Eval Loss', color='orange')
    ax1.plot(data_hce_0_1['epoch'], data_hce_0_1['train_loss'], label='HCE (alpha=0.1) Train Loss', color='green', linestyle='--')
    ax1.plot(data_hce_0_1['epoch'], data_hce_0_1['eval_loss'], label='HCE (alpha=0.1) Eval Loss', color='lime')
    ax1.plot(data_hce_0_5['epoch'], data_hce_0_5['train_loss'], label='HCE (alpha=0.5) Train Loss', color='blue', linestyle='--')
    ax1.plot(data_hce_0_5['epoch'], data_hce_0_5['eval_loss'], label='HCE (alpha=0.5) Eval Loss', color='skyblue')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_title("Training and Evaluation Loss")
    ax1.grid()
    fig.savefig(os.path.join(folder, f"loss_summary_cce_hce"))

    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data_cce['epoch'], data_cce['eval_top1'], label='CCE Eval Top-1 Accuracy', color='red')
    ax2.plot(data_cce['epoch'], data_cce['eval_top5'], label='CCE Eval Top-5 Accuracy', color='orange')
    ax2.plot(data_hce_0_1['epoch'], data_hce_0_1['eval_top1'], label='HCE (alpha=0.1) Eval Top-1 Accuracy', color='green')
    ax2.plot(data_hce_0_1['epoch'], data_hce_0_1['eval_top5'], label='HCE (alpha=0.1) Eval Top-5 Accuracy', color='lime')
    ax2.plot(data_hce_0_5['epoch'], data_hce_0_5['eval_top1'], label='HCE (alpha=0.5) Eval Top-1 Accuracy', color='blue')
    ax2.plot(data_hce_0_5['epoch'], data_hce_0_5['eval_top5'], label='HCE (alpha=0.5) Eval Top-5 Accuracy', color='skyblue')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="lower right")
    ax2.set_title("Evaluation Accuracy")
    ax2.grid()
    fig.savefig(os.path.join(folder, f"acc_summary_cce_hce"))

    plt.show()