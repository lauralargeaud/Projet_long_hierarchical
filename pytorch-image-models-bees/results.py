from scripts.results import *

if __name__ == "__main__":
    display_models_summary("output/train_bis", output_folder="output/img_bis/summary")
    display_models_barplots_multiple("output/test_bis", output_folder="output/img_bis")