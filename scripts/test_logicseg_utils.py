from logic_seg_utils import *

def test_get_layer_matrix():
    path = "/mnt/c/Users/rubcr/OneDrive/Bureau/projet_long/pytorch-image-models-bees/scripts/data_test/hierarchy.csv"
    La = get_layer_matrix(path,True)
    
def main():
    test_get_layer_matrix()

main()