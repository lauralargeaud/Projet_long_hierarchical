import torch
from Logicseg_message_passing import *

def test_messsage_passing():

    # Définition des matrices P et M factices
    P_raw = [[0, 0, 0, 0, 0, 0, 0],
             [0, 0, 1.0, 0, 0, 0, 0],
             [0, 1.0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 1.0, 0, 0],
             [0, 0, 0, 1.0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1.0],
             [0, 0, 0, 0, 0, 1.0, 0]]
    M_raw = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    H_raw = [[0, 1.0, 1.0, 0, 0, 0, 0],
             [0, 0, 0, 1.0, 1.0, 0, 0],
             [0, 0, 0, 0, 0, 1.0, 1.0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0]]
    La_raw = [[1.0,0,0,0,0,0,0], 
              [0,1.0,1.0,0,0,0,0],
              [0,0,0,1.0,1.0,1.0,1.0]]
    y_pred = torch.tensor([[1, 1, 0, 1, 0, 0, 0],
                           [1, 1, 0.1, 0.1, 0.1, 0.1, 0.1],
                           [0.9, 0.1, 0.5, 0.99, 0.01, 0.5, 0.01], 
                           [0.9, 0.9, 0.05, 0.99, 0.01, 0.01, 0.01]], dtype=torch.float32)

    mess_pass = MessagePassing(H_raw, P_raw, M_raw, La_raw, 2, "cpu")

    y_pred = mess_pass.process(y_pred)

    print(y_pred)

test_messsage_passing()


    
