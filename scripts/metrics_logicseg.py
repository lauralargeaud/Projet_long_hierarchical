""" Métriques de logicSeg
"""


def accuracy_logicseg(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    print("output shape", output.shape)
    print("target shape", target.shape)
    print("topk", topk)
    return 
