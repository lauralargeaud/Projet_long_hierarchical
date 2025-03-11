# README générale du projet long

## Fichiers de timm modifiés:
* [train.py](train.py)
* [inference.py](inference.py)
* [mixup.py](/timm/data/mixup.py) (méthode "one_hot")

## Principaux fichiers créés:
[Utilitaire général](scripts/utils.py)
* LogicSeg: 
    * [LogicSeg loss](timm/loss/logicseg_loss.py)
    * [Asymmetric loss (ASL)](timm/loss/logicseg/asym_loss.py)
    * [Multi-BCE](timm/loss/logicseg/multi_bce_loss.py)
    * [Message passing](scripts/Logicseg_message_passing.py)
    * [Utilitaire LogicSeg](/scripts/logic_seg_utils.py)
* Making Better Mistakes:
    * [Hierarchical Cross Entropy (HCE)](timm/loss/hierarchical_cross_entropy.py)
    * [Utilitaire Soft Labels](scripts/soft_labels_utils.py)
* Métriques:
    * [Classe des métriques](scripts/metrics_hierarchy.py)

## Arguments implémentés

### train.py

*  <span style="color: orange;"><b>General</b></span>
    * --csv-tree
        > <i>Path to csv describing the tree structure of the labels.</i>
    <br />
    
* <span style="color: orange;"><b>LogicSeg</b></span>
    * --logicseg
        > <i>Enable LogicSeg loss.</i>
    * --crule-loss-weight
        > <i>Set the weight of the Closs.</i>
    * --drule-loss-weight
        > <i>Set the weight of the Dloss.</i>
    * --erule-loss-weight
        > <i>Set the weight of the Dloss.</i>
    * --target-loss-weight
        > <i>Set the weight of the loss used to compute the error between output and target.</i>

    * --logicseg-method 
        > <i>Set the loss used to compute the error between ouput and target (ce, bce, asl, multi_bce).</i>
        * --alpha-layer (if using multi bce)
            > <i>Coefficient used to compute the weight of each hierarchical level in the Multi-BCE loss.</i>
        * --asl-gamma-pos (if using asl)
            > <i>Set the gamma_pos coef used for the positive samples in the ASL.</i>
        * --asl-gamma-neg (if using asl)
            > <i>Set the gamma_pos coef used for the negative samples in the ASL.</i>
        * --asl-thresh-shifting (if using asl)
            > <i>Set the threshold coef used for the probability shifting in the ASL.</i>

* <span style="color: orange;"><b>Making Better Mistakes</b></span>
    * --softlabel
        > <i>Convert ground-truth labels to soft labels before calculating the loss.</i>
    * --softlabels-beta
        > <i>Beta parameter for soft_labels transformation.</i>
    <br />
    
    <br />

    * --hce-loss
        > <i>Enable Hierarchical Cross Entropy loss.</i>
    * --hce-alpha
        > <i>Set the alpha of the hce loss.</i>


    
### inference.py

* <span style="color: orange;"><b>General</b></span>
    * --conf-matrix
        > <i>Make confusion matrix.</i>

    * --csv-tree
        > <i>Path to hierarchy csv</i>
    <br />

<br />

* <span style="color: orange;"><b>LogicSeg</b></span>
    * --logicseg
        > <i>Enable logicseg.</i>
    * --message-passing
        > <i>Apply logicseg message passing processing to output.</i>
    * --message-passing-iter-count
        > <i>Number of iteration of the message passing.</i>