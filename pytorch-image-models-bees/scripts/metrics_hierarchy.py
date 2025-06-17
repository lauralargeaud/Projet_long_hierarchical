import torch
from scripts.logic_seg_utils import get_logicseg_predictions
import pandas as pd

class MetricsLabels:
    """
    Classe pour stocker les labels des différentes métriques
    """

    accuracy_top1 = "Top 1 accuracy"
    accuracy_top5 = "Top 5 accuracy"
    hierarchical_distance_predictions = "Top 1 hierarchical distance prediction"
    topk_hierarchical_distance_predictions = "Top 5 hierarchical distance prediction"
    hierarchical_distance_mistakes = "hierarchical distance mistakes"

    c_rule_respect_seuil_relatif = "Respect of the c rule with seuil relatif"
    d_rule_respect_seuil_relatif = "Respect of the d rule with seuil relatif"
    e_rule_respect_seuil_relatif = "Respect of the e rule with seuil relatif"

    relative_c_rule_respect_seuil_relatif = "Relative respect of the c rule with seuil relatif"
    relative_d_rule_respect_seuil_relatif = "Relative respect of the d rule with seuil relatif"
    relative_e_rule_respect_seuil_relatif = "Relative respect of the e rule with seuil relatif"

    cd_rule_respect_seuil_max = "Respect of the c & d rule with seuil max"
    relative_cd_rule_respect_seuil_max = "Relative respect of the c & d rule with seuil max"
    



class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricsHierarchy:
    """
    Classe pour calculer et stocker différentes métriques de performance d'une IA.
    """

    def __init__(self, H : torch.Tensor, device):
        """Initialise le dictionnaire pour stocker les métriques."""
        self.metrics = {
            MetricsLabels.accuracy_top1: AverageMeter(),
            MetricsLabels.accuracy_top5: AverageMeter(),
            MetricsLabels.hierarchical_distance_predictions: AverageMeter(),
            MetricsLabels.topk_hierarchical_distance_predictions: AverageMeter(),
            MetricsLabels.hierarchical_distance_mistakes: AverageMeter(),
            MetricsLabels.c_rule_respect_seuil_relatif: AverageMeter(),
            MetricsLabels.d_rule_respect_seuil_relatif: AverageMeter(),
            MetricsLabels.e_rule_respect_seuil_relatif: AverageMeter(),
            MetricsLabels.relative_c_rule_respect_seuil_relatif: AverageMeter(),
            MetricsLabels.relative_d_rule_respect_seuil_relatif: AverageMeter(),
            MetricsLabels.relative_e_rule_respect_seuil_relatif: AverageMeter(),
            MetricsLabels.cd_rule_respect_seuil_max: AverageMeter(),
            MetricsLabels.relative_cd_rule_respect_seuil_max: AverageMeter(),
        }

        self.device = device
        self.H = torch.tensor(H, dtype=torch.float32).to(self.device) #torch tensor    

    def save_metrics_csv(self, filepath):
        df = pd.DataFrame({label: [f"{meter.avg:.4f}"] for label, meter in self.metrics.items()})
        df.to_csv(filepath, index_label="row")
            

    def get_metrics_string(self):
        # Generate a single string with all metrics
        metrics_str = "\n".join([f"{label}: {meter.avg:.4f}" for label, meter in self.metrics.items()])
        return metrics_str
    

    def compute_all_metrics(self, output, target, branches_and_nodes, L, augmented_target = None):
        self.topk_accuracy_logicseg(output, target, 1)
        self.topk_accuracy_logicseg(output, target, 5)

        self.hierarchical_distance_mistake(output, target)
        self.topk_hierarchical_distance_predictions(output, target, 5)
        self.topk_hierarchical_distance_predictions(output, target, 1)

        if augmented_target != None:
            target = augmented_target
        
        self.cd_rule_respect_percentage_seuil_max(branches_and_nodes, L)
        
        tolerance = 0.15 # pour le seuil relatif

        self.c_rule_respect_percentage(branches_and_nodes, L, tolerance)
        self.d_rule_respect_percentage(branches_and_nodes, L, tolerance)
        self.e_rule_respect_percentage(branches_and_nodes, L, tolerance)


    
    def lca_height(self, node1: int, node2: int):
        """Trouve la distance qui sépare node1 de leur Lowest Common Ancestor (LCA).

        :param node1: Premier nœud.
        :param node2: Deuxième nœud.
        :return: distance de node1 au LCA
        """
        distance_node1 = 0
        current_node1 = node1
        current_node2 = node2

        while current_node1 != current_node2:
            parents1 = torch.where(self.H[:, current_node1] == 1)[0].tolist()
            parents2 = torch.where(self.H[:, current_node2] == 1)[0].tolist()

            if len(parents1) == 0:
                parents1 = [current_node1]
                distance_node1 -= 1
            if len(parents2) == 0:
                parents2 = [current_node2]
            if len(parents1) > 1 or len(parents2) > 1:
                raise Exception("2 or more parents for one node")
            
            distance_node1 += 1

            # Mise à jour des parents pour continuer la recherche
            current_node1 = parents1[0]
            current_node2 = parents2[0]

        return distance_node1
    

    def hierarchical_distance_mistake(self, output, target):
        """
        Calcule la distance hierarchique des erreurs

        Args:
            output (torch.Tensor): Prédictions du modèle (logits).
            target (torch.Tensor): Labels réels.
            label_matrix (torch.Tensor): Matrice des labels.
        """

        _, indices_branches_in = output.topk(1, dim=1)
        _, indices_branches_target = target.topk(1, dim=1)

        # Initialiser la distance totale
        total_distance = 0.0
        total_mistakes = 0

        for i in range(target.size(0)):  # Boucle sur tous le batch
            pred_class = indices_branches_in[i].item()
            true_class = indices_branches_target[i].item()

            if pred_class != true_class:
                distance = self.lca_height(pred_class, true_class)
                total_distance += distance
                total_mistakes += 1

        self.metrics[MetricsLabels.hierarchical_distance_mistakes].update(total_distance / total_mistakes)

    def topk_hierarchical_distance_predictions(self, output, target, k=5):
        """
        Calcule la distance hiérarchique moyenne des erreurs pour les k meilleures prédictions.

        Args:
            output (torch.Tensor): Prédictions du modèle (logits), de taille (batch_size, num_classes).
            target (torch.Tensor): Labels réels, de taille (batch_size,).
            label_matrix (torch.Tensor): Matrice des labels.
            k (int): Nombre de classes à considérer dans le top-k.

        Returns:
            float: Distance hiérarchique moyenne des erreurs pour le top-k.
        """

        # Obtenir les indices des k meilleures prédictions et de la cible
        _, indices_branches_in = output.topk(k, dim=1)  # (batch_size, k)
        _, indices_branches_target = target.topk(1, dim=1)  # (batch_size, 1)
        
        # Initialiser la distance totale
        total_distance = 0.0

        for i in range(target.size(0)):  # Boucle sur tous les exemples
            true_class = indices_branches_target[i].item()  # Vraie classe
            distances = []

            for j in range(k):  # Comparer aux k classes les plus probables 
                pred_class = indices_branches_in[i, j].item()
                if pred_class != true_class: 
                    distance = self.lca_height(pred_class, true_class)
                    distances.append(distance)

            # Prendre la moyenne des distances des top-k prédictions
            total_distance += sum(distances) / k

        # Stocker le résultat
        if k == 5:
            self.metrics[MetricsLabels.topk_hierarchical_distance_predictions].update(total_distance / target.size(0))
        elif k == 1:
            self.metrics[MetricsLabels.hierarchical_distance_predictions].update(total_distance / target.size(0))


    def cd_rule_respect_percentage_seuil_max(self, output: torch.Tensor, L):
        """
        Calcule le pourcentage d'échantillons respectant la C-Rule et compte le nombre d'étages où elle est violée.

        Args:
            output (torch.Tensor): Matrice des prédictions du modèle (batch_size, num_classes).
            L (list or torch.Tensor): Seuils pour chaque classe.

        Returns:
            float: Pourcentage des échantillons respectant la C-Rule.
            torch.Tensor: Nombre d'étages où la C-Rule est violée pour chaque échantillon.
        """
        L = torch.tensor(L, dtype=torch.float32).to(self.device)

        batch_size, num_classes, output_pred = self.seuil_max(output, L)
        tree_height,_ = L.shape

        # Calcul des activations des super-classes via la matrice H (Hiérarchie)
        H = self.H.float()  # Matrice hiérarchique (num_classes, num_classes)
        Hs = (torch.repeat_interleave(output_pred.T, repeats=num_classes, dim=1) == 1) & (H.repeat(1, batch_size) == 1)
        Hs = torch.sum(Hs.float(), dim=0)

        enfants = torch.sum(H, dim=0)
        enfants_batch = enfants.repeat(batch_size, 1)  # (batch_size, num_classes)

        # Vérifier que si une classe est activée, sa super-classe l'est aussi
        violation_mask = (output_pred > 0) & (Hs.reshape(batch_size, num_classes) == 0) & (enfants_batch != 0)

        # Compter les échantillons respectant la règle (aucune violation)
        batch_respect = (torch.sum(violation_mask, dim=1) == 0).float()
        total_respect = torch.mean(batch_respect)
        
        # Calcul du nombre d'étages où la règle est violée 
        # (calcule le nombre de noeuds mais avec le max comme seuil = au nombre d'étages)
        levels_violated = torch.sum(violation_mask.float(), dim=1)  # Somme des violations pondérées par H
        total_violation = torch.mean(levels_violated / (tree_height -1))

        # Mise à jour des métriques
        self.metrics[MetricsLabels.cd_rule_respect_seuil_max].update(total_respect)
        self.metrics[MetricsLabels.relative_cd_rule_respect_seuil_max].update(1- total_violation)



    def c_rule_respect_percentage(self, output: torch.Tensor, L, tolerance):
        """
        Calcule le pourcentage d'échantillons respectant la C-Rule et compte le nombre d'étages où elle est violée.

        Args:
            output (torch.Tensor): Matrice des prédictions du modèle (batch_size, num_classes).
            L (list or torch.Tensor): Seuils pour chaque classe.

        Returns:
            float: Pourcentage des échantillons respectant la C-Rule.
            torch.Tensor: Nombre d'étages où la C-Rule est violée pour chaque échantillon.
        """
        L = torch.tensor(L, dtype=torch.float32).to(self.device)
        batch_size, num_classes = output.shape
        tree_height, _ = L.shape

        # Seuil pour binariser les prédictions (0 ou 1)
        output_pred = self.seuil_relatif(output, L, tolerance)
        total_activated_nodes = torch.sum(torch.sum(output_pred, dim = 1), dim = 0)

        # Calcul des activations des super-classes via la matrice H (Hiérarchie)
        H = self.H.float()  # Matrice hiérarchique (num_classes, num_classes)
        Hs = (torch.repeat_interleave(output_pred.T, repeats=num_classes, dim=1) == 1) & (H.repeat(1, batch_size) == 1)
        Hs = torch.sum(Hs.float(), dim=0)

        enfants = torch.sum(H, dim=0)
        enfants_batch = enfants.repeat(batch_size, 1)  # (batch_size, num_classes)

        # Vérifier que si une classe est activée, sa super-classe l'est aussi
        violation_mask = (output_pred > 0) & (Hs.reshape(batch_size, num_classes) == 0) & (enfants_batch != 0)

        # Compter les échantillons respectant la règle (aucune violation)
        batch_respect = (torch.sum(violation_mask, dim=1) == 0).float()
        total_respect = torch.mean(batch_respect)
        
        # Calcul du nombre d'étages où la règle est violée 
        # (calcule le nombre de noeuds mais avec le max comme seuil = au nombre d'étages)
        levels_violated = torch.sum(violation_mask.float(), dim=1)  # Somme des violations pondérées par H
        total_violation = torch.sum(levels_violated, dim = 0) / total_activated_nodes

        # Mise à jour des métriques
        self.metrics[MetricsLabels.c_rule_respect_seuil_relatif].update(total_respect)
        self.metrics[MetricsLabels.relative_c_rule_respect_seuil_relatif].update(1- total_violation)

    def d_rule_respect_percentage(self, output: torch.Tensor, L, tolerance):
        """
        Calcule le pourcentage d'échantillons respectant la D-Rule.

        Args:
            output (torch.Tensor): Matrice des prédictions du modèle (batch_size, num_classes).
            label_matrix (torch.Tensor): Matrice one-hot des labels réels (batch_size, num_classes).

        Returns:
            float: Pourcentage des échantillons respectant la D-Rule.
        """
        batch_size, _ = output.shape
        tree_height, _ = L.shape

        L = torch.tensor(L, dtype = torch.float32).to(self.device)
        # Seuil pour binariser les prédictions (0 ou 1)
        output_pred = self.seuil_relatif(output, L, tolerance)
        total_activated_nodes = torch.sum(torch.sum(output_pred, dim = 1), dim = 0)

        # Calcul des activations des super-classes via la matrice H (Hiérarchie)
        H = self.H.float()  # Matrice hiérarchique (num_classes, num_classes)
        Hs = H @ output_pred.T  # (num_classes, batch_size), prédit les activations correctes des super-classes
        
        parents = torch.sum(H, dim=1)
        parents_batch = parents.repeat(batch_size,1).T  # (batch_size, num_classes)

        violation_mask = (output_pred.T > 0) & (Hs == 0) & (parents_batch != 0)  # (num_classes, batch_size)
        
        # Compter les échantillons respectant la règle (aucune violation)
        batch_respect = (torch.sum(violation_mask, dim=0) == 0).float()  # (batch_size,)
        total_respect = torch.mean(batch_respect)

        # Calcul du nombre d'étages où la règle est violée 
        # (calcule le nombre de noeuds mais avec le max comme seuil = au nombre d'étages)
        levels_violated = torch.sum(violation_mask.float(), dim=1)  # Somme des violations pondérées par H
        total_violation = torch.sum(levels_violated, dim = 0) / total_activated_nodes

        self.metrics[MetricsLabels.relative_d_rule_respect_seuil_relatif].update(1 - total_violation)
        self.metrics[MetricsLabels.d_rule_respect_seuil_relatif].update(total_respect)



    def  e_rule_respect_percentage(self, output: torch.Tensor, L, tolerance):
        """
        Calcule le pourcentage d'échantillons respectant la E-Rule.

        Args:
            output (torch.Tensor): Matrice des prédictions du modèle (batch_size, num_classes).
            label_matrix (torch.Tensor): Matrice one-hot des labels réels (batch_size, num_classes).

        Returns:
            float: Pourcentage des échantillons respectant la E-Rule.
        """
        L = torch.tensor(L, dtype = torch.float32).to(self.device)
        tree_height, _ = L.shape
        output_pred = self.seuil_relatif(output, L, tolerance)  # Matrice binaire (batch_size, num_classes)
        total_activated_nodes = torch.sum(torch.sum(output_pred, dim = 1), dim = 0)


        H = self.H.float()  # Matrice hiérarchique (num_classes, num_classes)
        Hs = H @ output_pred.T

        violation_mask = (Hs > 1)  # (num_classes, batch_size)
        
        # Compter les échantillons respectant la règle (aucune violation)
        batch_respect = (torch.sum(violation_mask, dim=0) == 0).float()  # (batch_size,)
        total_respect = torch.mean(batch_respect)

        # Calcul du nombre d'étages où la règle est violée 
        # (calcule le nombre de noeuds mais avec le max comme seuil = au nombre d'étages)
        levels_violated = torch.sum(violation_mask.float(), dim=1)  # Somme des violations pondérées par H
        total_violation = torch.sum(levels_violated, dim = 0) / total_activated_nodes

        self.metrics[MetricsLabels.relative_e_rule_respect_seuil_relatif].update(1 - total_violation)
        self.metrics[MetricsLabels.e_rule_respect_seuil_relatif].update(total_respect)



    def seuil_max(self, output, L):
        '''Repere les noeuds activés par le modele en considerant le maximum par hauteur de l'arbre'''

        batch_size, num_classes = output.shape
        tree_height, _ = L.shape

        output_pred = torch.repeat_interleave(output.T, repeats=tree_height, dim=1)
        augmented_L = L.repeat(batch_size,1).T

        output_pred = output_pred*augmented_L

        _, indices = torch.max(output_pred, dim = 0)

        out = torch.zeros(batch_size, num_classes).to(self.device)

        compteur = 0
        num_ligne = 0
        for i in indices:
            out[num_ligne, i] = 1
            compteur += 1
            if compteur == tree_height:
                compteur = 0
                num_ligne += 1
        
        return batch_size,num_classes,out



    def seuil_relatif(self, output, L, tolerance):
        ''' Regarde le maximum par etage et considere comme allumé tout les noeud à tolerance % de ce maximum'''

        batch_size, num_classes = output.shape
        tree_height, _ = L.shape

        output_pred = torch.repeat_interleave(output.T, repeats=tree_height, dim=1)
        augmented_L = L.repeat(batch_size,1).T

        output_pred = output_pred*augmented_L

        valeurs_max, _ = torch.max(output_pred, dim = 0)
        valeurs_seuil = valeurs_max * (1 - tolerance)

        output_seuil = (output_pred > valeurs_seuil).float()
        output_seuil = torch.reshape(output_seuil, (num_classes, tree_height, batch_size))
        out = torch.sum(output_seuil, dim=1)  

        return out.T


    def topk_accuracy_logicseg(self, probas_branches_input, onehot_targets, topk=1):
        """
        Generic function that computes the topk accuracy (= the accuracy over the topk top predictions) 
        for the specified values of topk.
        """
        topk = min(topk,probas_branches_input.shape[1])
        _, indices_branches_target = onehot_targets.topk(1, dim=1) # (nb_pred, 1), (nb_pred, 1)
        indices_branches_target = indices_branches_target.repeat(1, topk) # (nb_pred, top_k)
        _, indices_branches_in = probas_branches_input.topk(topk , dim=1) # (nb_pred, top_k), (nb_pred, top_k)
        acc = torch.sum(torch.any(indices_branches_in == indices_branches_target, dim=1), dim=0) / indices_branches_in.shape[0]

        if (topk == 1):
            self.metrics[MetricsLabels.accuracy_top1].update(acc)
        
        if (topk == 5):
            self.metrics[MetricsLabels.accuracy_top5].update(acc)


    def reset_metrics(self):
        """
        Réinitialise les métriques stockées.
        """
        self.metrics = {key: AverageMeter() for key in self.metrics}


    def compute_tree_matrix(self):
        """
        Retourne une matrice contenant les noeuds triés par hauteur dans l'arbre
        """
        pass

    """Fonction non mise a jour"""
    def compute_metrics(self, output, target, label_matrix, device):
        """
        Compute all the define metrics using the given data.
        """
        label_matrix = torch.tensor(label_matrix, dtype=torch.float32).to(self.device)
        self.hierarchical_distance_mistake(output, target)
        self.topk_hierarchical_distance_mistake(output, target, k=1)
        self.topk_hierarchical_distance_mistake(output, target)
        self.c_rule_respect_percentage(output, label_matrix)
        self.d_rule_respect_percentage(output, label_matrix)
        self.e_rule_respect_percentage(output)
        # self.accuracy_topk_1_5(output, target, label_matrix, device)
        self.topk_accuracy_logicseg(output, target, topk=1)
        self.topk_accuracy_logicseg(output, target, topk=5)
