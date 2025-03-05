import unittest
from metrics_hierarchy import *  # Import de la classe principale
import torch

class TestHierarchicalClassifier(unittest.TestCase):
    def setUp(self):
        """
        Initialisation des données pour les tests.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = 5
        self.hierarchy_matrix = torch.tensor([[0, 1, 1, 0, 0],   
                                            [0, 0, 0, 1, 0],
                                            [0, 0, 0, 0, 1],
                                            [0, 0, 0, 0, 0], 
                                            [0, 0, 0, 0, 0]]).to(self.device)
        self.metrics = MetricsHierarchy(self.hierarchy_matrix, self.device)

    def test_all(self):
        """Exécute le calcul de toutes les métriques via compute_all_metrics."""
        output = torch.tensor([[0.1, 0.5, 0.2, 0.1, 0.1],
                               [0.2, 0.1, 0.6, 0.05, 0.05],
                               [0.3, 0.1, 0.1, 0.4, 0.1]]).to(self.device)
        target = torch.tensor([
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0]
        ]).to(self.device)

        # Calcul de toutes les métriques
        self.metrics.compute_all_metrics(output, target)

        # Affichage des résultats des métriques
        print(self.metrics.get_metrics_string())

    def atest_hierarchical_distance_mistake(self):
        """
        Test du calcul de la distance hiérarchique des erreurs.
        """
        output = torch.tensor([[0.1, 0.5, 0.2, 0.1, 0.1],   # Prédiction : classe 1
                               [0.2, 0.1, 0.6, 0.05, 0.05],  # Prédiction : classe 2
                               [0.3, 0.1, 0.1, 0.4, 0.1]]).to(self.device)  # Prédiction : classe 3
        
        target = torch.tensor([[1, 0, 0, 0, 0],   
                               [0, 0, 1, 0, 0], 
                               [0, 0, 0, 1, 0]]).to(self.device)  # Les vraies classes

        label_matrix = torch.eye(self.num_classes).to(self.device)  # Matrice identité pour simplifier

        self.metrics.hierarchical_distance_mistake(output, target)
        print("Distance hiérarchique moyenne des erreurs : " + str(self.metrics.metrics[MetricsLabels.hierarchical_distance_mistakes]))
    
    def atest_topk_hierarchical_distance_mistake(self):
        """
        Test du calcul du topk distance hiérarchique des erreurs.
        """
        output = torch.tensor([[0.1, 0.5, 0.2, 0.1, 0.1],   # Prédiction : classe 1
                               [0.2, 0.1, 0.6, 0.05, 0.05],  # Prédiction : classe 2
                               [0.3, 0.1, 0.1, 0.4, 0.1]]).to(self.device)  # Prédiction : classe 3
        
        target = torch.tensor([[1, 0, 0, 0, 0],   
                               [0, 0, 1, 0, 0], 
                               [0, 0, 0, 1, 0]]).to(self.device)  # Les vraies classes

        label_matrix = torch.eye(self.num_classes).to(self.device)  # Matrice identité pour simplifier

<<<<<<< HEAD
        self.metrics.topk_hierarchical_distance_mistake(output, target, label_matrix, 3)
        print("Distance top-k hiérarchique moyenne des erreurs : " + str(self.metrics.metrics[MetricsLabels.hierarchical_distance_mistakes]))
    
    def test_c_rule_respect(self):
        """
        Test du calcul du pourcentage de respect de la c rule.
        """
=======
        self.metrics.topk_hierarchical_distance_mistake(output, target, 3)
        print("Distance topk hiérarchique moyenne des erreurs : " + str(self.metrics.metrics[MetricsLabels.hierarchical_distance_mistakes]))
    
    def atest_c_rule_respect(self):
        """Test du calcul du pourcentage de respect de la c rule."""
>>>>>>> f3834073064cea5c996a65c13f1fa5b9f5d3af6d
        output = torch.tensor([[0.6, 0. , 0., 0.6, 0.]   # Prédiction : classe 1
                               ]).to(self.device)  # Prédiction : classe 3
        
        target = torch.tensor([[1, 0, 0, 0, 0],   
                               [0, 0, 1, 0, 0], 
                               [0, 0, 0, 1, 0]]).to(self.device)  # Les vraies classes

        label_matrix = torch.eye(self.num_classes).to(self.device)  # Matrice identité pour simplifier

<<<<<<< HEAD
        self.metrics.c_rule_respect_percentage(output, target, label_matrix)
        print("Pourcentage de respect de la c-rule : " + str(self.metrics.metrics[MetricsLabels.c_rule_respect]))

    def test_d_rule_respect(self):
        """
        Test du calcul du pourcentage de respect de la d rule.
        """
=======
        self.metrics.c_rule_respect_percentage(output, target)
        print("pourcentage de respect de la c-rule : " + str(self.metrics.metrics[MetricsLabels.c_rule_respect]))

    def atest_d_rule_respect(self):
        """Test du calcul du pourcentage de respect de la d rule."""
>>>>>>> f3834073064cea5c996a65c13f1fa5b9f5d3af6d
        output = torch.tensor([[0.6, 0. , 0., 0.6, 0.]   # Prédiction : classe 1
                               ]).to(self.device)  # Prédiction : classe 3
        
        target = torch.tensor([[1, 0, 0, 0, 0],   
                               [0, 0, 1, 0, 0], 
                               [0, 0, 0, 1, 0]]).to(self.device)  # Les vraies classes

        label_matrix = torch.eye(self.num_classes).to(self.device)  # Matrice identité pour simplifier

        self.metrics.d_rule_respect_percentage(output, target, label_matrix)
        print("Pourcentage de respect de la d-rule : " + str(self.metrics.metrics[MetricsLabels.d_rule_respect]))

<<<<<<< HEAD
    def test_e_rule_respect(self):
        """
        Test du calcul du pourcentage de respect de la e rule.
        """
=======
    def atest_e_rule_respect(self):
        """Test du calcul du pourcentage de respect de la e rule."""
>>>>>>> f3834073064cea5c996a65c13f1fa5b9f5d3af6d
        output = torch.tensor([[0.6, 0.6 , 0.6, 0.6, 0.]   # Prédiction : classe 1
                               ]).to(self.device)  # Prédiction : classe 3
        
        target = torch.tensor([[1, 0, 0, 0, 0],   
                               [0, 0, 1, 0, 0], 
                               [0, 0, 0, 1, 0]]).to(self.device)  # Les vraies classes

        label_matrix = torch.eye(self.num_classes).to(self.device)  # Matrice identité pour simplifier

<<<<<<< HEAD
        self.metrics.e_rule_respect_percentage(output, target, label_matrix)
        print("Pourcentage de respect de la e-rule : " + str(self.metrics.metrics[MetricsLabels.e_rule_respect]))
=======
        self.metrics.e_rule_respect_percentage(output, target)
        print("pourcentage de respect de la e-rule : " + str(self.metrics.metrics[MetricsLabels.e_rule_respect]))
>>>>>>> f3834073064cea5c996a65c13f1fa5b9f5d3af6d

if __name__ == "__main__":
    unittest.main()