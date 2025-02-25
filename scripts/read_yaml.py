import yaml
import os

def read_yaml(filepath):
    """
    Read YAML file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print("Fichier non trouvé :", filepath)
    except yaml.YAMLError as e:
        print("Erreur lors de la lecture du YAML :", e)

def compute_model_name(filepath):
    data = read_yaml(filepath)
    modified_logicseg = data["modified_logicseg"] if "modified_logicseg" in data else False
    logicseg = data["logicseg"]
    is_logicseg = modified_logicseg or logicseg
    if is_logicseg:
        method = data["method"]
        if method == "bce":
            return "LogicSeg (BCE)", "logicseg_bce"
        elif method == "multi_bce":
            return "LogicSeg (Multi BCE)", "logicseg_multibce"
        else:
            return "LogicSeg", "logicseg"
    else:
        hce = data["hce_loss"]
        bce = data["bce_loss"]
        hce_alpha = data["hce_alpha"]
        if hce:
            return f"HCE (alpha={hce_alpha})", f'hce_alpha_{str(hce_alpha).replace(".", "_")}'
        elif bce:
            return "BCE", "bce"
        else:
            return "CCE", "cce"

def get_method(data):
    if "method" in data:
        return data["method"]
    else:
        modified_logicseg = data["modified_logicseg"] if "modified_logicseg" in data else False
        logicseg = data["logicseg"]
        if logicseg:
            return "bce"
        elif modified_logicseg:
            return "multi_bce"
        else:
            return "error"

def extract_data_from_yaml(filepath):
    data = read_yaml(filepath)
    modified_logicseg = data["modified_logicseg"] if "modified_logicseg" in data else False
    logicseg = data["logicseg"]
    is_logicseg = modified_logicseg or logicseg
    opt = data["opt"]
    sched = data["sched"]
    num_classes = data["num_classes"]
    print(f'=== File: {filepath} ===')
    print(f'{"Hierarchical Cross Entropy" if not is_logicseg else "LogicSeg"}')
    print(f"opt: {opt}")
    print(f"sched: {sched}")
    print(f"num_classes: {num_classes}")
    if is_logicseg:
        bce_loss_weight = data["bce_loss_weight"]
        crule_loss_weight = data["crule_loss_weight"]
        drule_loss_weight = data["drule_loss_weight"]
        erule_loss_weight = data["erule_loss_weight"]
        method = get_method(data)
        print(f"logicseg method: {method}")
        print(f"bce_loss_weight: {bce_loss_weight}")
        print(f"crule_loss_weight: {crule_loss_weight}")
        print(f"drule_loss_weight: {drule_loss_weight}")
        print(f"erule_loss_weight: {erule_loss_weight}")
    else:
        hce = data["hce_loss"]
        bce = data["bce_loss"]
        cce = not hce and not bce
        hce_alpha = data["hce_alpha"]
        print(f"hce: {hce}")
        print(f"bce: {bce}")
        print(f"cce: {cce}")
        print(f"hce_alpha: {hce_alpha}")
    cutmix = data["cutmix"]
    mixup = data["mixup"]
    hflip = data["hflip"]
    print(f"cutmix: {cutmix}")
    print(f"mixup: {mixup}")
    print(f"hflip: {hflip}")
    print("==================================================")

if __name__ == "__main__":
    root_folder = "output/train"
    filename = "args.yaml"
    for results_folder in os.listdir(root_folder):
        results_folder_path = os.path.join(root_folder, results_folder)
        filepath = os.path.join(results_folder_path, filename)
        data_yaml = extract_data_from_yaml(filepath)
