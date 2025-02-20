import os
import glob
from PIL import Image
import shutil

def keep_only_n_images(root_folder, n=25):
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        
        if os.path.isdir(subdir_path):
            image_files = sorted(glob.glob(os.path.join(subdir_path, "*.*")))
            image_files = [f for f in image_files if f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp"))]
            
            if len(image_files) > n:
                for img in image_files[n:]:
                    os.remove(img)
            else:
                pass
                print(f"No images found in {subdir_path}")

def remove_classes(root_folder, classes_to_remove):
    for class_ in classes_to_remove:
        class_folder = os.path.join(root_folder, class_)
        shutil.rmtree(class_folder)

def resize_images(input_folder, output_folder, size=(224, 224)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        try:
            with Image.open(input_path) as img:
                new_img = img.resize(size)  # Conserve le ratio tout en s'adaptant à la taille max
                new_img.save(output_path)
                print(f"Image redimensionnée et enregistrée : {output_path}")
        except Exception as e:
            print(f"Erreur lors du traitement de {filename}: {e}")

def copy_folder(src, dst):
    try:
        if not os.path.exists(src):
            print(f"Le dossier src '{src}' n'existe pas.")
            return
        
        if os.path.exists(dst):
            print(f"Le dossier dst '{dst}' existe déjà.")
            shutil.rmtree(dst)
            print(f"Suppresion de {dst}.")
            shutil.copytree(src, dst)
            print(f"Le dossier '{src}' a été copié vers '{dst}'.")
        else:
            shutil.copytree(src, dst)
            print(f"Le dossier '{src}' a été copié vers '{dst}'.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

def create_dataset_test(classes_to_remove):
    dataset_dirname = "data/small-collomboles-n-classes/dataset"
    dataset_test_dirname = "data/small-collomboles-n-classes/dataset_test"

    copy_folder(dataset_dirname, dataset_test_dirname)

    min_class = 150
    dataset_test_dirname_train = os.path.join(dataset_test_dirname, "train")
    dataset_test_dirname_val = os.path.join(dataset_test_dirname, "val")
    classes_folder = os.listdir(dataset_test_dirname_train)
    classes_folder.sort()
    for class_folder in classes_folder:
        class_path = os.path.join(dataset_test_dirname_train, class_folder)
        images_path = os.listdir(class_path)
        n_image = len(images_path)
        print(f"{class_folder}: {n_image}")
        if n_image < min_class:
            min_class = n_image

    remove_classes(dataset_test_dirname_train, classes_to_remove)
    remove_classes(dataset_test_dirname_val, classes_to_remove)

    keep_only_n_images(dataset_test_dirname_train, n=10)
    keep_only_n_images(dataset_test_dirname_val, n=2)

if __name__ == "__main__":
    # resize_images("data/simple/logicseg", "data/simple/logicseg/resize", size=(224, 224))
    # dirname = "data/small-collomboles-2-class/dataset/val"
    # keep_only_first_image(dirname)
    classes_to_remove = [
        "Allacma fusca",
        "Anurida maritima",
        "Bilobella aurantiaca",
        # "Bilobella braunerae",
        "Bourletiella arvalis",
        # "Bourletiella hortensis",
        "Brachystomella parvula",
        "Caprainea marginata",
        # "Ceratophysella denticulata",
        "Ceratophysella longispina",
        "Cyphoderus albinus",
        "Deuterosminthurus bicinctus",
        # "Deuterosminthurus pallipes",
        "Dicyrtoma fusca",
        "Dicyrtomina flavosignata",
        "Dicyrtomina minuta",
        # "Dicyrtomina ornata",
        "Dicyrtomina saundersi",
        "Dicyrtomina signata",
        "Entomobrya albocincta",
        "Entomobrya atrocincta",
        "Entomobrya corticalis",
        "Entomobrya marginata",
        "Entomobrya multifasciata",
        "Entomobrya muscorum",
        "Entomobrya nicoleti",
        "Entomobrya nigrocincta",
        "Entomobrya nivalis",
        # "Entomobrya superba",
        "Fasciosminthurus quinquefasciatus",
        "Folsomia candida",
        # "Folsomia quadrioculata",
        "Heteromurus major",
        # "Heteromurus nitidus",
        "Hypogastrura viatica",
        "Isotoma riparia",
        # "Isotoma viridis",
        "Isotomiella minor",
        "Isotomurus gallicus",
        "Isotomurus maculatus",
        "Isotomurus palustris",
        # "Isotomurus unifasciatus",
        "Kalaphorura burmeisteri",
        "Lepidocyrtus curvicollis",
        "Lepidocyrtus cyaneus",
        "Lepidocyrtus fimetarius",
        "Lepidocyrtus lignorum",
        # "Lepidocyrtus paradoxus",
        "Megalothorax minimus",
        "Monobella grassei",
        "Neanura muscorum",
        "Neelus murinus",
        "Orchesella bifasciata",
        "Orchesella cincta",
        "Orchesella flavescens",
        "Orchesella quinquefasciata",
        # "Orchesella villosa",
        "Parisotoma notabilis",
        "Podura aquatica",
        # "Pogonognathellus flavescens",
        "Pogonognathellus longicornis",
        # "Seira ferrarii",
        "Seira musarum",
        # "Sminthurides aquaticus",
        "Sminthurides malmgreni",
        "Sminthurides penicillifer",
        "Sminthurinus aureus",
        "Sminthurinus elegans",
        "Sminthurinus lawrencei",
        "Sminthurinus niger",
        "Sminthurinus trinotatus",
        "Sminthurus viridis",
        # "Tomocerus minor",
        "Tomocerus vulgaris",
        "Vertagopus asiaticus",
        "Vitronura giselae",
        # "Willowsia buski",
        "Willowsia nigromaculata",
        "Willowsia platani",
    ]
    create_dataset_test(classes_to_remove)