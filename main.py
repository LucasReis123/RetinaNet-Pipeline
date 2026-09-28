import os
import yaml
from detectron2.data import DatasetCatalog, MetadataCatalog
from dataset_utils import DatasetUtils
from dataset_mapper import get_yolo_dicts
from trainer import DetectronTrainer
from evaluator import YoloEvaluator
from analyzer import DetectronAnalyzer


def env_str(name, default):
    """
        Reads a string from the environment, falling back to default when the variable is unset
        or empty. Lets Docker (or the shell) override any setting without editing this file.

    Args:
        name (str): Environment variable name.
        default (str): Value used when the variable isn't set.

    Returns:
        str: The variable's value or the default.
    """
    value = os.getenv(name)
    return value if value not in (None, "") else default


def env_int(name, default):
    """
        Same as env_str, converting the value to int.

    Args:
        name (str): Environment variable name.
        default (int): Value used when the variable isn't set.

    Returns:
        int: The variable's value or the default.
    """
    return int(env_str(name, default))


def env_float(name, default):
    """
        Same as env_str, converting the value to float.

    Args:
        name (str): Environment variable name.
        default (float): Value used when the variable isn't set.

    Returns:
        float: The variable's value or the default.
    """
    return float(env_str(name, default))


def env_bool(name, default):
    """
        Same as env_str, reading "1", "true", "yes" or "on" (any case) as True and anything else
        as False.

    Args:
        name (str): Environment variable name.
        default (bool): Value used when the variable isn't set.

    Returns:
        bool: The variable's value or the default.
    """
    return str(env_str(name, default)).strip().lower() in ("1", "true", "yes", "on")


# HIPERPARÂMETROS E CONFIGURAÇÕES

# Dataset Control Panel
DATASET_NAME = env_str("DATASET_NAME", "Torso21")      # "heart" ou "Torso21"
DATASET_ROOT = env_str("DATASET_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Dateset"))
DATASET_PATH = os.path.join(DATASET_ROOT, DATASET_NAME)
DATASET_YAML = os.path.join(DATASET_PATH, "data.yaml")

TRAIN_IMG_DIR = os.path.join(DATASET_PATH, "train", "images")
TRAIN_LBL_DIR = os.path.join(DATASET_PATH, "train", "labels")
VAL_IMG_DIR = os.path.join(DATASET_PATH, "valid", "images")
VAL_LBL_DIR = os.path.join(DATASET_PATH, "valid", "labels")

# Dataset Splitting
VAL_RATIO = env_float("VAL_RATIO", 0.20)  # Porcentagem para validação (20%)
TEST_RATIO = env_float("TEST_RATIO", 0.10) # Porcentagem para testes (10%)

# Force Generic Class
FORCE_SINGLE_CLASS = env_bool("FORCE_SINGLE_CLASS", False)
SINGLE_CLASS_NAME = env_str("SINGLE_CLASS_NAME", "object")
SINGLE_CLASS_ID = env_int("SINGLE_CLASS_ID", 0)

# Modelo (config do model zoo do Detectron2)
MODEL_CONFIG = env_str("MODEL_CONFIG", "COCO-Detection/retinanet_R_50_FPN_3x.yaml")

# Treinamento
MAX_ITER = env_int("MAX_ITER", 300)                          # Detectron usa iterações, não épocas
BASE_LR = env_float("BASE_LR", 0.00025)                      # Taxa de aprendizado inicial
IMGS_PER_BATCH = env_int("IMGS_PER_BATCH", 2)                # Imagens por batch
NUM_WORKERS = env_int("NUM_WORKERS", 2)                      # Processos do dataloader
DEVICE = env_str("DEVICE", "cuda")                           # "cuda" ou "cpu"
SEED = env_int("SEED", -1)                                   # -1 = aleatório

# Avaliação
SCORE_THRESH = env_float("SCORE_THRESH", 0.5)  # Confiança mínima das predições

# Diretórios de Saída
OUTPUT_DIR = env_str("OUTPUT_DIR", f"RetinaNet_Treinamentos_{DATASET_NAME}")

# Flags de Execução
FIX_DATASET = env_bool("FIX_DATASET", False)   # True: Prepara o dataset
DO_TRAIN = env_bool("DO_TRAIN", True)          # True: TREINA O MODELO
DO_EVAL = env_bool("DO_EVAL", True)            # True: calcula mAP, matriz de confusão e curva PR
DO_ANALYSIS = env_bool("DO_ANALYSIS", True)    # True: gera os gráficos de análise

TRAIN_SET = f"{DATASET_NAME}_train"
VAL_SET = f"{DATASET_NAME}_val"


def register_datasets(classes):
    """
        Registers the train and valid splits in the Detectron2 catalogs, converting the YOLO
        labels on demand through get_yolo_dicts.

    Args:
        classes (list[str]): Class names, in the same order as the YOLO class IDs.
    """
    class_mapping = {str(i): i for i in range(len(classes))}

    DatasetCatalog.register(TRAIN_SET, lambda: get_yolo_dicts(TRAIN_IMG_DIR, TRAIN_LBL_DIR, class_mapping))
    MetadataCatalog.get(TRAIN_SET).set(thing_classes=classes)
    print(f"[+] Conjunto de treino '{TRAIN_SET}' registrado.")

    DatasetCatalog.register(VAL_SET, lambda: get_yolo_dicts(VAL_IMG_DIR, VAL_LBL_DIR, class_mapping))
    MetadataCatalog.get(VAL_SET).set(thing_classes=classes)
    print(f"[+] Conjunto de validação '{VAL_SET}' registrado.")


def main():
    """
        Runs the pipeline steps enabled by the flags above: dataset preparation (FIX_DATASET),
        training (DO_TRAIN), COCO evaluation (DO_EVAL) and loss plots (DO_ANALYSIS), in that order.
    """
    print("="*50)
    print(" PIPELINE DE DETECÇÃO - RETINANET (DETECTRON2) ")
    print("="*50)

    # Preparação do Dataset
    if FIX_DATASET:
        print("\n--- PREPARAÇÃO DO DATASET ---")
        utils = DatasetUtils(dataset_path=DATASET_PATH)

        # Fundindo todas as classes para uma detecção genérica
        if FORCE_SINGLE_CLASS:
            utils.fix_labels(target_class_id=SINGLE_CLASS_ID)
            utils.update_yaml(yaml_path=DATASET_YAML, single_class_name=SINGLE_CLASS_NAME)
        else:
            # Atualizando as pastas no yaml sem mudar as classes originais
            utils.update_yaml(yaml_path=DATASET_YAML)

        # Fazendo split 70/20/10 dinâmico
        utils.split_dataset(val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)

    # Registro do Dataset
    print("\n--- REGISTRO DO DATASET ---")
    with open(DATASET_YAML, 'r') as f:
        classes = yaml.safe_load(f).get('names', ["object"])
    register_datasets(classes)

    trainer = DetectronTrainer(num_classes=len(classes), output_dir=OUTPUT_DIR, model_config=MODEL_CONFIG)
    trainer.setup_datasets(TRAIN_SET, VAL_SET)
    trainer.configure_hyperparameters(
        base_lr=BASE_LR,
        max_iter=MAX_ITER,
        imgs_per_batch=IMGS_PER_BATCH,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        seed=SEED
    )

    # Treinamento
    if DO_TRAIN:
        print("\n--- TREINAMENTO ---")
        trainer.train()

    # Avaliação
    if DO_EVAL:
        print("\n--- AVALIAÇÃO E MÉTRICAS ---")
        evaluator = YoloEvaluator(trainer.get_config(), VAL_SET)
        evaluator.evaluate(score_thresh=SCORE_THRESH)

    # Análise
    if DO_ANALYSIS:
        print("\n--- ANÁLISE DOS RESULTADOS ---")
        analyzer = DetectronAnalyzer(output_dir=OUTPUT_DIR)
        analyzer.plot_training_curves()

    print("\nPipeline finalizado!")

if __name__ == "__main__":
    main()
