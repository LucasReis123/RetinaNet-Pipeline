import os
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer

class DetectronTrainer:
    def __init__(self, num_classes=1, output_dir="./output",
                 model_config="COCO-Detection/retinanet_R_50_FPN_3x.yaml"):
        """
            Builds the Detectron2 config from a model zoo file, starting from its COCO pretrained
            weights and adapting the head to num_classes.

        Args:
            num_classes (int): Number of object classes in the dataset.
            output_dir (str): Where checkpoints, metrics.json and plots are saved.
            model_config (str): Model zoo config file (e.g. COCO-Detection/retinanet_R_101_FPN_3x.yaml).
        """
        self.cfg = get_cfg()
        self.output_dir = output_dir

        # RetinaNet de estágio único com Focal Loss (padrão: ResNet-50 + FPN)
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config))

        # Pesos pré-treinados na COCO
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config)

        self.cfg.MODEL.RETINANET.NUM_CLASSES = num_classes  # RetinaNet não tem ROI heads
        self.cfg.OUTPUT_DIR = self.output_dir

    def setup_datasets(self, train_dataset_name, val_dataset_name):
        """
            Tells the config which registered datasets to use for training and evaluation.

        Args:
            train_dataset_name (str): Name registered in DatasetCatalog for training.
            val_dataset_name (str): Name registered in DatasetCatalog for evaluation.
        """
        self.cfg.DATASETS.TRAIN = (train_dataset_name,)
        self.cfg.DATASETS.TEST = (val_dataset_name,)

    def configure_hyperparameters(self, base_lr=0.00025, max_iter=300, num_workers=2, imgs_per_batch=2,
                                  device="cuda", seed=-1):
        """
            Sets the solver and dataloader values. The learning rate is kept constant (no drop
            steps).

        Args:
            base_lr (float): Initial learning rate.
            max_iter (int): Total training iterations.
            num_workers (int): Number of dataloader worker processes.
            imgs_per_batch (int): Images per batch.
            device (str): "cuda" or "cpu".
            seed (int): Random seed (-1 picks a random one).
        """
        self.cfg.DATALOADER.NUM_WORKERS = num_workers
        self.cfg.SOLVER.IMS_PER_BATCH = imgs_per_batch
        self.cfg.SOLVER.BASE_LR = base_lr
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.SOLVER.STEPS = []  # Sem queda de learning rate
        self.cfg.MODEL.DEVICE = device
        self.cfg.SEED = seed

    def train(self):
        """
            Trains from the pretrained weights (never resumes) and saves model_final.pth in
            output_dir.
        """
        print(f"Iniciando treinamento do RetinaNet no diretório: {self.output_dir}")
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        print("Treinamento finalizado.")

    def get_config(self):
        """
            Returns the config used by the trainer, so the evaluator runs with the same settings.

        Returns:
            detectron2.config.CfgNode: The Detectron2 config.
        """
        return self.cfg
