import os
import cv2
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor

class YoloEvaluator:
    def __init__(self, cfg, val_dataset_name):
        """
            Keeps the training config and the validation set used to evaluate the trained model.

        Args:
            cfg (detectron2.config.CfgNode): Config returned by DetectronTrainer.get_config().
            val_dataset_name (str): Name registered in DatasetCatalog for validation.
        """
        self.cfg = cfg
        self.val_dataset_name = val_dataset_name

    def evaluate(self, score_thresh=0.5):
        """
            Loads model_final.pth from the output directory, computes the COCO metrics on the
            validation set and saves the confusion matrix and PR curve.

        Args:
            score_thresh (float): Minimum confidence for a prediction to be kept.

        Returns:
            dict: COCO results from Detectron2 (results['bbox'] has AP, AP50, AP75, ...).
        """
        print(f"Iniciando avaliação no dataset: {self.val_dataset_name}")

        # Pesos do treino atual
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        self.cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_thresh

        predictor = DefaultPredictor(self.cfg)
        evaluator = COCOEvaluator(self.val_dataset_name, output_dir=self.cfg.OUTPUT_DIR)
        val_loader = build_detection_test_loader(self.cfg, self.val_dataset_name)

        print("Calculando métricas COCO (mAP)...")
        results = inference_on_dataset(predictor.model, val_loader, evaluator)

        print("\n--- RESUMO DA AVALIAÇÃO (RETINANET) ---")
        if "bbox" in results:
            print(f"mAP (AP): {results['bbox']['AP']:.4f}")
            print(f"mAP@0.5 (AP50): {results['bbox']['AP50']:.4f}")
            print(f"mAP@0.75 (AP75): {results['bbox']['AP75']:.4f}")

        print("\nGerando matriz de confusão e curva PR...")
        self._generate_custom_plots(predictor)
        return results

    def _generate_custom_plots(self, predictor, iou_threshold=0.5):
        """
            Runs the predictor on every validation image and saves a multiclass confusion matrix
            (with a Background row/column for misses and false positives) and an overall
            Precision-Recall curve in the output directory.

        Args:
            predictor (detectron2.engine.DefaultPredictor): Predictor with the trained weights.
            iou_threshold (float): Minimum IoU for a prediction to match a ground-truth box.
        """
        classes = MetadataCatalog.get(self.val_dataset_name).get("thing_classes", ["object"])
        num_classes = len(classes)

        # Linha = classe real, coluna = classe prevista. Último índice = Background
        cm = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

        dataset_dicts = DatasetCatalog.get(self.val_dataset_name)

        all_scores = []
        all_matches = []  # 1 = acerto (IoU e classe), 0 = erro
        total_gt = 0

        for d in dataset_dicts:
            img = cv2.imread(d["file_name"])
            if img is None:
                continue
            outputs = predictor(img)

            # Ground truths
            annotations = d["annotations"]
            gt_boxes = [obj["bbox"] for obj in annotations]
            gt_classes = [obj["category_id"] for obj in annotations]

            gt_boxes = torch.tensor(gt_boxes) if len(gt_boxes) > 0 else torch.empty((0, 4))
            total_gt += len(gt_boxes)

            # Predições
            instances = outputs["instances"].to("cpu")
            pred_boxes = instances.pred_boxes.tensor
            pred_classes = instances.pred_classes.numpy() if instances.has("pred_classes") else np.array([])
            scores = instances.scores.numpy() if instances.has("scores") else np.array([])

            # Sem predições: todo GT é falso negativo
            if len(pred_boxes) == 0:
                for gt_c in gt_classes:
                    cm[gt_c, num_classes] += 1
                continue

            # Sem GT: toda predição é falso positivo
            if len(gt_boxes) == 0:
                for i, pred_c in enumerate(pred_classes):
                    cm[num_classes, pred_c] += 1
                    all_scores.append(scores[i])
                    all_matches.append(0)
                continue

            iou_matrix = torchvision.ops.box_iou(pred_boxes, gt_boxes)

            # Associa primeiro as predições mais confiantes
            sorted_indices = torch.argsort(instances.scores, descending=True)

            gt_matched = set()
            for idx in sorted_indices:
                i = idx.item()
                pred_c = pred_classes[i]
                all_scores.append(scores[i])

                # GT com maior IoU para esta predição
                ious = iou_matrix[i]
                best_gt_idx = ious.argmax().item()
                best_iou = ious[best_gt_idx].item()

                if best_iou >= iou_threshold and best_gt_idx not in gt_matched:
                    gt_matched.add(best_gt_idx)
                    gt_c = gt_classes[best_gt_idx]
                    cm[gt_c, pred_c] += 1
                    all_matches.append(1 if gt_c == pred_c else 0)
                else:
                    # Sem match: falso positivo
                    cm[num_classes, pred_c] += 1
                    all_matches.append(0)

            # GTs sem predição: falso negativo
            for j, gt_c in enumerate(gt_classes):
                if j not in gt_matched:
                    cm[gt_c, num_classes] += 1

        # Matriz de confusão
        plt.figure(figsize=(num_classes + 3, num_classes + 2))
        labels = classes + ['Background']
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels,
                    yticklabels=labels)
        plt.ylabel('Real (Ground Truth)')
        plt.xlabel('Previsto (Predição)')
        plt.title(f'Matriz de Confusão Multiclasse (IoU > {iou_threshold})')
        cm_path = os.path.join(self.cfg.OUTPUT_DIR, "confusion_matrix_retinanet.png")
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()

        # Curva Precision-Recall geral
        if len(all_scores) > 0:
            indices = np.argsort(all_scores)[::-1]
            all_matches = np.array(all_matches)[indices]

            cum_tp = np.cumsum(all_matches == 1)
            cum_fp = np.cumsum(all_matches == 0)

            precision = cum_tp / (cum_tp + cum_fp + 1e-6)
            recall = cum_tp / (total_gt + 1e-6)

            plt.figure(figsize=(7, 5))
            plt.plot(recall, precision, color='blue', linewidth=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Curva Precision-Recall Geral (RetinaNet)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xlim(0, 1.05)
            plt.ylim(0, 1.05)
            pr_path = os.path.join(self.cfg.OUTPUT_DIR, "pr_curve_retinanet.png")
            plt.savefig(pr_path, bbox_inches='tight')
            plt.close()
            print(f"[+] Gráficos salvos em: {cm_path} e {pr_path}")
        else:
            print("[!] Nenhuma predição foi feita pelo modelo para plotar a Curva PR.")
