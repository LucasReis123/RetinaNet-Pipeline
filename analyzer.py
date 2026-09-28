import os
import json
import matplotlib.pyplot as plt

class DetectronAnalyzer:
    def __init__(self, output_dir):
        """
            Points the analyzer to the Detectron2 output directory, whose metrics.json is used to
            plot the training losses.

        Args:
            output_dir (str): Directory where Detectron2 saved the training outputs.
        """
        self.output_dir = output_dir
        self.metrics_file = os.path.join(self.output_dir, "metrics.json")

    @staticmethod
    def _get_loss(data, name):
        """
            Reads a loss from one metrics.json entry. Models with several stages (Cascade R-CNN)
            log name_stage0, name_stage1, ... instead of name, so those are summed.

        Args:
            data (dict): One line of metrics.json.
            name (str): Loss name, e.g. 'loss_cls' or 'loss_box_reg'.

        Returns:
            float: The loss value, or 0 if it isn't logged.
        """
        if name in data:
            return data[name]
        return sum(v for k, v in data.items() if k.startswith(f"{name}_stage"))

    def plot_training_curves(self):
        """
            Plots total, classification and box regression loss per iteration from metrics.json and
            saves the figure as graficos_treinamento_retinanet.png in output_dir.
        """
        print(f"Gerando gráficos de análise a partir de {self.metrics_file}")

        if not os.path.exists(self.metrics_file):
            print(f"[!] Arquivo {self.metrics_file} não encontrado. O treinamento completou?")
            return

        iters = []
        total_loss = []
        loss_cls = []
        loss_box_reg = []

        # Um JSON por linha
        with open(self.metrics_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Só linhas com loss
                    if 'total_loss' in data and 'iteration' in data:
                        iters.append(data['iteration'])
                        total_loss.append(data['total_loss'])
                        loss_cls.append(self._get_loss(data, 'loss_cls'))
                        loss_box_reg.append(self._get_loss(data, 'loss_box_reg'))
                except json.JSONDecodeError:
                    continue

        if not iters:
            print("[!] Nenhuma métrica válida encontrada no arquivo.")
            return

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Análise do Treinamento RetinaNet (Detectron2)', fontsize=16)

        # Total Loss
        axes[0].plot(iters, total_loss, label='Total Loss', color='red')
        axes[0].set_title('Loss Total')
        axes[0].set_xlabel('Iterações')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # Class Loss
        axes[1].plot(iters, loss_cls, label='Class Loss', color='blue')
        axes[1].set_title('Loss de Classificação')
        axes[1].set_xlabel('Iterações')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.7)

        # Box Loss (Regressão)
        axes[2].plot(iters, loss_box_reg, label='Box Reg Loss', color='orange')
        axes[2].set_title('Loss de Regressão (Box)')
        axes[2].set_xlabel('Iterações')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        custom_plot_path = os.path.join(self.output_dir, "graficos_treinamento_retinanet.png")
        plt.savefig(custom_plot_path)
        print(f"[+] Gráfico de análise salvo em: {custom_plot_path}")

if __name__ == "__main__":
    analyzer = DetectronAnalyzer("RetinaNet_Treinamentos")
    analyzer.plot_training_curves()
