import os
import glob
import random
import shutil
import yaml

class DatasetUtils:
    def __init__(self, dataset_path):
        """
            Keeps the dataset root (the folder that holds train/, valid/, test/ and data.yaml) so the
            other methods can find the images and labels.

        Args:
            dataset_path (str): Path to the dataset root directory.
        """
        self.dataset_path = dataset_path

    def fix_labels(self, target_class_id=0):
        """
            Rewrites every YOLO label file under the dataset so all boxes use target_class_id,
            turning a multi-class dataset into a single generic class. Files are changed in place.

        Args:
            target_class_id (int): Class ID written in every label line.
        """
        print(f"Corrigindo as labels para a classe única {target_class_id}")
        label_files = glob.glob(os.path.join(self.dataset_path, "**", "labels", "*.txt"), recursive=True)

        modified_count = 0
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()

            new_lines = []
            changed = False
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                if int(parts[0]) != target_class_id:
                    parts[0] = str(target_class_id)
                    changed = True
                new_lines.append(" ".join(parts) + "\n")

            if changed:
                with open(label_file, 'w') as f:
                    f.writelines(new_lines)
                modified_count += 1

        print(f"Arquivos de label corrigidos: {modified_count}/{len(label_files)}")

    def update_yaml(self, yaml_path, single_class_name=None):
        """
            Points the train/val/test entries of data.yaml to the dataset's split folders (test is only
            added if the folder exists). If single_class_name is given, the classes are replaced by it.

        Args:
            yaml_path (str): Path to the dataset's data.yaml file.
            single_class_name (str | None): Name of the single generic class, or None to keep the
                original classes.
        """
        print(f"[*] Atualizando arquivo {yaml_path}...")
        if not os.path.exists(yaml_path):
            print(f"[!] Erro: {yaml_path} não encontrado.")
            return

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if single_class_name:
            data['nc'] = 1
            data['names'] = [single_class_name]

        data['train'] = '../train/images'
        data['val'] = '../valid/images'

        test_dir = os.path.join(os.path.dirname(yaml_path), "test")
        if os.path.exists(test_dir):
            data['test'] = '../test/images'

        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, sort_keys=False)

        print("Arquivo YAML atualizado com sucesso!")

    def split_dataset(self, val_ratio=0.20, test_ratio=0.10):
        """
            Moves a random share of train/ images (and their labels) into valid/ and test/, e.g.
            val_ratio=0.2 and test_ratio=0.1 give Train=0.7, Val=0.2, Test=0.1. Does nothing if
            valid/ already has images, so running it twice doesn't split again.

        Args:
            val_ratio (float): Fraction of the train images moved to valid/.
            test_ratio (float): Fraction of the train images moved to test/ (0 disables the test set).
        """
        train_img_dir = os.path.join(self.dataset_path, "train", "images")
        train_lbl_dir = os.path.join(self.dataset_path, "train", "labels")

        val_img_dir = os.path.join(self.dataset_path, "valid", "images")
        val_lbl_dir = os.path.join(self.dataset_path, "valid", "labels")

        test_img_dir = os.path.join(self.dataset_path, "test", "images")
        test_lbl_dir = os.path.join(self.dataset_path, "test", "labels")

        if not os.path.exists(train_img_dir):
            print("Diretório de treino não encontrado.")
            return

        # Check se o dataset já foi splitado alguma vez
        if os.path.exists(val_img_dir) and len(os.listdir(val_img_dir)) > 0:
            print("Conjunto de validação já existe. Ignorando split.")
            return

        os.makedirs(val_img_dir, exist_ok=True)
        os.makedirs(val_lbl_dir, exist_ok=True)

        if test_ratio > 0:
            os.makedirs(test_img_dir, exist_ok=True)
            os.makedirs(test_lbl_dir, exist_ok=True)

        images = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        total_images = len(images)
        val_size = int(total_images * val_ratio)
        test_size = int(total_images * test_ratio)

        val_images = images[:val_size]
        test_images = images[val_size:val_size + test_size]

        print(f"Movendo {val_size} imagens para validação e {test_size} para teste")

        def move_files(file_list, dest_img_dir, dest_lbl_dir):
            """
                Moves each image and its matching .txt label (when it exists) out of train/.

            Args:
                file_list (list[str]): Image file names inside train/images.
                dest_img_dir (str): Destination directory for the images.
                dest_lbl_dir (str): Destination directory for the labels.

            Returns:
                int: Number of images moved.
            """
            moved = 0
            for img_name in file_list:
                src_img = os.path.join(train_img_dir, img_name)
                dst_img = os.path.join(dest_img_dir, img_name)
                shutil.move(src_img, dst_img)

                lbl_name = os.path.splitext(img_name)[0] + ".txt"
                src_lbl = os.path.join(train_lbl_dir, lbl_name)
                dst_lbl = os.path.join(dest_lbl_dir, lbl_name)

                if os.path.exists(src_lbl):
                    shutil.move(src_lbl, dst_lbl)
                moved += 1
            return moved

        moved_val = move_files(val_images, val_img_dir, val_lbl_dir)
        moved_test = move_files(test_images, test_img_dir, test_lbl_dir) if test_ratio > 0 else 0

        print(f"[+] Split concluído! Validação: {moved_val} amostras | Teste: {moved_test} amostras.")
