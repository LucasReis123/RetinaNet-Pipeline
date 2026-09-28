# RetinaNet - Detecção de Objetos

Pipeline de treinamento do **RetinaNet** (Detectron2) para detecção de objetos. É um detector de estágio único que introduziu a Focal Loss, que reduz o peso dos exemplos fáceis (fundo) para lidar com o desbalanceamento entre objetos e fundo. Ele prepara o dataset, converte as labels YOLO para o formato do Detectron2, treina, avalia com as métricas COCO e gera gráficos de análise, tudo a partir de um único ponto de entrada (`main.py`). O dataset usado nos resultados deste documento é o **Torso 21** (futebol de robôs, RoboCup), mas qualquer dataset no formato YOLO serve.

**Sumário**

1. [Estrutura do repositório](#1-estrutura-do-repositório)
2. [Como funciona o pipeline](#2-como-funciona-o-pipeline)
3. [Como rodar](#3-como-rodar)
4. [Parâmetros](#4-parâmetros)
5. [Resultados do treinamento (Torso 21)](#5-resultados-do-treinamento-torso-21)

---

## 1. Estrutura do repositório

```
VisaoComputacional/
├── Dateset/                     # Datasets no formato YOLO (um por pasta)
│   └── Torso21/
│       ├── data.yaml
│       ├── train/{images,labels}
│       ├── valid/{images,labels}   # criado pelo split
│       └── test/{images,labels}    # criado pelo split
└── RetinaNet/
    ├── main.py                  # Ponto de entrada: configurações, registro do dataset e orquestração
    ├── dataset_utils.py         # DatasetUtils: ajuste de labels, data.yaml e split train/val/test
    ├── dataset_mapper.py        # get_yolo_dicts: converte labels YOLO para o formato do Detectron2
    ├── trainer.py               # DetectronTrainer: monta o config do model zoo e treina
    ├── evaluator.py             # YoloEvaluator: métricas COCO, matriz de confusão e curva PR
    ├── analyzer.py              # DetectronAnalyzer: gráficos de loss a partir do metrics.json
    ├── requirements.txt         # Dependências Python (PyTorch e Detectron2 são instalados à parte)
    ├── Dockerfile               # Imagem de treinamento (compila o Detectron2; CUDA parametrizável)
    ├── docker-compose.yml       # Serviços train (GPU) e train-cpu
    ├── .env.example             # Todos os parâmetros com seus valores padrão
    └── RetinaNet_Treinamentos_<DATASET>/   # Saída de cada treinamento
```

---

## 2. Como funciona o pipeline

O `main.py` executa as etapas abaixo, em ordem. O registro do dataset sempre roda; as outras são ligadas por flags:

| Etapa | Flag | O que faz |
|-------|------|-----------|
| **Preparação do dataset** | `FIX_DATASET` | Atualiza o `data.yaml` para apontar para `train/`, `valid/` e `test/`. Se `FORCE_SINGLE_CLASS` estiver ativo, reescreve todas as labels para uma única classe genérica. Em seguida move uma fração aleatória de `train/` para `valid/` (`VAL_RATIO`) e `test/` (`TEST_RATIO`). Se `valid/` já tiver imagens, o split é ignorado. |
| **Registro do dataset** | — | Lê as classes do `data.yaml` e registra `<DATASET>_train` e `<DATASET>_val` no Detectron2. As labels YOLO (normalizadas) são convertidas para caixas em pixels. |
| **Treinamento** | `DO_TRAIN` | Parte dos pesos COCO do `MODEL_CONFIG` (baixados na primeira vez para `.cache/`) e treina por `MAX_ITER` iterações. Salva `model_final.pth` e `metrics.json` em `OUTPUT_DIR`. |
| **Avaliação** | `DO_EVAL` | Carrega `model_final.pth`, calcula AP, AP50 e AP75 (COCO) no conjunto de validação e salva `confusion_matrix_retinanet.png` e `pr_curve_retinanet.png`. |
| **Análise** | `DO_ANALYSIS` | Lê o `metrics.json` e salva `graficos_treinamento_retinanet.png` (loss total, de classificação e de regressão por iteração). |

> A preparação do dataset **altera os arquivos no disco** (move imagens e, com `FORCE_SINGLE_CLASS`, reescreve labels). Mantenha uma cópia do dataset original se precisar dele intacto.

> O Detectron2 conta **iterações**, não épocas. Uma época equivale a `imagens de treino / IMGS_PER_BATCH` iterações (no Torso 21: 5.567 / 2 ≈ 2.784 iterações).

---

## 3. Como rodar

### 3.1 Com Docker (recomendado)

**Pré-requisitos:** Docker com Compose v2 e, para GPU, driver NVIDIA + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
cd RetinaNet

# Copie os parâmetros e ajuste o que quiser
cp .env.example .env

# Treina na GPU (o primeiro build compila o Detectron2 e leva alguns minutos)
docker compose up --build

# Treina na CPU
docker compose --profile cpu up --build train-cpu
```

Os parâmetros também podem ser passados direto na linha de comando, sem editar o `.env`:

```bash
MAX_ITER=3000 OUTPUT_DIR=RetinaNet_3000it docker compose up
```

O container monta esta pasta em `/workspace/app` e o dataset (`DATASET_HOST_DIR`, padrão `../Dateset`) em `/workspace/datasets`, então os resultados aparecem direto em `RetinaNet_Treinamentos_<DATASET>/` no host, com o seu usuário como dono (`UID`/`GID`). Os pesos COCO baixados ficam em `.cache/` e são reaproveitados nas próximas execuções.

> Com o `OUTPUT_DIR` padrão, um novo treino **sobrescreve** o anterior. Use outro `OUTPUT_DIR` para preservar resultados.

**Escolhendo a versão de CUDA.** O Detectron2 não tem pacote pronto, então o Dockerfile o compila com o `nvcc` da imagem base. Por isso a versão de CUDA da base (`CUDA_VERSION`) tem que bater com a do PyTorch (`TORCH_CUDA`), e o `TORCH_VERSION` precisa existir para aquele CUDA:

| `CUDA_VERSION` | `TORCH_CUDA` | `TORCH_VERSION` / `TORCHVISION_VERSION` | Quando usar |
|----------------|--------------|-----------------------------------------|-------------|
| `11.8.0` | `cu118` | `2.7.1` / `0.22.1` | Drivers antigos (≥ 520) |
| `12.4.1` | `cu124` | `2.6.0` / `0.21.0` | Drivers ≥ 550 |
| `12.6.3` (padrão) | `cu126` | `2.9.1` / `0.24.1` | Drivers ≥ 560 |
| `12.8.1` | `cu128` | `2.9.1` / `0.24.1` | GPUs Blackwell (RTX 50xx) ou drivers ≥ 570 |

Só a combinação padrão foi testada. Exemplo com outra:

```bash
CUDA_VERSION=11.8.0 TORCH_CUDA=cu118 TORCH_VERSION=2.7.1 TORCHVISION_VERSION=0.22.1 docker compose build
```

**Arquiteturas de GPU.** `TORCH_CUDA_ARCH_LIST` define para quais GPUs o Detectron2 é compilado (`7.5` = RTX 20xx, `8.6` = RTX 30xx, `8.9` = RTX 40xx, `12.0` = RTX 50xx, esta última só com CUDA 12.8). O padrão cobre de Turing a Hopper. Colocar só a da sua GPU (ex.: `TORCH_CUDA_ARCH_LIST=8.6`) deixa o build bem mais rápido. Se faltar RAM na compilação, reduza `MAX_JOBS`.

O serviço `train-cpu` usa `ubuntu:22.04` como base e compila o Detectron2 sem CUDA.

### 3.2 Localmente (sem Docker)

```bash
cd RetinaNet
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126
pip install "setuptools<81" ninja
pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2.git"
pip install -r requirements.txt

python main.py
```

Sem variáveis de ambiente, valem os padrões definidos no topo do `main.py` (lá `FIX_DATASET` vem desligado e as outras etapas ligadas).

---

## 4. Parâmetros

Todos os valores do topo do `main.py` podem ser sobrescritos por variável de ambiente com o mesmo nome. A lista completa, com os padrões, está em [`.env.example`](.env.example).

| Grupo | Variáveis |
|-------|-----------|
| **Dataset** | `DATASET_NAME`, `DATASET_ROOT`, `VAL_RATIO`, `TEST_RATIO`, `FORCE_SINGLE_CLASS`, `SINGLE_CLASS_NAME`, `SINGLE_CLASS_ID` |
| **Modelo** | `MODEL_CONFIG` (ex.: `COCO-Detection/retinanet_R_101_FPN_3x.yaml`) |
| **Treino** | `MAX_ITER`, `BASE_LR`, `IMGS_PER_BATCH`, `NUM_WORKERS`, `DEVICE` (`cuda`, `cpu`), `SEED` (`-1` = aleatório) |
| **Avaliação** | `SCORE_THRESH` |
| **Saída** | `OUTPUT_DIR` |
| **Etapas** | `FIX_DATASET`, `DO_TRAIN`, `DO_EVAL`, `DO_ANALYSIS` |
| **Docker (build)** | `CUDA_VERSION`, `PYTHON_VERSION`, `TORCH_CUDA`, `TORCH_VERSION`, `TORCHVISION_VERSION`, `DETECTRON2_REF`, `TORCH_CUDA_ARCH_LIST`, `MAX_JOBS` |
| **Docker (execução)** | `GPU_COUNT`, `SHM_SIZE`, `UID`, `GID`, `DATASET_HOST_DIR` |

---

## 5. Resultados do Treinamento (Torso 21)

Esta seção apresenta o treinamento do **RetinaNet** no dataset **Torso 21**, configurado para detectar elementos e marcações em um campo de futebol de robôs (RoboCup). Resultados e pesos estão em `RetinaNet_Treinamentos_Torso21/`.

### 5.1 Visão Geral do Treinamento

* **Algoritmo:** RetinaNet (estágio único com Focal Loss, Detectron2).
* **Config do model zoo:** `COCO-Detection/retinanet_R_50_FPN_3x.yaml` (ResNet-50 + FPN, pré-treinado na COCO).
* **Dataset:** Torso 21 (6 classes: `L`, `T`, `X`, `ball`, `goalpost`, `robot`), 5.567 imagens de treino e 1.590 de validação.
* **Treinamento:** `300` iterações com `2` imagens por batch = 600 imagens vistas, cerca de **11% de uma época**.
* **Tempo de Treinamento:** ~0,30 s por iteração, **~1,5 minuto(s)** no total.
* **Hardware Utilizado:** GPU NVIDIA RTX 3060.

### 5.2 Hiperparâmetros

* **Iterações (`MAX_ITER`):** `300`
* **Imagens por batch (`IMGS_PER_BATCH`):** `2`
* **Taxa de aprendizado (`BASE_LR`):** `0.00025`, constante (sem queda).
* **Limiar de confiança na avaliação (`SCORE_THRESH`):** `0.5`

### 5.3 Resultados e Métricas Obtidas (COCO, conjunto de validação)

| Métrica | Valor |
|---------|-------|
| **AP (mAP@0.5:0.95)** | **13,7%** |
| **AP50 (mAP@0.5)** | **19,1%** |
| **AP75** | 15,1% |
| **AP small / medium / large** | 7,7% / 13,7% / 33,2% |
| **Loss total (1ª → última iteração registrada)** | 2,19 → 0,89 |

AP por classe (mAP@0.5:0.95):

| `L` | `T` | `X` | `ball` | `goalpost` | `robot` |
|-----|-----|-----|--------|------------|---------|
| 0,0 | 0,2 | 0,0 | 63,6 | 1,0 | 17,0 |

### 5.4 Análise Gráfica dos Resultados

A pasta `RetinaNet_Treinamentos_Torso21/` tem:

1. **`graficos_treinamento_retinanet.png`:** loss total, de classificação e de regressão por iteração. As três caem de forma consistente e ainda não estabilizaram, sinal de que o treino parou cedo.
2. **`confusion_matrix_retinanet.png`:** matriz de confusão multiclasse (IoU > 0.5), com uma linha/coluna de Background para falsos negativos e falsos positivos.
3. **`pr_curve_retinanet.png`:** curva Precisão-Revocação geral (todas as classes juntas).
4. **`coco_instances_results.json` / `instances_predictions.pth`:** predições brutas usadas no cálculo das métricas COCO.

### 5.5 Localização dos Pesos (Weights)

* **`RetinaNet_Treinamentos_Torso21/model_final.pth`**: pesos ao final das 300 iterações.

### 5.6 Conclusões

1. **Treino curto demais:** com ~11% de uma época, o modelo só aprendeu razoavelmente a classe `ball` (AP 63,6%). As interseções de linha (`L`, `T`, `X`), que são pequenas, ficaram com AP próximo de zero.
2. **Não é comparável ao YOLOv8/RT-DETR:** aqueles treinaram dezenas de épocas completas. Para uma comparação justa, o `MAX_ITER` precisa cobrir várias épocas (ex.: `MAX_ITER=30000` ≈ 10 épocas com batch 2), de preferência com queda de learning rate.
3. **Limiar de avaliação:** as métricas usam `SCORE_THRESH=0.5`. Um limiar menor (ex.: `0.05`, o padrão da COCO) tende a aumentar o AP, porque mantém mais predições na curva.
4. **Melhor dos três modelos Detectron2** com o mesmo orçamento de 300 iterações (AP 13,7% contra 11,2% do Faster R-CNN e 12,0% do Cascade R-CNN), e o único com AP relevante em `robot` (17,0%).
