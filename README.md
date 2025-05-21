# CohereClassifier

## English

### Description

This project implements a text classification pipeline with support for various linguistic preprocessing types (POS, RST, etc.), using Transformer-based models. It includes scripts for training, inference, dataset translation, and data processing, as well as integration with discourse parsing frameworks.

---

### Directory Structure

```
.
├── classifier.py              # Main script for training classification models
├── custom_pipeline.py         # Custom pipeline for classification using HuggingFace Transformers
├── data_processor/            # Modules for preprocessing and dataset manipulation
│   ├── adversarial_filter/    # Scripts for adversarial data filtering
│   ├── dmrst_parser/          # Discourse parser (RST), includes README and training/inference scripts
│   ├── ftbr.py                # FakeTrueBr dataset handling
│   ├── gcdc.py                # GCDC dataset handling
│   ├── pos_mix.py, pos_tags.py# POS tags manipulation scripts
│   ├── rst_mix.py, rst_tags.py# RST tags manipulation scripts
│   ├── translated.py          # Translated datasets handling
│   └── commonstories.py       # CommonStories dataset handling
├── figs/                      # Figures and plots generated during experiments
├── infer.sh                   # Shell script for batch inference
├── infer_checkpoints.py       # Script for inference using trained model checkpoints
├── translate.py               # Script for automatic dataset translation
├── .gitignore                 # Git ignore file
└── README.md                  # This file
```

---

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd cohereclassifier
   ```
2. Install dependencies (virtual environment recommended):
   ```bash
   pip install -r requirements.txt
   ```
   > **Note:** Also install specific dependencies for the parsing modules (see `data_processor/dmrst_parser/README.md`).

---

### Main Scripts and Usage

#### 1. Classifier Training

```bash
python classifier.py --dataset_path <path_to_dataset> [options]
```
**Main options:**
- `--batch_size`: Batch size (default: 5)
- `--epochs`: Number of epochs (default: 10)
- `--tokenizer`: HuggingFace tokenizer name (default: xlm-roberta-longformer-base-16384)
- `--rst`: Use RST tags (flag)
- `--pos`: Use POS tags (flag)
- `--lokr`: Use LoKr (flag)
- `--lr`: Learning rate (default: 1e-5)
- `--wr`: Warmup rate (default: 0.0)
- `--processed`: Indicates if the dataset is already processed
- `--runs`: Number of runs (default: 5)
- `--cycles`: Number of cycles (default: 1)
- `--debug`: Debug mode

#### 2. Inference

To run batch inference using multiple checkpoints:
```bash
bash infer.sh
```
Or directly:
```bash
python infer_checkpoints.py --dataset_path <path_to_processed_dataset> --prefix <checkpoints_prefix>
```

#### 3. Dataset Translation

To translate a CSV file with a "text" column from English to Portuguese:
```bash
python translate.py --df <path_to_csv>
```
The translated file will be saved with the `_translated.csv` suffix.

---

### Preprocessing and Discourse Parsing

The `data_processor/dmrst_parser` directory contains a multilingual RST parser. See detailed usage, training, and inference instructions in `data_processor/dmrst_parser/README.md`.

---

### Figures

The `figs/` directory contains plots and visualizations generated during experiments, useful for result analysis.

---

### Notes

- For scripts that use GPU, ensure your environment is properly configured with CUDA.
- The project uses HuggingFace Transformers and integrates with WandB for experiment tracking.
- For details on discourse parsing, see the specific README in `data_processor/dmrst_parser/`.

---

---

## Português

### Descrição

Este projeto implementa um pipeline de classificação de textos com suporte a diferentes tipos de pré-processamento linguístico (POS, RST, etc.), utilizando modelos baseados em Transformers. Inclui scripts para treinamento, inferência, tradução de datasets e processamento de dados, além de integrações com frameworks de parsing discursivo.

---

### Estrutura de Diretórios

```
.
├── classifier.py              # Script principal para treinamento de modelos de classificação
├── custom_pipeline.py         # Pipeline customizado para classificação usando HuggingFace Transformers
├── data_processor/            # Módulos para pré-processamento e manipulação de datasets
│   ├── adversarial_filter/    # Scripts para filtragem adversarial de dados
│   ├── dmrst_parser/          # Parser e segmentador discursivo (RST), inclui README e scripts de treinamento/inferência
│   ├── ftbr.py                # Manipulação do dataset FakeTrueBr
│   ├── gcdc.py                # Manipulação do dataset GCDC
│   ├── pos_mix.py, pos_tags.py# Scripts para manipulação de tags POS
│   ├── rst_mix.py, rst_tags.py# Scripts para manipulação de tags RST
│   ├── translated.py          # Manipulação de datasets traduzidos
│   └── commonstories.py       # Manipulação do dataset CommonStories
├── figs/                      # Figuras e gráficos gerados durante experimentos
├── infer.sh                   # Script shell para rodar inferências em lote
├── infer_checkpoints.py       # Script para inferência usando checkpoints de modelos treinados
├── translate.py               # Script para tradução automática de datasets
├── .gitignore                 # Arquivos e pastas ignorados pelo git
└── README.md                  # Este arquivo
```

---

### Instalação

1. Clone o repositório:
   ```bash
   git clone <url-do-repositorio>
   cd cohereclassifier
   ```
2. Instale as dependências (recomenda-se uso de ambiente virtual):
   ```bash
   pip install -r requirements.txt
   ```
   > **Obs:** Certifique-se de instalar também as dependências específicas dos módulos de parsing (ver `data_processor/dmrst_parser/README.md`).

---

### Principais Scripts e Como Usar

#### 1. Treinamento de Classificador

```bash
python classifier.py --dataset_path <caminho_para_dataset> [opções]
```
**Opções principais:**
- `--batch_size`: Tamanho do batch (default: 5)
- `--epochs`: Número de épocas (default: 10)
- `--tokenizer`: Nome do tokenizer HuggingFace (default: xlm-roberta-longformer-base-16384)
- `--rst`: Usa tags RST (ação)
- `--pos`: Usa tags POS (ação)
- `--lokr`: Usa LoKr (ação)
- `--lr`: Taxa de aprendizado (default: 1e-5)
- `--wr`: Warmup rate (default: 0.0)
- `--processed`: Indica se o dataset já está processado
- `--runs`: Número de execuções (default: 5)
- `--cycles`: Número de ciclos (default: 1)
- `--debug`: Modo debug

#### 2. Inferência

Para rodar inferências em lote usando múltiplos checkpoints:
```bash
bash infer.sh
```
Ou diretamente:
```bash
python infer_checkpoints.py --dataset_path <caminho_para_dataset_processado> --prefix <prefixo_dos_checkpoints>
```

#### 3. Tradução de Datasets

Para traduzir um arquivo CSV com uma coluna "text" do inglês para o português:
```bash
python translate.py --df <caminho_para_csv>
```
O arquivo traduzido será salvo com o sufixo `_translated.csv`.

---

### Pré-processamento e Parsing Discursivo

O diretório `data_processor/dmrst_parser` contém um parser RST multilíngue. Veja instruções detalhadas de uso, treinamento e inferência em `data_processor/dmrst_parser/README.md`.

---

### Figuras

O diretório `figs/` contém gráficos e visualizações gerados durante os experimentos, úteis para análise de resultados.

---

### Observações

- Para rodar scripts que utilizam GPU, certifique-se de que o ambiente está corretamente configurado com CUDA.
- O projeto utiliza o framework HuggingFace Transformers e integrações com o WandB para tracking de experimentos.
- Para detalhes sobre parsing discursivo, consulte o README específico em `data_processor/dmrst_parser/`.

