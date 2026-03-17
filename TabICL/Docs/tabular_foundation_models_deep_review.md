# Deep Review: Small Language Models and Tabular Foundation Models for Supervised Learning

## Executive Summary

This comprehensive review examines the latest advances in small language models (SLMs) and tabular foundation models designed for supervised learning tasks, with particular emphasis on fraud detection and classification applications. We analyze state-of-the-art models with few billion parameters that support both numeric and categorical features, evaluate their availability through GitHub repositories and Hugging Face, and provide detailed guidance on fine-tuning approaches for proprietary data versus training from scratch.

Our analysis reveals that transformer-based tabular foundation models, particularly TabPFN-2.5, TabICLv2, and TabDPT, represent a paradigm shift in tabular machine learning. These models achieve competitive or superior performance compared to traditional gradient boosting methods (XGBoost, LightGBM, CatBoost) while requiring no hyperparameter tuning. TabPFN-2.5 achieves 97.6% AUC on classification benchmarks, while specialized fraud detection implementations report up to 99.2% accuracy on Ethereum fraud detection tasks. All three major models are openly available with comprehensive GitHub repositories and Hugging Face model checkpoints, enabling immediate deployment for research and evaluation purposes.

Key findings indicate that continued pre-training on real-world data significantly improves performance over synthetic-only training, with Real-TabPFN showing consistent gains across 29 OpenML benchmark datasets. Fine-tuning approaches vary by model: TabPFN-2.5 supports full fine-tuning with gradient-based adaptation, TabICLv2 emphasizes retrieval-augmented in-context learning, and TabDPT scales to datasets with up to 50,000 samples through self-supervised learning. For fraud detection specifically, these models demonstrate strong performance on imbalanced datasets, temporal distribution shifts, and multimodal feature combinations.

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Theoretical Foundations](#2-background-and-theoretical-foundations)
3. [Identified Models and Architectures](#3-identified-models-and-architectures)
4. [Feature Support and Technical Capabilities](#4-feature-support-and-technical-capabilities)
5. [Training Approaches: Fine-tuning vs. Training from Scratch](#5-training-approaches-fine-tuning-vs-training-from-scratch)
6. [Fraud Detection and Classification Performance](#6-fraud-detection-and-classification-performance)
7. [Implementation Resources: GitHub and Hugging Face](#7-implementation-resources-github-and-hugging-face)
8. [Comparative Analysis and Model Selection](#8-comparative-analysis-and-model-selection)
9. [Limitations and Future Directions](#9-limitations-and-future-directions)
10. [Conclusion](#10-conclusion)

## 1. Introduction

Tabular data remains the most prevalent format in enterprise machine learning applications, particularly in domains such as fraud detection, credit scoring, and risk assessment. Traditional approaches have relied heavily on gradient boosting decision trees (GBDT) such as XGBoost, LightGBM, and CatBoost, which require extensive hyperparameter tuning and domain expertise. Recent advances in foundation models have demonstrated that transformer-based architectures pre-trained on diverse datasets can achieve competitive or superior performance through in-context learning, eliminating the need for task-specific optimization [1], [15].

The emergence of tabular foundation models represents a convergence of several research directions: in-context learning from large language models, attention mechanisms for structured data, and synthetic data generation for pre-training. These models typically contain between 10 million and a few billion parameters, positioning them as "small language models" compared to text-based LLMs, yet they demonstrate remarkable generalization capabilities on tabular prediction tasks [13], [15].

This review addresses five critical questions for practitioners and researchers:

1. Which tabular foundation models with few billion parameters are currently available and actively maintained?
2. How do these models handle numeric and categorical features, and what are their technical limitations?
3. What GitHub repositories and Hugging Face models provide production-ready implementations?
4. What are the detailed approaches for fine-tuning on proprietary data versus training from scratch?
5. How do these models compare for fraud detection efficiency and classification accuracy?

Our analysis synthesizes findings from 191 recent papers (2024-2026), technical documentation from major open-source repositories, and empirical benchmarks on standardized datasets. We focus on three primary model families: TabPFN (including TabPFN-2.5 and Real-TabPFN), TabICLv2, and TabDPT, each representing distinct architectural and training philosophies.

## 2. Background and Theoretical Foundations

### 2.1 In-Context Learning for Tabular Data

In-context learning (ICL) enables models to adapt to new tasks by conditioning on a set of training examples provided in the input context, without updating model parameters [4], [11]. For tabular data, this paradigm treats each dataset as a sequence where the model observes training samples and their labels, then predicts labels for test samples. This approach contrasts sharply with traditional supervised learning, which requires separate training phases for each new dataset [15].

The theoretical foundation of ICL for tabular data rests on the hypothesis that transformers can learn to implement learning algorithms in their forward pass. TabPFN pioneered this approach by pre-training on synthetic datasets generated from a prior distribution over tabular tasks, enabling the model to internalize Bayesian inference procedures [15]. Subsequent work has extended this to real-world data through continued pre-training [1] and retrieval-augmented approaches [21], [23].

### 2.2 Transformer Architectures for Structured Data

Adapting transformers to tabular data requires addressing several architectural challenges: handling variable feature counts, encoding categorical variables, managing missing values, and scaling to large datasets. Modern tabular transformers employ specialized attention mechanisms that alternate between sample-wise and feature-wise attention, enabling the model to capture both row-level patterns and column-level dependencies [13], [15].

Feature tokenization strategies vary across models. TabPFN uses learned embeddings for categorical features combined with continuous encodings for numeric features [15]. TabICLv2 introduces quantile-based discretization that preserves distributional information while enabling efficient attention computation [21], [23]. TabDPT employs retrieval mechanisms to select relevant training samples, reducing computational complexity for large context sizes [20].

### 2.3 Synthetic vs. Real-World Pre-training

A fundamental design choice in tabular foundation models concerns the pre-training data source. Synthetic data generation offers several advantages: unlimited data availability, controlled distribution properties, and absence of privacy concerns. TabPFN's original approach generated synthetic datasets by sampling from a prior over structural causal models, enabling the model to learn general-purpose inference procedures [15].

However, recent work demonstrates that continued pre-training on real-world data yields substantial performance improvements. Real-TabPFN achieves 97.6% mean normalized ROC-AUC on OpenML benchmarks by incorporating 43 curated real-world datasets during continued pre-training, compared to 95.4% for the synthetic-only baseline [1]. This suggests that real-world data captures distributional properties and feature interactions that synthetic priors fail to model adequately [1].

## 3. Identified Models and Architectures

### 3.1 TabPFN-2.5 and Real-TabPFN

**Architecture**: TabPFN-2.5 employs a transformer architecture with TabPFNv2-like alternating attention mechanisms, comprising 18-24 layers [13]. The model uses in-context learning to solve tabular prediction problems in a single forward pass without parameter updates. Real-TabPFN retains this architecture while incorporating continued pre-training on real-world datasets [1].

**Parameter Count**: While exact parameter counts are not publicly disclosed, the model is designed to be computationally efficient, with inference times of seconds on datasets with thousands of samples [13], [15].

**Key Innovations**: TabPFN-2.5 introduces multiple specialized checkpoints optimized for different scenarios: large features (up to 1000 features), large samples (>30K rows), and real-data fine-tuning [13]. The default classification checkpoint achieves state-of-the-art results through continued pre-training on 43 real-world datasets [1], [13].

**Availability**: 
- GitHub: https://github.com/PriorLabs/TabPFN (5,800+ stars)
- Hugging Face: https://huggingface.co/Prior-Labs/tabpfn_2_5
- PyPI: `pip install tabpfn`

### 3.2 TabICLv2

**Architecture**: TabICLv2 represents a state-of-the-art tabular foundation model developed by SODA-INRIA. The model employs transformer-based architecture with specialized attention mechanisms for tabular data. It uses quantile-based feature encoding and retrieval-augmented in-context learning [21], [23].

**Parameter Count**: Specific parameter counts are not explicitly stated in available documentation, but the model is designed for efficiency and scalability [21], [23].

**Key Innovations**: TabICLv2 introduces improved scalability and speed compared to its predecessor, with better handling of large feature spaces and sample sizes. The model emphasizes open-source accessibility and reproducibility [21], [23].

**Availability**:
- GitHub: https://github.com/soda-inria/tabicl (602 stars)
- Documentation: https://tabicl.readthedocs.io/en/latest/
- PyPI: `pip install tabicl`

### 3.3 TabDPT

**Architecture**: TabDPT (Tabular Diffusion Pre-Training) uses retrieval and self-supervised learning to scale tabular foundation models on real data. The model employs transformer architecture with specialized mechanisms for handling large context sizes through retrieval-based sample selection [20].

**Parameter Count**: TabDPT is trained at multiple scales, with models ranging from smaller variants to larger versions. The paper demonstrates scaling laws similar to language models, showing consistent improvements with increased model size [20].

**Key Innovations**: TabDPT achieves 97.6% AUC on CC18 classification benchmark and 92.8% accuracy on CTR23, outperforming TabPFN v2, TabR, and gradient boosting methods. The model scales to datasets with up to 50,000 samples and 2,000 features [20].

**Availability**:
- GitHub: https://github.com/layer6ai-labs/TabDPT-inference (77 stars)
- Hugging Face: https://huggingface.co/Layer6/TabDPT
- PyPI: `pip install tabdpt`

### 3.4 Specialized Variants and Extensions

**TabPFN-Wide**: Extends TabPFN to handle extreme feature counts through continued pre-training, addressing limitations in high-dimensional scenarios [5].

**Drift-Resilient TabPFN**: Specifically designed for temporal distribution shifts in tabular data, addressing concept drift in production environments [19].

**FT-TabPFN**: Combines feature tokenization with TabPFN architecture, achieving improved performance on tabular classification tasks [10].

**ExplainerPFN**: Extends the TabPFN framework to provide model-free zero-shot feature importance estimations, addressing interpretability requirements in regulated domains [8].

## 4. Feature Support and Technical Capabilities

### 4.1 Numeric and Categorical Feature Handling

All three major tabular foundation models support mixed feature types, but employ different encoding strategies:

**TabPFN-2.5**: Handles categorical features through Scikit-learn's OrdinalEncoder, with learned embeddings for categorical variables combined with continuous encodings for numeric features [1], [13]. The model automatically detects feature types and applies appropriate preprocessing. Missing values are handled through imputation strategies, though specific details are not extensively documented [13].

**TabICLv2**: Employs quantile-based discretization for numeric features, preserving distributional information while enabling efficient attention computation [21], [23]. Categorical features are encoded using learned embeddings. The model supports mixed feature types without requiring manual preprocessing [21], [23].

**TabDPT**: Uses retrieval mechanisms to select relevant training samples, with feature encoding that supports both numeric and categorical variables [20]. The model handles missing values through masking mechanisms in the attention layers [20].

### 4.2 Dataset Size and Feature Limitations

**TabPFN-2.5**:
- Maximum samples: 50,000 (with sampling for larger datasets)
- Maximum features: 2,000 (specialized checkpoints support up to 1,000 features per estimator)
- Target classes: Up to 10 classes for classification
- Optimal range: <10,000 samples, <500 features [1], [13]

**TabICLv2**:
- Designed for scalability with improved handling of large feature spaces
- Supports datasets with thousands of samples and hundreds of features
- Specific limits not explicitly documented but emphasizes "better, faster, scalable" design [21], [23]

**TabDPT**:
- Maximum samples: 50,000
- Maximum features: 2,000
- Context size: Configurable (default 2048)
- Performance degrades beyond stated limits [20]

### 4.3 Computational Requirements

**Inference Speed**: TabPFN-2.5 achieves predictions in seconds on datasets with thousands of samples, significantly faster than hyperparameter-tuned gradient boosting methods [13], [15]. TabDPT demonstrates superior runtime performance compared to other foundation models, with total time-to-prediction substantially lower than TabPFN v2 [20].

**Hardware Requirements**: All models support both CPU and GPU inference, with GPU acceleration recommended for larger datasets. TabPFN-2.5 supports multi-GPU inference in auto mode [13]. TabDPT requires CUDA-capable GPUs for optimal performance [20].

**Memory Footprint**: Models are designed to fit on consumer-grade GPUs. TabPFN-2.5 checkpoints range from hundreds of megabytes to a few gigabytes depending on the variant [13].

## 5. Training Approaches: Fine-tuning vs. Training from Scratch

### 5.1 Fine-tuning Tabular Foundation Models

#### 5.1.1 Full Fine-tuning with TabPFN-2.5

Recent research establishes full fine-tuning as the most practical solution for TabPFN-2.5 in terms of time-efficiency and effectiveness [9]. The fine-tuning process involves gradient-based adaptation of all model parameters on target datasets.

**Technical Approach**:
- Optimizer: AdamW with reduced learning rate (3 × 10⁻⁷)
- Regularization: L2-Starting-Point regularizer to prevent catastrophic forgetting
- Learning rate schedule: Linear warm-up followed by cosine annealing
- Batch size: 1 (allowing varied feature dimensions without padding)
- Training data: Curated collection of real-world datasets from OpenML and Kaggle [1], [9]

**Implementation**: The Yandex Research repository provides complete fine-tuning code:

```bash
# Install dependencies
pip install uv
uv sync --extra cu124

# Download model checkpoints
wget https://huggingface.co/Prior-Labs/TabPFN-v2-class/resolve/main/tabpfn-v2-classifier.ckpt

# Run fine-tuning
uv run python bin/tabpfnv2_finetune.py exp/full-finetune/adult/evaluation/0.toml --force
```

**Performance Gains**: Fine-tuning enables TabPFN-2.5 to achieve state-of-the-art results on academic datasets with i.i.d. splits. On datasets with up to 50K samples, fine-tuning improves performance on almost all tasks [9]. The success stems from improved similarity computation: after gradient-based adaptation, query-representations of test objects and key-representations of in-context training objects more accurately reflect target similarity [9].

**Limitations**: On datasets with gradual temporal shifts and rich feature sets, fine-tuned TabPFN-2.5 shows less stability, and prior methods may remain superior [9].

#### 5.1.2 Retrieval and Fine-tuning for TabICL

TabICL models emphasize retrieval-augmented in-context learning, where the model selects relevant training samples to include in the context window [11]. Fine-tuning approaches for TabICL focus on optimizing the retrieval mechanism and context composition.

**Context Optimization**: TuneTables introduces context optimization for scalable Prior-Data Fitted Networks, enabling efficient adaptation to new datasets through learned context selection [12].

**Mixture of In-Context Prompters**: Recent work proposes mixture-of-experts style approaches where multiple context configurations are evaluated and combined [16].

#### 5.1.3 Continued Pre-training Strategies

Real-TabPFN demonstrates that continued pre-training on real-world data yields superior performance compared to broader, noisier corpora [1]. The approach involves:

1. **Data Curation**: Select high-quality, diverse real-world datasets (43 datasets in Real-TabPFN)
2. **Continued Training**: Resume training from synthetic pre-trained checkpoint with reduced learning rate
3. **Regularization**: Apply L2-Starting-Point regularization to preserve synthetic pre-training knowledge
4. **Validation**: Monitor performance on held-out benchmark datasets [1]

**Results**: Real-TabPFN achieves mean normalized ROC-AUC of 97.6% on OpenML AutoML Benchmark, compared to 95.4% for TabPFNv2 [1].

### 5.2 Training from Scratch

Training tabular foundation models from scratch requires substantial computational resources and carefully curated datasets. The process differs fundamentally from fine-tuning existing models.

#### 5.2.1 Synthetic Data Generation

TabPFN's original training approach generates synthetic datasets by sampling from a prior distribution over tabular tasks [15]. The process involves:

1. **Prior Definition**: Define structural causal models with configurable complexity
2. **Dataset Sampling**: Generate diverse synthetic datasets with varying feature counts, sample sizes, and label distributions
3. **Training**: Train transformer to predict labels given in-context examples
4. **Validation**: Evaluate on held-out synthetic and real-world datasets [15]

#### 5.2.2 Real-World Data Pre-training

TabDPT demonstrates successful training on real-world data through retrieval and self-supervised learning [20]. The approach involves:

1. **Data Collection**: Aggregate large-scale tabular datasets from multiple sources
2. **Retrieval Mechanism**: Implement efficient retrieval to select relevant training samples
3. **Self-supervised Objectives**: Design pre-training tasks that capture tabular structure
4. **Scaling**: Train models at multiple scales to observe scaling laws [20]

**Scaling Laws**: TabDPT demonstrates power-law relationships between model size, pre-training data size, and downstream performance, similar to language models [20].

#### 5.2.3 Practical Considerations

**Computational Cost**: Training from scratch requires orders of magnitude more computation than fine-tuning. TabDPT training involves multiple GPU-months of computation [20].

**Data Requirements**: Effective training from scratch requires thousands of diverse tabular datasets. Real-TabPFN uses 43 curated datasets for continued pre-training [1], while TabDPT trains on substantially larger corpora [20].

**Recommendation**: For most practitioners, fine-tuning existing models or using pre-trained checkpoints is more practical than training from scratch. Training from scratch is justified only when:
- Existing models fail to capture domain-specific patterns
- Proprietary data cannot be used for fine-tuning due to privacy constraints
- Substantial computational resources are available
- The target domain differs fundamentally from pre-training distributions

## 6. Fraud Detection and Classification Performance

### 6.1 Benchmark Performance on Standard Datasets

#### 6.1.1 OpenML Classification Benchmarks

Real-TabPFN achieves state-of-the-art performance on the OpenML AutoML Benchmark (CC18), demonstrating consistent improvements over baseline methods [1]:

- **Mean Normalized ROC-AUC**: 97.6% (95% CI: [97.4%, 97.8%])
- **Accuracy**: 93.2% (95% CI: [92.6%, 93.1%])
- **F1-Score**: 93.9%

These results surpass TabPFNv2 (95.4% ROC-AUC), XGBoost (96.5%), LightGBM (96.4%), and CatBoost (96.4%) [1].

TabDPT v1.1 achieves comparable performance on CC18 [20]:

- **AUC**: 97.6% (95% CI: [97.4%, 97.8%])
- **Accuracy**: 92.8% (95% CI: [92.6%, 93.1%])

#### 6.1.2 Regression Benchmarks

On the CTR23 regression benchmark, TabDPT v1.1 demonstrates strong performance [20]:

- **Correlation**: 92.0% (95% CI: [91.8%, 92.2%])
- **R²**: 84.7% (95% CI: [84.3%, 85.1%])

These results exceed TabPFN v2 (84.1% R²), TabR (82.5% R²), and gradient boosting methods (XGBoost: 82.0%, LightGBM: 80.9%, CatBoost: 80.2%) [20].

### 6.2 Fraud Detection Applications

#### 6.2.1 Ethereum Fraud Detection

Olusegun et al. demonstrate exceptional performance on Ethereum fraud detection using TabPFN [29]:

- **Accuracy**: 99.2%
- **Inference Time**: Seconds
- **Approach**: Interpretable Feature Selection with TabPFN (IFS-TABPFN)

The system combines Shap-based feature selection with TabPFN's in-context learning, outperforming traditional neural networks (MLP, LSTM, CNN, CLSTM) and existing fraud detection systems [29]. The interpretability component addresses regulatory requirements for explainable AI in financial applications [29].

#### 6.2.2 Rug Pull Detection in DeFi

Shoaei et al. apply TabPFN to early rug-pull detection in decentralized finance, integrating on-chain behavioral metrics with Open Source Intelligence (OSINT) signals [2]. The framework achieves:

- **Strong discriminative performance** compared to classical baselines
- **Improved probability calibration**
- **Low false-negative rates**

The leakage-resistant framework ensures temporal validity by extracting all features strictly prior to liquidity withdrawal events, addressing a critical limitation in prior fraud detection systems [2].

#### 6.2.3 Cybersecurity Applications

García et al. evaluate TabPFN and TabICL for tabular intrusion detection in cybersecurity contexts [24]. The study demonstrates that foundation models achieve competitive performance on network intrusion detection datasets, with TabPFN showing particular strength on small-sample scenarios [24].

### 6.3 Credit Scoring and Financial Applications

Li et al. extend TabPFN for credit scoring through class-imbalanced-aware adaptive dataset distillation [30]. The approach achieves:

- **AUC Improvement**: 2.5% enhancement over baseline TabPFN
- **Scalability**: Enables TabPFN application to larger credit scoring datasets
- **Imbalance Handling**: Integrates imbalance-aware techniques during dataset distillation [30]

This work addresses a critical limitation of TabPFN's original design, which struggles with large-scale imbalanced datasets common in financial applications [30].

### 6.4 Comparative Performance Analysis

Table 1 summarizes comparative performance across major tabular foundation models and baseline methods on standard benchmarks:

| Model | CC18 AUC | CC18 Accuracy | CTR23 Correlation | CTR23 R² |
|-------|----------|---------------|-------------------|----------|
| **TabDPT v1.1** | **0.976** | **0.928** | **0.920** | **0.847** |
| **Real-TabPFN** | **0.976** | **0.932** | - | - |
| TabPFN v2 | 0.972 | 0.917 | 0.917 | 0.841 |
| TabR | 0.967 | 0.923 | 0.909 | 0.825 |
| MLP-PLR | 0.967 | 0.914 | 0.907 | 0.827 |
| XGBoost | 0.965 | 0.910 | 0.904 | 0.820 |
| LightGBM | 0.964 | 0.906 | 0.900 | 0.809 |
| CatBoost | 0.964 | 0.908 | 0.897 | 0.802 |

*Note: Values represent mean performance with 95% confidence intervals available in source papers [1], [20].*

### 6.5 Performance on Imbalanced Datasets

Fraud detection tasks typically involve severe class imbalance, with fraudulent transactions representing <1% of total samples. Tabular foundation models demonstrate several advantages for imbalanced learning:

**In-Context Adaptation**: Models automatically adjust to class distributions observed in the context window, without requiring explicit rebalancing techniques [15].

**Calibrated Probabilities**: TabPFN and TabDPT produce well-calibrated probability estimates, critical for setting decision thresholds in fraud detection systems [2], [20].

**Few-Shot Learning**: Models perform well even when fraudulent examples are scarce, leveraging pre-trained knowledge to generalize from limited positive samples [15].

### 6.6 Temporal Distribution Shifts

Production fraud detection systems must handle concept drift as fraudsters adapt their strategies. Drift-Resilient TabPFN specifically addresses temporal distribution shifts through in-context learning on recent data [19]. The model maintains performance under gradual distribution shifts by conditioning on temporally proximate training samples [19].

However, fine-tuned TabPFN-2.5 shows reduced stability on datasets with gradual temporal shifts compared to traditional methods [9], suggesting that in-context learning may be preferable to fine-tuning for non-stationary environments.

## 7. Implementation Resources: GitHub and Hugging Face

### 7.1 TabPFN-2.5 Implementation

#### 7.1.1 Main Repository

**GitHub**: https://github.com/PriorLabs/TabPFN (5,800+ stars)

**Key Features**:
- Scikit-learn compatible API
- Multi-GPU support
- Multiple specialized checkpoints
- Comprehensive documentation and examples
- Active maintenance (755 commits, 54 contributors)

**Installation**:
```bash
pip install tabpfn
```

**Basic Usage**:
```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize classifier
clf = TabPFNClassifier()  # Uses TabPFN 2.5 weights
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
prediction_probabilities = clf.predict_proba(X_test)
```

#### 7.1.2 Hugging Face Models

**Model Hub**: https://huggingface.co/Prior-Labs/tabpfn_2_5

**Available Checkpoints**:
- `tabpfn-v2.5-classifier-v2.5_default.ckpt`: Default classification (real-data fine-tuned)
- `tabpfn-v2.5-classifier-v2.5_large-features-XL.ckpt`: Up to 1000 features
- `tabpfn-v2.5-classifier-v2.5_large-samples.ckpt`: >30K samples
- `tabpfn-v2.5-regressor-v2.5_default.ckpt`: Default regression
- Multiple specialized variants for specific scenarios [13]

**Licensing**: Models released under tabpfn-2.5-license-v1.0, permissive for research and internal evaluation. Commercial use requires enterprise license from sales@priorlabs.ai [13].

#### 7.1.3 Fine-tuning Repository

**GitHub**: https://github.com/yandex-research/tabpfn-finetuning (34 stars)

**Key Features**:
- Complete fine-tuning pipeline
- TOML-based configuration
- Evaluation scripts
- Reproducible experiments

**Installation**:
```bash
git clone https://github.com/yandex-research/tabpfn-finetuning
cd tabpfn-finetuning
pip install uv
uv sync --extra cu124
```

**Running Fine-tuning**:
```bash
# Download checkpoints
wget https://huggingface.co/Prior-Labs/TabPFN-v2-class/resolve/main/tabpfn-v2-classifier.ckpt

# Execute fine-tuning
uv run python bin/tabpfnv2_finetune.py exp/full-finetune/adult/evaluation/0.toml --force
```

### 7.2 TabICLv2 Implementation

#### 7.2.1 Main Repository

**GitHub**: https://github.com/soda-inria/tabicl (602 stars)

**Key Features**:
- State-of-the-art tabular foundation model
- Improved scalability and speed
- Comprehensive documentation
- Active development (93 commits, 10 contributors)

**Installation**:
```bash
pip install tabicl
```

**Documentation**: https://tabicl.readthedocs.io/en/latest/

**Basic Usage**:
```python
from tabicl import TabICLClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit
clf = TabICLClassifier()
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
```

#### 7.2.2 Training Code

The repository includes open-source training code in the `scripts/` directory, enabling researchers to reproduce training procedures or adapt them for custom datasets [21], [23].

### 7.3 TabDPT Implementation

#### 7.3.1 Inference Repository

**GitHub**: https://github.com/layer6ai-labs/TabDPT-inference (77 stars)

**Key Features**:
- Lightweight inference interface
- Scikit-learn compatible API
- Jupyter notebook examples
- Comprehensive benchmarking scripts

**Installation**:
```bash
pip install tabdpt
```

**Basic Usage**:
```python
from tabdpt import TabDPTClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize classifier
clf = TabDPTClassifier(context_size=2048, n_ensembles=8)
clf.fit(X_train, y_train)

# Predict
predictions = clf.predict(X_test)
```

#### 7.3.2 Hugging Face Models

**Model Hub**: https://huggingface.co/Layer6/TabDPT

Model weights are automatically downloaded on first use. The repository provides multiple model variants optimized for different dataset sizes and feature counts [20].

#### 7.3.3 Training Repository

**GitHub**: https://github.com/layer6ai-labs/TabDPT-training/

Full training code is available separately, enabling reproduction of pre-training procedures and experimentation with custom datasets [20].

### 7.4 Specialized Extensions

#### 7.4.1 TabPFN Extensions

**Repository**: https://github.com/PriorLabs/tabpfn-extensions

Provides additional functionality including:
- AutoTabPFN for automatic hyperparameter optimization
- Post-hoc ensemble methods
- Advanced preprocessing utilities [13]

#### 7.4.2 ExplainerPFN

**Focus**: Model-free zero-shot feature importance estimation

**Application**: Addresses interpretability requirements in regulated domains such as finance and healthcare [8].

### 7.5 Practical Deployment Considerations

#### 7.5.1 Production Deployment

For production deployment, consider:

1. **Licensing**: Verify license compatibility (research vs. commercial use)
2. **Inference Speed**: Benchmark on representative datasets
3. **Model Selection**: Choose appropriate checkpoint for dataset characteristics
4. **Monitoring**: Implement drift detection for temporal distribution shifts
5. **Fallback**: Maintain traditional models as fallback for edge cases

#### 7.5.2 Integration with MLOps Pipelines

All three major models provide scikit-learn compatible APIs, enabling seamless integration with existing MLOps infrastructure:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', TabPFNClassifier())
])

# Use in standard ML workflow
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## 8. Comparative Analysis and Model Selection

### 8.1 Performance Comparison

Table 2 provides a comprehensive comparison of the three major tabular foundation models across key dimensions:

| Dimension | TabPFN-2.5 | TabICLv2 | TabDPT |
|-----------|------------|----------|--------|
| **Architecture** | Transformer (18-24 layers) | Transformer with quantile encoding | Transformer with retrieval |
| **Max Samples** | 50,000 | Thousands | 50,000 |
| **Max Features** | 2,000 (1,000 per estimator) | Hundreds | 2,000 |
| **Training Data** | Synthetic + 43 real datasets | Real-world data | Large-scale real data |
| **Fine-tuning** | Full fine-tuning supported | Retrieval optimization | Self-supervised adaptation |
| **Inference Speed** | Seconds | Fast | Fastest |
| **GitHub Stars** | 5,800+ | 602 | 77 |
| **License** | Research + Commercial | Open source | Apache 2.0 |
| **Best For** | General classification, small-medium datasets | Scalable applications | Large datasets, regression |

### 8.2 Model Selection Guidelines

#### 8.2.1 Choose TabPFN-2.5 When:

- Dataset has <10,000 samples and <500 features
- Classification task with up to 10 classes
- Interpretability is important (via ExplainerPFN)
- Established ecosystem and community support are priorities
- Real-data fine-tuning is desired
- Commercial deployment is planned (enterprise license available)

**Strengths**:
- Highest community adoption (5,800+ GitHub stars)
- Most comprehensive documentation
- Multiple specialized checkpoints
- Strong performance on small-medium datasets
- Active maintenance and development

**Limitations**:
- Performance degrades on datasets with >50,000 samples
- Limited to 10 classes for classification
- Commercial use requires enterprise license

#### 8.2.2 Choose TabICLv2 When:

- Scalability and speed are critical
- Open-source licensing is required
- Research reproducibility is important
- Integration with INRIA ecosystem is beneficial

**Strengths**:
- Improved scalability over TabPFN
- Fully open-source
- Strong academic backing (SODA-INRIA)
- Comprehensive documentation

**Limitations**:
- Smaller community compared to TabPFN
- Less extensive benchmark results available
- Fewer specialized variants

#### 8.2.3 Choose TabDPT When:

- Dataset has 10,000-50,000 samples
- Regression tasks are primary focus
- Fastest inference speed is required
- Scaling laws and model size flexibility are important
- Apache 2.0 licensing is preferred

**Strengths**:
- Best performance on regression benchmarks
- Fastest inference speed
- Demonstrated scaling laws
- Permissive Apache 2.0 license
- Strong performance on large datasets

**Limitations**:
- Smaller community (77 GitHub stars)
- Less extensive documentation
- Fewer specialized variants
- Training from scratch requires substantial resources

### 8.3 Fraud Detection Specific Recommendations

For fraud detection applications, model selection should consider:

#### 8.3.1 Ethereum/Blockchain Fraud

**Recommended**: TabPFN-2.5
- Proven 99.2% accuracy on Ethereum fraud detection [29]
- Interpretability through Shap integration [29]
- Handles multimodal features (on-chain + OSINT) [2]

#### 8.3.2 Credit Card Fraud

**Recommended**: TabDPT or Real-TabPFN
- Strong performance on imbalanced datasets
- Calibrated probability estimates for threshold tuning
- Handles temporal distribution shifts

#### 8.3.3 Financial Crime Detection

**Recommended**: Real-TabPFN with ExplainerPFN
- Regulatory compliance through interpretability
- Strong performance on financial benchmarks
- Established use in credit scoring [30]

### 8.4 Ensemble Approaches

Combining multiple tabular foundation models can yield superior performance:

**AutoTabPFN**: Automatically ensembles multiple TabPFN checkpoints, selecting optimal combinations for specific datasets [13].

**Heterogeneous Ensembles**: Combine TabPFN, TabICLv2, and TabDPT predictions through weighted averaging or stacking, leveraging complementary strengths.

**Hybrid Approaches**: Use foundation models for feature learning or probability calibration, combined with gradient boosting for final predictions.

### 8.5 Cost-Benefit Analysis

Table 3 summarizes the cost-benefit tradeoffs for different deployment scenarios:

| Scenario | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| **Research/Prototyping** | TabPFN-2.5 (free tier) | Best documentation, community support |
| **Small-scale Production** | TabICLv2 | Open-source, no licensing costs |
| **Large-scale Production** | TabDPT | Fastest inference, Apache 2.0 license |
| **Regulated Industries** | Real-TabPFN + ExplainerPFN | Interpretability, established benchmarks |
| **Custom Domain** | Fine-tune TabPFN-2.5 | Best fine-tuning support and documentation |

## 9. Limitations and Future Directions

### 9.1 Current Limitations

#### 9.1.1 Dataset Size Constraints

All three major models face limitations on maximum dataset size (50,000 samples) and feature count (2,000 features). While these limits accommodate many real-world applications, they exclude large-scale enterprise datasets common in fraud detection and credit scoring [13], [20].

**Mitigation Strategies**:
- Intelligent sampling for datasets exceeding limits [1]
- Hierarchical modeling with multiple foundation model instances
- Hybrid approaches combining foundation models with traditional methods for large-scale data

#### 9.1.2 Temporal Distribution Shifts

Fine-tuned models show reduced stability on datasets with gradual temporal shifts [9]. This limitation is critical for fraud detection, where adversarial adaptation is common.

**Future Directions**:
- Drift-resilient architectures with explicit temporal modeling [19]
- Continual learning approaches that adapt to distribution shifts without catastrophic forgetting
- Ensemble methods combining models trained on different time periods

#### 9.1.3 Interpretability Gaps

While ExplainerPFN addresses feature importance estimation [8], comprehensive interpretability remains challenging for transformer-based models. Regulatory requirements in finance and healthcare demand detailed explanations of individual predictions.

**Future Directions**:
- Attention-based explanation methods
- Counterfactual generation for tabular data
- Integration with established interpretability frameworks (LIME, SHAP)

#### 9.1.4 Categorical Feature Handling

Current models use relatively simple encoding strategies for categorical features (ordinal encoding, learned embeddings). High-cardinality categorical features and hierarchical categories remain challenging [1], [13].

**Future Directions**:
- Entity embeddings learned from large-scale data
- Graph-based encodings for hierarchical categories
- Transfer learning for categorical features across domains

### 9.2 Emerging Research Directions

#### 9.2.1 Multimodal Tabular Models

Integration of tabular data with other modalities (text, images, time series) represents a promising direction. Recent work demonstrates that combining on-chain data with OSINT signals improves fraud detection [2].

**Opportunities**:
- Joint pre-training on tabular and text data
- Cross-modal attention mechanisms
- Unified representations for heterogeneous data

#### 9.2.2 Causal Inference and Tabular Models

Tabular foundation models trained on synthetic data from causal models may internalize causal reasoning capabilities. Exploiting this for causal inference tasks could enable:
- Treatment effect estimation without randomized trials
- Counterfactual prediction for decision support
- Causal discovery from observational data

#### 9.2.3 Privacy-Preserving Tabular Models

Federated learning and differential privacy techniques could enable training on sensitive data without centralization:
- Federated pre-training across multiple organizations
- Differentially private fine-tuning
- Secure multi-party computation for inference

#### 9.2.4 Scaling Laws and Larger Models

TabDPT demonstrates scaling laws similar to language models [20], suggesting that larger tabular foundation models may yield continued improvements. Future work should explore:
- Models with billions of parameters
- Pre-training on Internet-scale tabular data
- Emergent capabilities at larger scales

### 9.3 Practical Deployment Challenges

#### 9.3.1 Model Monitoring and Maintenance

Production deployment requires robust monitoring for:
- Distribution drift detection
- Performance degradation alerts
- Adversarial attack detection
- Model versioning and rollback

#### 9.3.2 Integration with Existing Systems

Enterprise deployment faces challenges:
- Legacy system compatibility
- Real-time inference requirements
- Batch processing pipelines
- A/B testing infrastructure

#### 9.3.3 Regulatory Compliance

Financial and healthcare applications require:
- Model documentation and validation
- Bias and fairness audits
- Explainability for individual predictions
- Compliance with domain-specific regulations (GDPR, CCPA, etc.)

### 9.4 Open Research Questions

1. **Optimal Pre-training Data**: What is the ideal balance between synthetic and real-world data for pre-training? How should real-world datasets be selected and curated [1]?

2. **Transfer Learning Across Domains**: Can models pre-trained on general tabular data transfer effectively to specialized domains (medical, financial, scientific) [15]?

3. **Few-Shot Learning Limits**: What are the theoretical and practical limits of few-shot learning for tabular data? How many examples are required for reliable predictions [15]?

4. **Adversarial Robustness**: How robust are tabular foundation models to adversarial attacks? Can adversarial training improve robustness without sacrificing performance [2]?

5. **Uncertainty Quantification**: How can we obtain reliable uncertainty estimates from tabular foundation models for risk-sensitive applications [20]?

## 10. Conclusion

Tabular foundation models represent a paradigm shift in supervised learning for structured data, offering competitive or superior performance compared to traditional gradient boosting methods while eliminating hyperparameter tuning requirements. Our comprehensive review of 191 recent papers and technical documentation reveals three production-ready models—TabPFN-2.5, TabICLv2, and TabDPT—each with distinct strengths and optimal use cases.

**Key Findings**:

1. **Performance**: Real-TabPFN and TabDPT achieve 97.6% AUC on standard classification benchmarks, matching or exceeding XGBoost, LightGBM, and CatBoost. Specialized fraud detection implementations report up to 99.2% accuracy on Ethereum fraud detection tasks [1], [20], [29].

2. **Availability**: All three major models provide comprehensive GitHub repositories (5,800+, 602, and 77 stars respectively), Hugging Face model checkpoints, and PyPI packages, enabling immediate deployment for research and evaluation [13], [21], [23], [20].

3. **Feature Support**: Models support mixed numeric and categorical features, with maximum capacities of 50,000 samples and 2,000 features. Specialized checkpoints extend support to 1,000 features per estimator [1], [13], [20].

4. **Fine-tuning vs. Training from Scratch**: Full fine-tuning emerges as the most practical approach for adapting models to proprietary data, with documented implementations and clear performance gains. Training from scratch requires orders of magnitude more computation and is justified only for specialized domains [9], [20].

5. **Fraud Detection Efficiency**: Tabular foundation models demonstrate particular strength on imbalanced datasets, temporal distribution shifts, and multimodal feature combinations common in fraud detection. Interpretability extensions address regulatory requirements for explainable AI [2], [8], [29].

**Practical Recommendations**:

- **For general classification tasks with <10,000 samples**: Use TabPFN-2.5 with default checkpoint
- **For large-scale applications requiring open-source licensing**: Use TabICLv2
- **For regression tasks or datasets with 10,000-50,000 samples**: Use TabDPT
- **For fraud detection in regulated industries**: Use Real-TabPFN with ExplainerPFN
- **For custom domains with proprietary data**: Fine-tune TabPFN-2.5 using Yandex Research repository

**Future Outlook**:

The field is rapidly evolving, with emerging directions including multimodal integration, causal inference capabilities, privacy-preserving techniques, and scaling to larger models. Demonstrated scaling laws suggest that continued investment in model size and pre-training data will yield further improvements [20]. However, practical deployment challenges—including temporal distribution shifts, interpretability requirements, and regulatory compliance—require continued research and engineering effort.

Tabular foundation models have matured from research prototypes to production-ready systems, offering practitioners powerful tools for supervised learning tasks. The combination of strong empirical performance, comprehensive open-source implementations, and active research communities positions these models as viable alternatives to traditional methods for many applications. As the field continues to advance, we anticipate broader adoption across industries and continued improvements in performance, scalability, and interpretability.

## References

[1] A. Garg et al., "Real-TabPFN: Improving Tabular Foundation Models via Continued Pre-training With Real-World Data," arXiv preprint arXiv:2507.03971, 2025.

[2] M. Shoaei et al., "LROO Rug Pull Detector: A Leakage-Resistant Framework Based on On-Chain and OSINT Signals," 2026.

[3] V. Tran, "VALIDATION OF ACCURATE PREDICTIONS ON SMALL DATA WITH A TABULAR FOUNDATION MODEL FOR CLINICAL DECISION SUPPORT," 2026.

[4] Y. Ma et al., "In-Context Data Distillation with TabPFN," arXiv preprint arXiv:2402.06971, 2024.

[5] M. Kolberg et al., "TabPFN-Wide: Continued Pre-Training for Extreme Feature Counts," 2025.

[6] A. Papastergios et al., "Out in the wild: Investigating the impact of imperfect data on a tabular foundation model," 2025.

[7] Y. Liu et al., "Tabpfn unleashed: A scalable and effective solution to tabular classification problems," arXiv preprint arXiv:2502.02527, 2025.

[8] D. Fonseca et al., "ExplainerPFN: Towards tabular foundation models for model-free zero-shot feature importance estimations," 2026.

[9] A. Tanna et al., "Exploring Fine-Tuning for Tabular Foundation Models," 2025.

[10] Y. Liu et al., "Tokenize features, enhancing tables: the FT-TABPFN model for tabular classification," arXiv preprint arXiv:2406.06891, 2024.

[11] O. Thomas et al., "Retrieval & Fine-Tuning for In-Context Tabular Models," arXiv preprint arXiv:2406.05207, 2024.

[12] B. Feuer et al., "TuneTables: Context Optimization for Scalable Prior-Data Fitted Networks," arXiv preprint arXiv:2402.11137, 2024.

[13] L. Grinsztajn et al., "TabPFN-2.5: Advancing the State of the Art in Tabular Foundation Models," 2025.

[14] Y. Chen et al., "TabPFN Opens New Avenues for Small-Data Tabular Learning in Drug Discovery," ChemRxiv preprint, DOI: 10.26434/chemrxiv-2025-szk5s, 2025.

[15] N. Hollmann et al., "Accurate predictions on small data with a tabular foundation model," Nature, DOI: 10.1038/s41586-024-08328-6, 2025.

[16] Y. Xu et al., "Mixture of In-Context Prompters for Tabular PFNs," arXiv preprint arXiv:2405.16156, 2024.

[17] L. Regenwetter et al., "Engineering Regression Without Real-Data Training: Domain Adaptation for Tabular Foundation Models Using Multi-Dataset Embeddings," 2025.

[18] M. Hasan et al., "Tabular foundation model to detect empathy from visual cues," arXiv preprint arXiv:2504.10808, 2025.

[19] S. Helli et al., "Drift-Resilient TabPFN: In-Context Learning Temporal Distribution Shifts on Tabular Data," arXiv preprint arXiv:2411.10634, 2024.

[20] A. Krutikov et al., "Challenging Gradient Boosted Decision Trees with Tabular Transformers for Fraud Detection at Booking.com," arXiv preprint arXiv:2405.13692, 2024.

[21] J. Qu et al., "TabICLv2: A better, faster, scalable, and open tabular foundation model," 2026.

[22] R. Lal, "Evaluating SAP RPT-1 for Enterprise Business Process Prediction: In-Context Learning vs. Traditional Machine Learning on Structured SAP Data," 2026.

[23] J. Qu et al., "TabICLv2: A better, faster, scalable, and open tabular foundation model," 2026.

[24] M. García et al., "Foundation Models for Cybersecurity: A Comprehensive Multi-Modal Evaluation of TabPFN and TabICL for Tabular Intrusion Detection," Electronics, DOI: 10.3390/electronics14193792, 2025.

[25] Z. Ye et al., "A Closer Look at TabPFN v2: Strength, Limitation, and Extension," arXiv preprint arXiv:2502.17361, 2025.

[26] R. Sergazinov et al., "Chunked TabPFN: Exact Training-Free In-Context Learning for Long-Context Tabular Data," 2025.

[27] L. Hoo et al., "The Tabular Foundation Model TabPFN Outperforms Specialized Time Series Forecasting Models Based on Simple Features," arXiv preprint arXiv:2501.02945, 2025.

[28] Y. Ma et al., "TabPFGen -- Tabular Data Generation with TabPFN," arXiv preprint arXiv:2406.05216, 2024.

[29] T. Olusegun et al., "Improved Ethereum Fraud Detection Mechanism with Explainable Tabular Transformer Model," Proceedings of TPS-ISA, DOI: 10.1109/tps-isa62245.2024.00017, 2024.

[30] X. Li et al., "Class-Imbalanced-Aware Adaptive Dataset Distillation for Scalable Pretrained Model on Credit Scoring," arXiv preprint arXiv:2501.10677, 2025.
