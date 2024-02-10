
  [![Build](https://github.com/SartajBhuvaji/SDGnE/actions/workflows/main.yaml/badge.svg)](https://github.com/SartajBhuvaji/SDGnE/actions/workflows/main.yaml)


## About
- SDGnE (Synthetic Data Generation and Evaluation) is a Python package designed to generate synthetic data and evaluate its quality using neural network models. 
- This tool is intended for developers and researchers who require synthetic datasets for testing and development.
- Current dittto uses <i>Autoencoders</i> and <i>SMOTE</i> to generate synthetic data.

## Getting Started
`pip install sdgne`
 
 ## Notebooks
 To get started, we have created notebook for the Autoencoder and SMOTE algorithm.

  ### Auto Encoder
  Autoencoders are a class of neural networks designed for unsupervised learning and representing features in a smaller space. They consist of an encoder and a decoder, intending to learn the input data's compressed representation (encoding).  We leverage this architecture to generate synthetic data.

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SartajBhuvaji/SDGnE/blob/main/notebooks/SDGnE_Autoencoder_Notebook.ipynb)

  ### SMOTE
  SMOTE, abbreviated as Synthetic Minority Oversampling Technique, is used to generate synthetic data from the original dataset. Over the years, several variants of SMOTE have been developed, each tailored to specific scenarios and requirements. These variants employ distinct methodologies and innovations to enhance the generation of synthetic data, thereby improving model performance by ensuring a more balanced distribution of classes. We provide a few SMOTE variants for synthetic data generation.

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SartajBhuvaji/SDGnE/blob/main/notebooks/SDGnE_SMOTE_Notebook.ipynb)

  ### Comparison
  In this notebook, we will compare the `Single Encoder Autoencoder` and the `SMOTE Algorithm` for synthetic data generation. We will generate synthetic data using both the algorithms and perform statistical evaluation.
  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SartajBhuvaji/SDGnE/blob/main/notebooks/SDGnE_Comparison_Notebook.ipynb)

 ## Features

- **Data Generation**: Create synthetic datasets that mimic the statistical properties of real-world data.
- **Neural Autoencoders**: Utilize various autoencoder architectures to learn data representations.
- **Evaluation Metrics**: Assess the quality of synthetic data using built-in evaluation metrics.
- **Extensibility**: Easily extend the package with custom data generators and evaluators.

 ## Links
 - **Documentation**: https://seattle-university.gitbook.io/sdgne/