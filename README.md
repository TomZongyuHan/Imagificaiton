# Abstract
The innovative concept of "block imagification" transforms high-dimensional molecular measurements into two-dimensional RGB images. This process maintains a one-to-one relationship with the corresponding sample while enabling all possible associations across different classes. The transformation from non-image data to image data provides a holistic molecular representation of a sample, enhancing phenotype classification through advanced image recognition techniques in computer vision. A transformed RGB image encapsulates molecular positions in a 2D blocked diagonal neighbor-embedded space, with RGB channels representing molecular abundance and gene intensity. The proposed method was applied to single-cell RNA sequencing (scRNA-seq) data to ‘imagify’ gene expression profiles of individual cells. Results show that a simple convolutional neural network trained on these single-cell transcriptomics images accurately classifies diverse cell types, outperforming the best-performing scRNA-seq classifiers such as Support Vector Machine (SVM), K-nearest Neighbor (KNN), and Random Forest (RF).

# Model Architecture
<img width="1283" alt="image" src="https://github.com/TomZongyuHan/Imagificaiton/assets/73565616/355afc6f-6018-4641-9953-996eae555ef2">
The figure shows the Omics Imagification structure and CNN architecture for classification of scRNA-seq datasets.

# Installation and Running experiments
You need to install packages necessary for running experiments. Please run the following command.
```python
pip install -r requirement.txt
```

The following command provides an example of training in the proposed method. Please select dataset and methods. run_single.sh is to run single dataset in single method. run_loop can run multiple datasets.
```sh
# ./run.sh task_name
./run_single.sh ASR
```

# Experiment and Results
![fig3](https://github.com/TomZongyuHan/Imagificaiton/assets/73565616/cc18fd8b-9c3e-44a4-9732-d5813bee6b61)

| Method                 | Raw          | CPM          |
| ---------------------- | ------------ | ------------ |
| Trachea_FACS                                         |
| ---------------------- | ------------ | ------------ |
| RF                     |  0.900       |  0.900       |
| KNN                    |  0.835       |  0.918       |
| SVM                    |  0.876       |  0.912       |
| Transformer+CNN(ours)  |  0.954       |  0.952       |


# Citation
Please use the following citation for this work:
```

```

# Note

