

<div align="center">
  <a href="(https://github.com/panliangrui/STAS/blob/main/SHIYITU.jpg)">
    <img src="https://github.com/panliangrui/STAS/blob/main/SHIYITU.jpg" width="700" height="400" />
  </a>

  <h1>STAS(micropapillary, solid nests, single cells)</h1>

  <p>
  Liangrui Pan et al. is a developer helper.
  </p>

  <p>
    <a href="https://github.com/misitebao/yakia/blob/main/LICENSE">
      <img alt="GitHub" src="https://img.shields.io/github/license/misitebao/yakia"/>
    </a>
  </p>

  <!-- <p>
    <a href="#">Installation</a> | 
    <a href="#">Documentation</a> | 
    <a href="#">Twitter</a> | 
    <a href="https://discord.gg/zRC5BfDhEu">Discord</a>
  </p> -->

  <div>
  <strong>
  <samp>

[English](README.md) · [简体中文](README.zh-Hans.md)

  </samp>
  </strong>
  </div>
</div>

# Feature-interactive Siamese graph encoder for predicting STAS in lung cancer histopathology images

## Table of Contents

<details>
  <summary>Click me to Open/Close the directory listing</summary>

- [Table of Contents](#table-of-contents)
- [Feature Preprocessing](#Feature-Preprocessing)
  - [Feature Extraction](#Feature-Extraction)
  - [Graph Construction](#Graph-Construction)
- [Train Models](#Train-models)
- [Test Models](#Test-Models)
- [Datastes](#Datastes)
- [Website](#Website)
- [License](#license)

</details>

## Feature Preprocessing

Use the pre-trained model for feature preprocessing and build the spatial topology of WSI.

### Feature Extraction

Features extracted based on KimiaNet and CTransPath.
Please refer to KimiaNet: https://github.com/KimiaLabMayo/KimiaNet.
Please refer to CTransPath: https://github.com/Xiyue-Wang/TransPath
```markdown
python new_cut7.py
```
```markdown
python new_cut7-1.py
```

### Graph Construction

Use KNN (K=9) to construct the spatial topology map.
```markdown
python construct_graph1.py
```

## Train Models
```markdown
train_feature1.py
```
## Test Models

- Best model for five-fold cross validation
  ```markdown
  link：https://pan.baidu.com/s/11dxmND9ZhEA-o-Hnql6_rg?pwd=l6gh 
  ```
- Best model finally tested
  ```markdown
  link：https://pan.baidu.com/s/1lT8x_ovemj3FXvfjTRjxmA?pwd=516i  
  ```
- Test the model to obtain predictions.
  ```markdown
  python test_stas.py
  ```

## Datastes

- Only features of the histopathology image data are provided as the data has a privacy protection agreement.
```markdown
  link：https://pan.baidu.com/s/1pJY1Cv9d-ML7jU09RnGOjg?pwd=rzj7
  https://zenodo.org/records/11611418

```
- We provide clinical data on STAS patients, including patient age, gender, stage and protein level expression data.
Please contact the corresponding author or first author by email.
## Website

We welcome you to visit our STAS test platform at http://plr.20210706.xyz:5000/.

```markdown
Clear all before use
1.Upload the pathology image of .svs file.
2.Click Submit to get the prediction results.
```
## License
The code will be updated after the paper is accepted!!
[License MIT](../LICENSE)
