


<div align="center">
  <a href="(https://github.com/panliangrui/STAS/blob/main/STAS%20prediction.png)">
    <img src="https://github.com/panliangrui/STAS/blob/main/STAS%20prediction.png" width="600" height="200" />
  </a>

  <h1>STAS(微乳、实性细胞巢巢、单细胞)</h1>

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

# 用于预测肺癌组织病理学图像中 STAS 的特征交互式孪生图编码器

## 目录

<details>
  <summary>Click me to Open/Close the directory listing</summary>

- [目录](#目录)
- [特征预处理](#特征预处理)
  - [特征提取](#特征提取)
  - [构图](#构图)
- [训练模型](#训练模型)
- [测试模型](#测试模型)
- [数据集](#数据集)
- [网站](#网站)
- [版权](#版权)

</details>

## 特征预处理

使用预训练模型进行特征预处理并构建WSI的空间拓扑图。

### 特征提取

基于KimiaNet和CTransPath提取特征。
KimiaNet请参考：https://github.com/KimiaLabMayo/KimiaNet
CTransPath请参考：https://github.com/Xiyue-Wang/TransPath
```markdown
python new_cut7.py
```
```markdown
python new_cut7-1.py
```

### 构图

使用KNN（K=9）构建空间拓扑图。
```markdown
python construct_graph1.py
```

## 训练模型
```markdown
train_feature1.py
```
## 测试模型

- 五重交叉验证的最佳模型
  ```markdown
  链接：https://pan.baidu.com/s/11dxmND9ZhEA-o-Hnql6_rg?pwd=l6gh 提取码：l6gh
  ```
- 最终测试的最佳模型
  ```markdown
  链接：https://pan.baidu.com/s/1lT8x_ovemj3FXvfjTRjxmA?pwd=516i 提取码：516i 
  ```
- 测试模型以获得预测。
  ```markdown
  python test_stas.py
  ```

## 数据集

- 由于数据具有隐私保护协议，仅提供组织病理学图像数据的特征。
```markdown
  链接：https://pan.baidu.com/s/1pJY1Cv9d-ML7jU09RnGOjg?pwd=rzj7 提取码：rzj7 


```
- 我们提供 STAS 患者的临床数据，包括患者年龄、性别、分期和蛋白水平表达数据。
请发邮件联系通讯作者或者第一作者。
## 网站

欢迎您访问我们的STAS测试平台：http://plr.20210706.xyz:5000/。
```markdown
使用前先清除所有
1.上传.svs文件的病理学图像。
2.点击提交，获得预测结果.
```

## 版权
代码将在论文接收后更新！！

[License MIT](../LICENSE)
