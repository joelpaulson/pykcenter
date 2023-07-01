# pykcenter
This is a Python implementation of the global k-center optimization algorithm from the following paper:

- Title: Global Optimization of K-Center Clustering
- Authors: Mingfei Shi, Kaixun Hua, Jiayang Ren, Yankai Cao
- Publisher: Proceedings of the 39th International Conference on Machine Learning, PMLR 162:19956-19966
- Year: 2022
- Link: https://proceedings.mlr.press/v162/shi22b.html

A Julia version of the code, which was the basis for this Python code, was released by the authors of the original paper [here](https://github.com/YankaiGroup/global_kcenter).

# Installation
The main purpose of this code is to provide a Python-compatiable and easy-to-install implementation of the algorithm above that allows you to globally solve kcenter clustering problems efficiently. You can install the package by running the setup.py file or by calling the following pip command linked to this Github page:

```
pip install git+https://github.com/joelpaulson/pykcenter
```

# Examples
There are two examples provided under the example folder that you can modify to suite the needs of your specific problems. 

# Todo
The code is currently not as efficient as the Julia implementation due to some inherent limitations of Python. I have attempted to overcome most of those limitations by using vectorized torch operations when possible, however, there are a couple of remaining steps that could be improved. I hope to make these more efficient in the future but please feel free to contribute by creating a pull request if you can improve upon the slow components. I have a simple test that profiles the cost of each function, which you can use to identify the key bottlenecks as the data size increases.
