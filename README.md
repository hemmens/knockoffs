# Pseudo-Perfect Knockoff Tests

#### Description

This is a collection of Python scripts designed to generate pseudo-perfect knockoffs using linear algebra. There are three main tests corresponding to different sections of our paper:

- A simple test for matching complex co-moments in _linalg3.py_,

- a test using real-world data in the _Swiss Stock Data_ folder,

- and a comparison with the Metropolized Sampling algorithm presented by Bates, Candès, Janson, and Wang (2020) in _MH_Compare_.

MH_Compare/t_core.py is available to ensure t_hemmens.py runs correctly. The original file can be found [here](https://github.com/wenshuow/metro/blob/f2c0b5c2eaf64d8759ab651d5aff4a787bcd9ae3/heavy-tailed-t/t_core.py).

All examples use the module _knockoff_lib2.py_.

Correspondence should be addressed to [Christopher Hemmens](mailto:chris@christopherhemmens.com).

#### Reference

Christopher Hemmens and Stephan Robert-Nicoud (2024). *Can linear algebra create perfect knockoffs?* Proceedings of the 7th International Conference on Big Data and Internet of Things

Stephen Bates, Emmanuel Candès, Lucas Janson and Wenshuo Wang (2020). *Metropolized Knockoff Sampling*. Journal of the American Statistical Association [[pdf](http://lucasjanson.fas.harvard.edu/papers/Metropolized_Knockoff_Sampling-Bates_ea-2019.pdf)] [[arXiv](https://arxiv.org/abs/1903.00434)] [[journal](https://www.tandfonline.com/doi/full/10.1080/01621459.2020.1729163)]
