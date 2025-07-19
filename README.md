# Triturus

| Kernel ID    | Description                                           | Operation                          | Source module                      |
| ------------ | ----------------------------------------------------- | ---------------------------------- | ---------------------------------- |
| vadd         | Vector addition                                       | $a_i+b_i$                          | [add](triturus/add.py)             |
| vamax        | Vector maximum                                        | $\max_i a_i$                       | [max](triturus/max.py)             |
| vmax         | Vector maximum with indices                           | $(\max_i a_i, \arg\max_i a_i)$     | [max](triturus/max.py)             |
| matmax       | Matrix maximum along one axis                         | $\max_i a_{ij}$ or $\max_j a_{ij}$ | [max](triturus/max.py)             |
| mm           | Matrix multiplication                                 | $\sum_j a_{ij}b_{jk}$              | [mm](triturus/mm.py)               |
| logmm2exp    | Log-matrix multiplication, second matrix in log-space | $\log(\sum_j a_{ij} \exp b_{jk})$  | [logmm2exp](triturus/logmm2exp.py) |
