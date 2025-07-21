# 🦎 Triturus 🦎

The following table describes the implemented kernels.

| Kernel ID    | Description                               | Operation                            | Source module                   |
| ------------ | ----------------------------------------- | ------------------------------------ | ------------------------------- |
| vadd         | Vector addition                           | $a_i+b_i$                            | [add](triturus/add.py)          |
| vamax        | Vector maximum                            | $\max_i a_i$                         | [max](triturus/max.py)          |
| vmax         | Vector maximum with indices               | $(\max_i a_i, \arg\max_i a_i)$       | [max](triturus/max.py)          |
| matmax       | Matrix maximum along one axis             | $\max_i a_{ij}$ or $\max_j a_{ij}$   | [max](triturus/max.py)          |
| mm           | Matrix multiplication                     | $\sum_j a_{ij}b_{jk}$                | [mm](triturus/mm.py)            |
| lm2exp       | Batch log-matmul, one matrix in log-space | $\log(\sum_j a_{rij} \exp b_{rjk})$  | [lm2exp](triturus/lm2exp.py)    |
