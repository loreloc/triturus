# 🦎 Triturus 🦎

The following table describes the implemented kernels.

| Kernel ID    | Description                               | Operation                            | Source                       |
| ------------ | ----------------------------------------- | ------------------------------------ | ---------------------------- |
| vadd         | Vector addition                           | $a_i+b_i$                            | [add](triturus/add.py)       |
| vamax        | Vector maximum                            | $\max_i a_i$                         | [max](triturus/max.py)       |
| vmax         | Vector maximum with indices               | $(\max_i a_i, \arg\max_i a_i)$       | [max](triturus/max.py)       |
| matmax       | Matrix maximum along one axis             | $\max_i a_{ij}$ or $\max_j a_{ij}$   | [max](triturus/max.py)       |
| mm           | Matrix multiplication                     | $\sum_j a_{ij}b_{jk}$                | [mm](triturus/mm.py)         |
| lm2exp       | Batch log-matmul, one matrix in log-space | $\log(\sum_j a_{rij} \exp b_{rjk})$  | [lm2exp](triturus/lm2exp.py) |

## Benchmarks Gallery

| Kernel ID    | Benchmark Description                                   | Baselines   | Results                      |
| ------------ | ------------------------------------------------------- | ----------- | ---------------------------- |
| vmax         | Vector maximum with and without indices                 | torch       | [here](#benchmark-of-vmax)   |
| matmax       | Matrix maximum along rows and columns                   | torch       | [here](#benchmark-of-matmax) |
| mm           | Matrix multiplication with square matrices              | torch       | [here](#benchmark-of-mm)     |
| lm2exp       | Batch log-matmul, square and rectangular batch matrices | torch + jit | [here](#benchmark-of-lm2exp) |

### Benchmark of vmax

![vmax](https://github.com/loreloc/triturus/releases/download/v0.1/benchmark-vmax.png)

### Benchmark of matmax

![matmax](https://github.com/loreloc/triturus/releases/download/v0.1/benchmark-matmax.png)

### Benchmark of mm

![mm](https://github.com/loreloc/triturus/releases/download/v0.1/benchmark-mm.png)

### Benchmark of lm2exp

![lm2exp](https://github.com/loreloc/triturus/releases/download/v0.1/benchmark-lm2exp.png)

