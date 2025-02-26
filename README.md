# QuantGplearn
A systematic framework for factor mining in quantitative investment strategies

## Improvements Made in This Project

This project introduces the following improvements:

1. Added new temporal operators as listed below, with extensive vectorization and Numba optimization for better performance. A complete list of these operators can be found here: [https://zhuanlan.zhihu.com/p/24627909174](https://zhuanlan.zhihu.com/p/24627909174).
2. The original `gplearn` package allowed setting constant parameters only within a single range. In this project, we enhanced this functionality by enabling the selection of constants from a set. This change is particularly useful for setting parameters based on economic logic, such as selecting quantization strategies for 1-day, 7-day intervals, etc.
3. Added a penalty term to the fitness function that limits the length of formulas, preventing formulas from growing indefinitely during iterations.
4. Introduced a maximum length restriction: if the length exceeds a certain threshold, no crossover, subtree mutation, point mutation, or point replacement will occur. Only pruning mutations will be applied.
5. Optimized the slow computation of all factors by utilizing `pathos.multiprocessing` for parallel processing.

## Example Code

Example code for this project can be found in `example/get_factors.py`. Running this script will generate the following example factors:

[ts_min(ts_atr(zscore_7d, high, mom_3d, 24, 24), 72),
 ts_macd(ts_hedge(ts_bar_bs(ts_zscore(ts_cdlbodym(zscore_7d, zscore_1d, 24), 72), ts_cmo(abs(mom_1d), 72), 24), ts_kurt(mul(ts_mom(ma_7d, 72), ts_delta(ma_1d, 72)), 72), 72, 24), 24, 24, 72)]


## Acknowledgements
I would like to express my gratitude to the developers of the following projects, whose work has significantly contributed to the development of this package:

- [gplearn](https://github.com/trevorstephens/gplearn): This repository provided foundational ideas and code that inspired the design and implementation of this package.
- [gplearnplus](https://github.com/ACEACEjasonhuang/gplearnplus): I used this repository as a reference for extending genetic programming functionalities in my project.

Thank you to both projects for their valuable contributions!