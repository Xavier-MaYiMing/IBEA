### IBEA: Indicator-based evolutionary algorithm

##### Reference: Zitzler E, Künzli S. Indicator-based selection in multiobjective search[C]//International Conference on Parallel Problem Solving from Nature. Berlin, Heidelberg: Springer Berlin Heidelberg, 2004: 832-842.

##### IBEA is a classic multi-objective evolutionary algorithm (MOEA).

| Variables | Meaning                                              |
| --------- | ---------------------------------------------------- |
| npop      | Population size                                      |
| iter      | Iteration number                                     |
| lb        | Lower bound                                          |
| ub        | Upper bound                                          |
| nobj      | The dimension of objective space (default = 3)       |
| eta_c     | Spread factor distribution index (default = 20)      |
| eta_m     | Perturbance factor distribution index (default = 20) |
| nvar      | The dimension of decision space                      |
| pop       | Population                                           |
| objs      | Objectives                                           |
| off       | Offspring                                            |
| off_objs  | The objectives of offspring                          |
| pf        | Pareto front                                         |

#### Test problem: DTLZ2

$$
\begin{aligned}
	& k = nvar - nobj + 1, \text{ the last $k$ variables is represented as $x_M$} \\
	& g(x_M) = \sum_{x_i \in x_M}(x_i - 0.5)^2 \\
	& \min \\
	& f_1(x) = (1 + g(x_M)) \cos{\frac{x_1 \pi}{2}} \cdots \cos{\frac{x_{M - 2} \pi}{2}} \cos{\frac{x_{M - 1} \pi}{2}} \\
	& f_2(x) = (1 + g(x_M)) \cos{\frac{x_1 \pi}{2}} \cdots \cos{\frac{x_{M - 2} \pi}{2}} \sin{\frac{x_{M - 1} \pi}{2}} \\
	& f_3(x) = (1 + g(x_M)) \cos{\frac{x_1 \pi}{2}} \cdots \sin{\frac{x_{M - 2} \pi}{2}} \\
	& \vdots \\
	& f_M(x) = (1 + g(x_M)) \sin(\frac{x_1 \pi}{2}) \\
	& \text{subject to} \\
	& x_i \in [0, 1], \quad \forall i = 1, \cdots, n
\end{aligned}
$$



#### Example

```python
if __name__ == '__main__':
    main(100, 150, np.array([0] * 7), np.array([1] * 7))
```

##### Output:

![Pareto front](/Users/xavier/Desktop/Xavier Ma/个人算法主页/IBEA/Pareto front.png)



