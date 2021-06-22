# Flow decomposition
This repository hosts a few simulations related to the flow decomposition of nonequilibrium stochastic processes proposed by Ping Ao ([SDE decomposition and A-type stochastic interpretation in nonequilibrium processes](https://link.springer.com/article/10.1007/s11467-017-0718-2)) for a series of different systems. Initially, we just took a couple of worked out examples from:
- [Tang, Ying, Song Xu, and Ping Ao. "Escape rate for nonequilibrium processes dominated by strong non-detailed balance force." The Journal of chemical physics 148.6 (2018): 064102.](https://aip.scitation.org/doi/full/10.1063/1.5008524)
- [Chaudhari, Pratik, and Stefano Soatto. "Stochastic gradient descent performs variational inference, converges to limit cycles for deep networks." International Conference on Learning Representations. 2018.](https://openreview.net/forum?id=HyWrIgW0W)
where the decomposition is defined by construction to be relatively straighforward.

Following that, we tried to find an actual decomposition for another system, taken from:
- [Beer, R. D. (1995). On the Dynamics of Small Continuous-Time Recurrent Neural Networks. Adaptive Behavior, 3(4), 469–509.](https://journals.sagepub.com/doi/10.1177/105971239500300405)
where chaotic dynamics generate a Lorenz-like attractor in a continuous-time recurrent neural network (CTRNN), see Figure 9. However the decomposition here is all but given for now, and a further investigation is thus needed.

## Applications
The goal here is to try and use Ao's decomposition in the analysis of landscape (energy) models of nonequilibrium systems adopted, for example, as models of decision making and working memory:
- [Yan, Han, et al. "Nonequilibrium landscape theory of neural networks." Proceedings of the National Academy of Sciences 110.45 (2013)](https://www.pnas.org/content/110/45/E4185)
- [Yan, Han, and Jin Wang. "Non-equilibrium landscape and flux reveal the stability-flexibility-energy tradeoff in working memory." PLoS Computational Biology 16.10 (2020)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008209)

inspired by the celebrated:
- [Wong KF, Wang XJ. A recurrent network mechanism of time integration in perceptual decisions. Journal of Neuroscience. 2006;26(4)](https://www.jneurosci.org/content/26/4/1314)

The main intuition is related to the presence of a nonequilibrium flux that is related to the hysteresis phenomenon used to represent memory or decisions by sustained activity in the models of neural circuits used in the last paper.
