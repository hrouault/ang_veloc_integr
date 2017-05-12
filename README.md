This repository contains the code associated to the paper:

[*Angular velocity integration in a fly heading circuit*](http://dx.doi.org/10.7554/eLife.23496)

Daniel B. Turner-Evans\*¹, Stephanie Wegener\*¹, Hervé Rouault¹,
Romain Franconville¹, Tanya Wolff¹, Johannes D. Seelig², Shaul Druckmann¹,
Vivek Jayaraman†¹

\* equal contributions

†Correspondence to: vivek@janelia.hhmi.org

¹ Janelia Research Campus
Howard Hughes Medical Institute
19700 Helix Drive,
Ashburn, VA 20147
USA
² Research Center CAESAR
Ludwig-Erhard-Allee 2
53175 Bonn
Germany


Angular velocity integration
----------------------------

The code is written in Python 3 and requires the following packages:

* SciPy
* NumPy
* Matplotlib 2

The code is divided into several scripts.
`veloc_integr` is the main script and gives the profiles, velocity curves,
etc. It links with the library `Ornstein_Uhlenbeck.py` which generates
artificial input velocity traces.
The script `plot_traces.py` plots these generated traces. Finally, the scripts
`analysis_linear.py` and `plot_lin.py` refers to the analysis of the angular
integration linearity.

