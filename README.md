# ParticleModel
A hybrid machine learning framework for modelling and control of particle processes using on-line/at-line particle analysis and other at-line/on-line process sensors. ParticleModel is implemented in Python, using the [TensorFlow framework](https://www.tensorflow.org/).

![Overview](/documentation/overview.svg)

## Modules overview
The framework consists of several modules outlined by the following:
- Data module: Data structure for storing time-series data
- Design of experiments module: Module for generating design of experiments
- Domain module: Domain module for discretization of particle distributions
- Reference model module: First principles reference models for testing of framework
- Hybrid model module: Hybrid modelling framework
- Process control module: Process control structures for particle processes

## Installation notes
Python > 3.7 is required for this code to work. It is recommended so set up an individual python enviroment for this installation.

The necessary python packages and versions can be found in [_requirements.txt_](requirements.txt).
To install all packages in one go, use the following pip-command:

```
pip install -r requirements.txt
```

## Litterature references
- R. F. Nielsen, N. A. Kermani, L. la Cour Freiesleben, K. V. Gernaey, S. S. Mansouri, <em>Novel strategies for predictive particle monitoring and control using advanced image analysis</em>, in: A. A. Kiss, E. Zondervan, R. Lakerveld, L. zkan (Eds.), 29th European Symposium on Computer Aided Process Engineering, volume 46 of Comput. Aided Chem. Eng., Elsevier, 2019, pp. 1435-1440. doi:10.1016/B978-0-12-818634-3.50240-X.
- R. F. Nielsen, N. Nazemzadeh, L. W. Sillesen, M. P. Andersson, K. V.
Gernaey, S. S. Mansouri, <em>Hybrid machine learning assisted modelling framework for particle processes</em>, Comput. Chem. Eng., Elsevier, 2020 (accepted)
