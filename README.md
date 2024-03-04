# Masters-Project

## Towards an Effective Theory of Machine Learning

## Setup


1. **Cloning the repository:**

   ```bash
   git clone https://github.com/pabloiyu/Masters-Project.git

2. **Create a Conda environment:**

   ```bash
   conda create --name my-project-env python=3.9  
   conda activate my-project-env

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt 

## Usage | Running Files

In order to run any file, it is sufficient to simply type the following into terminal:

   ```bash
      python name_of_file.py
   ```

Within each file, there are a set of hyperparameters that can be modified to achieve the desired result. 

## Usage | Quantities at Initialization

* **compute_K_V_1input.py**

   * Computes and compares theoretical and computational values for the observables K and V in the case where we have one input vector. 
  
* **1overn_dependence_K_1input.py**

   * The computational value for K approximates both the leading order term and $O(1/n^{p})$ corrections. To differentiate these contributions, we calculate K for various values of n and plot K against 1/n. Fitting a straight line to this plot yields the leading order coefficient $(K_{0})$ as the intercept and the O(1/n) contribution $(K_{1})$ as the slope.

* **compute_K_2inputs.py**

   * Computes and compares theoretical and computational values for the observable K in the case where we have 2 input vectors.  

* **compute_NTK_initialisation.py**

   * Computes and compares experimental and theoretical values for the Neural Tangent Kernel (NTK) at initialisation.

* **1overn_dependence_NTK_initialisation.py**

   * The computational value for the NTK approximates both the leading order term and $O(1/n^{p})$ corrections. To differentiate these contributions, we calculate the NTK for various values of n and plot the NTK against 1/n. Fitting a straight line to this plot yields the leading order coefficient $(H_{0})$ as the intercept and the O(1/n) contribution $(H_{1})$ as the slope.

## Usage | Quantities at Training

NOTE: To run these files, it is recommended to have a GPU as they will run significantly slower on a CPU. If no GPU is available, it is recommended to modify the hyperparameters to make the computation more manageable. 

* **nconvergence_NTK.py**

   * Tries to computaionally validate Theorem 2.1 of arXiv: 1902.06720.
 
* **evolution_NTK.py**

   * In progress. Tries to validate Figure 10 of arXiv: 1909.11304.
