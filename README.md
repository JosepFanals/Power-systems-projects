# ASD
 The Alternating Search Directions method (ASD) is an iterative technique to solve the power flow problem of traditional electric power systems. The main reference is [this paper](https://www.sciencedirect.com/science/article/abs/pii/S0378779616302292).

-----------------
### Installation

There are a couple of options to install the software:

1. Clone the [ASD repository from GitHub][1]:
   
*Use this option if you are familiar with Git*
      
   - From the command line:
        - `git clone https://github.com/JosepFanals/ASD`
   - Or from the [ASD GitHub repository page][1]:
        - Click the green **Clone or download** button, then **Open in Desktop**.

2. Download the repository as a .zip file from the GitHub page:
    - Go to the [ASD GitHub repository page][1].
    - Click the green **Clone or download** button, then **Download ZIP**.
    
---------------
### Running ASD

In order to employ the ASD method, you only need to run the ```Code/ASD.py``` file. You can change and modify the grid you are simulating as long as you mantain the same data format as in ```Code/data_IEEE14.py``` and ```Code/data_IEEE30.py```, which serve as examples.

Remember to tune the parameters as explained in the [paper](https://github.com/JosepFanals/ASD/blob/master/Escrit/bare_jrnl.pdf) accordingly.

---------------
### Learning about ASD

I suggest reading the following papers to gain knowledge and ituition about ASD:
* **[D. Borzacchiello, M. H. Malik, F. Chinesta, R. García-Blanco and Pedro Díez, "Unified formulation of a family of iterative solvers for power systems analysis", in Electric Power Systems Research, vol. 140, pp. 201–208, 2016](https://www.sciencedirect.com/science/article/abs/pii/S0378779616302292)**: the main reference, enough to understand properly the ASD method.
* **[My paper](https://github.com/JosepFanals/ASD/blob/master/Escrit/bare_jrnl.pdf)**: details the modification of search directions and the modeling of PV buses. This causes the algorithm to be faster.

------------
### License

This works is distributed under the [MIT License](https://opensource.org/licenses/MIT).

---------------------
### Acknowledgements

All this was possible thanks to the help I received from [Santiago Peñate Vera](https://github.com/SanPen).

[1]: https://github.com/JosepFanals/ASD
