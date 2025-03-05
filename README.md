# libThinPlateShells

This code repo implements the techniques presented in the Eurographics 2025 conference paper: [Corotational Hinge-based Thin Plates/Shells](https://arxiv.org/pdf/2502.10872). 

## Bending Models for the Thin Shell

### Corotational hinge-based thin plates/shells
- We present a **corotational edge-based hinge curvature operator** for thin shell simulation, including a specific variant for thin plate simulation. (EP/ES --- edge-based hinge thin plate/shell model)
- We propose a **corotational FVM (finite volume method) hinge curvature operator** for thin shell simulation, along with a specific variant for thin plate simulation. (FS/FP --- FVM hinge thin plate/shell model)
- We introduce a **corotational smoothed hinge curvature operator** for thin shell simulation, as well as a specific variant for thin plate simulation. (SS/SP --- smoothed hinge thin plate/shell model)
- All **six thin plate/shell models** feature **constant bending energy Hessians**, with detailed boundary conditions for accurate simulation. 

### Quadratic and Cubic Shells
We aim to improve the accuracy of the [Quadratic Shell](https://www.cs.columbia.edu/cg/quadratic/quadratic.cpp) (QS), and the [Cubic Shell](https://www.cs.columbia.edu/cg/pdfs/140-cubicShells-a4.pdf) model (CS) by identifying the rational bending rigidity for edge-based hinge bending models.

The corresponding **Quadratic Thin Plate/Shell** (QTP/QTS) formulations with the rational bending rigidity are provide in the Appendix of our paper. QTP/QTS also can be seen as a specific variant of EP/ES.

### Thin shells using midedge operators
The midedge operators are adapted from the [libshell](https://github.com/evouga/libshell) library, which includes three types of formulations: MidedgeAve, MidedgeTan, MidedgeSin, for comparison. (MidEdgeAve/Tan/Sin)

--------------------------------------
13 different bending formulations are included in this repo. These formulations are labeled as: **EP, ES, FP, FS, SP, SS**, QS, CS, **QTP, QTS**, MidEdgeAve, MidEdgeSin, MidEdgeTan.

## Benchmark Cases

We provide the tested benchmark data in the **data** folder. For the geometrically non-linear shell benchmarks, the **ABAQUS .inp** files are also provided.

### Linear plate bending benchmark.
- A square plate with simply supported boundaries, subjected to a uniform load perpendicular to its plane.

### Geometrically non-linear shell benchmarks.
- Cantilever plate subjected to end shear force.
- Hemispherical shell subjected to alternating radial forces.

## Data Structure and Parallelization
The data structure of this repository is inspired by [libshell](https://github.com/evouga/libshell) and [WrinkledTensionFields](https://github.com/zhenchen-jay/WrinkledTensionFields). The code has been parallelized using the [Threading Building Blocks (TBB)](https://github.com/wjakob/tbb) library to enhance performance.


## Dependencies
- [Libigl](https://github.com/libigl/libigl.git)
- [Polyscope](https://github.com/nmwsharp/polyscope.git)
- [TBB](https://github.com/wjakob/tbb.git)
- [Json](https://github.com/nlohmann/json.git) 
- [Suite Sparse](https://people.engr.tamu.edu/davis/suitesparse.html)


## Build
The code has been tested only on Linux.
### Install SuiteSparse
```
sudo apt-get update -y
sudo apt-get install -y libsuitesparse-dev
```
### Clone and Build the Project
Use the following commands to download and compile the project:
```
    git clone https://github.com/liangqx-hku/libThinPlateShells.git
    cd LibThinShells
    mkdir build
    cd build
    cmake ..
    make
```

## Run the Project
You can execute the program with the following command:
```
sh runCmd.sh
```
**Customize Input and Output**

In the runCmd.sh, you can specify the paths for the input JSON file and the output mesh file:
```
./bin/ThinShellCli_bin -i ../data/elastic/slit_plate/slit_plate.json -o ../data/elastic/slit_plate/output/output_slit_plate51.obj

```

## Configuration Parameters
The following parameters in the JSON file should be specified, others are default for thin shell tests in this project.
- **`bending_type`**: Specifies the bending model. Options include **EP, ES, FP, FS, SP, SS** for our proposed models. QS/CS specify Quadratic/Cubic Shlls.
  - For **midedge formulations**, use **`midEdgeShell`** with `sff_type` to specify the discrete second fundamental form:  
    - `midedgeAve` (default)  
    - `midedgeTan`  
    - `midedgeSin`  
  - When using **EP, ES, FP, FS, SP, SS**, `sff_type` must be set to `midedgeAve` for data structure consistency.  
  - *(Note: QTP/QTS are implemented within EP/ES but are not provided for direct execution.)*  

- **`clamped_DOFs`**: Path to the `.dat` file containing clamped degrees of freedom, formatted as `(DOF index, coordinate at the clamped dof)`.  

- **`curedge_DOFs`**: Specifies **midedge rotation degrees of freedom** for midedge formulations.  

- **`rest_mesh`**: Path to the `.obj` file for the initial configuration.  

- **`output_mesh`**: Path to the `.obj` file for the final configuration.  

- **`point_Forces`**: Path to the `.dat` file storing all point forces, formatted as `(DOF index, force value)`.  

- **`gravity`**: Gravity force magnitude in the **X, Y, and Z** directions.  

- **`density`**: Material density.  

- **`poisson_ratio`**: Poisson’s ratio.  

- **`thickness`**: Shell thickness.  

- **`youngs_modulus`**: Young’s modulus.  

- **`stretching_type`**: Specifies the material model (**StVK** is used in this project).  

- **`rest_flat`**: A flag indicating whether the initial mesh is flat (zero second fundamental form) for midedge formulations.  


## GUI with Polyscope
We also provide a GUI version of the executable program: ThinShellGui_bin. The input parameters for the GUI version are identical to those used in the command-line counterpart.

## Contact
If you have any question, please feel free to contact me at liangqixin23@outlook.com

