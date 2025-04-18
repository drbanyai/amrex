Single-Level Gauss-Seidel Pressure Solver
========================================

Testbed implementing a single-level Gauss-Seidel pressure solver on a MAC grid.

Physics
-------

Solves the pressure Poisson equation:

```math
\nabla^2 p = \nabla \cdot \vec{u}
```

where:
- p is the pressure
- u is the velocity field

The Gauss-Seidel method is used to solve this equation iteratively. For a cell (i,j,k), the pressure update is:

```math
p_{i,j,k}^{n+1} = \frac{1}{6} \left( p_{i+1,j,k}^n + p_{i-1,j,k}^n + p_{i,j+1,k}^n + p_{i,j-1,k}^n + p_{i,j,k+1}^n + p_{i,j,k-1}^n - h^2 \nabla \cdot \vec{u}_{i,j,k} \right)
```

where:
- n is the iteration number
- h is the grid spacing

Geometry and Mesh
----------------

The test case uses a simple 3D domain with:
- Cell-centered pressure field
- Face-centered velocity components (MAC grid)
- Periodic boundary conditions

Initialization and Boundary Conditions
------------------------------------

The initial test case:
1. Sets up a simple velocity field with known divergence
2. Initializes pressure to zero
3. Uses periodic boundary conditions for both velocity and pressure

Implementation Details
---------------------

The solver:
1. Computes the divergence of the velocity field
2. Performs Gauss-Seidel iterations to solve for pressure
3. Checks for convergence using the L2 norm of the residual
4. Optionally applies under-relaxation for stability

The code is structured to:
- Make it easy to understand the basic algorithm
- Provide a foundation for extending to multi-level
- Allow for easy modification of parameters and boundary conditions 