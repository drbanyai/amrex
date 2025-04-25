/*
 *  Testbed implementing a single-level Gauss-Seidel pressure solver on a MAC grid.
 */

/*--------------------------------------------------------------------
  standard includes
  --------------------------------------------------------------------*/
#include <AMReX.H>

/*--------------------------------------------------------------------
  non-standard includes
  --------------------------------------------------------------------*/
#include "helpers.H"

/*--------------------------------------------------------------------
  forward declarations
  --------------------------------------------------------------------*/
int MyMain();

/*--------------------------------------------------------------------
  function definitions
  --------------------------------------------------------------------*/
int main(int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    const int ret = MyMain();
    amrex::Finalize();
    return ret;
}

int MyMain()
{
    constexpr int nlevels = 3;
    amrex::Vector<amrex::Geometry> geom(nlevels);
    amrex::Vector<amrex::BoxArray> ba(nlevels);
    amrex::Vector<amrex::DistributionMapping> dm(nlevels);
    amrex::Vector<amrex::MultiFab> pressure(nlevels);
    amrex::Vector<std::array<amrex::MultiFab, 3>> velocity(nlevels);

    amrex::Vector<int> nx(nlevels), ny(nlevels), nz(nlevels);
    amrex::Vector<double> dx(nlevels);

    // Set up each level: base grid 4x4x4, 2x refinement per level
    for (int lev = 0; lev < nlevels; ++lev) {
        nx[lev] = ny[lev] = nz[lev] = 4 * (1 << lev); // 4, 8, 16
        dx[lev] = 1.0 / nx[lev];
        geom[lev] = DefineGeometry(nx[lev], ny[lev], nz[lev], dx[lev]);
        ba[lev] = DefineBoxArray(nx[lev], ny[lev], nz[lev]);
        dm[lev] = DefineDM(ba[lev]);
        DefineFABs(pressure[lev], velocity[lev], ba[lev], dm[lev]);
    }

    constexpr amrex::Real tolerance = 1.0e-6;
    constexpr amrex::Real omega = 1.0;
    const int max_iterations = 500;

    for (int lev = 0; lev < nlevels; ++lev) {
        amrex::Print() << "\nLevel " << lev << ": domain size " << nx[lev] << "x" << ny[lev] << "x" << nz[lev] << ", dx = " << dx[lev] << "\n";
        amrex::Print() << "  Tolerance: " << tolerance << ", Max iterations: " << max_iterations << ", Relaxation: " << omega << "\n";

        // Nonzero face indices (centered)
        int i_face = nx[lev]/2;
        int j_face = ny[lev]/2;
        int k_face = nz[lev]/2;

        InitializeVelocity(velocity[lev], geom[lev], i_face, j_face, k_face);
        SolvePressure(pressure[lev], velocity[lev], geom[lev], tolerance, max_iterations, omega, i_face, j_face, k_face);
        CheckResults(pressure[lev], velocity[lev], geom[lev], i_face, j_face, k_face);
    }

    // Optionally: Sample and output results for all levels to a combined file here
    return 0;
}


/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/ 
