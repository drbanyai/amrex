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
    // Number of levels (AMR-ready, even if only single-level for now)
    constexpr int nlevels = 2;
    constexpr int base_n = 4;
    constexpr amrex::Real domain_length = 1.0; // meters
    constexpr amrex::Real tolerance = 1.0e-6;
    constexpr amrex::Real omega = 1.0;
    constexpr int max_iterations = 500;

    // First create and work with fine-level data
    const int fine_n = 8;
    LevelData fine_level = MakeDenseLevelData(fine_n, domain_length);

    // Loop over levels: setup geometry, mesh, fields
    std::vector<LevelData> composite_levels = MakeSparseCompositeLevels(base_n, nlevels, domain_length);

    // Work with fine-level data
    amrex::Print() << "\nComplete fine solution, domain: " << fine_level.geom.Domain() << ", dx = " << fine_level.geom.CellSize()[0] << "\n";
    amrex::Print() << "  Tolerance: " << tolerance << ", Max iterations: " << max_iterations << ", Relaxation: " << omega << "\n";
    SolvePressure(fine_level.pressure, fine_level.velocity, fine_level.geom, tolerance, max_iterations, omega);
    CheckResults(fine_level.pressure, fine_level.velocity, fine_level.geom);

    // Loop over levels: solve and check results
    // TODO: Need to implement full composite solve
    for (int lev = 0; lev < nlevels; ++lev) {
        amrex::Print() << "\nLevel: " << lev << ", domain: " << composite_levels[lev].geom.Domain() << ", dx = " << composite_levels[lev].geom.CellSize()[0] << "\n";
        amrex::Print() << "  Tolerance: " << tolerance << ", Max iterations: " << max_iterations << ", Relaxation: " << omega << "\n";
        
        // Fill ghost cells for pressure from coarser level
        if (lev > 0) {
            FillPressureGhostCells(composite_levels[lev], composite_levels[lev-1]);
        }
        
        // Fine-level solve
        SolvePressure(composite_levels[lev].pressure, composite_levels[lev].velocity, composite_levels[lev].geom, tolerance, max_iterations, omega);

        if (lev > 0) {
          SolvePressureCorrection(composite_levels[lev-1].pressure,
                                  composite_levels[lev].pressure,
                                  composite_levels[lev-1].geom,
                                  composite_levels[lev].geom,
                                  tolerance,
                                  max_iterations,
                                  omega);
          // TODO: Fill fine ghosts from coarse level
          // TODO: Re-solve fine level
        }
    }

    amrex::Print() << "\nChecking results for all levels\n";
    for (int lev = 0; lev < nlevels; ++lev) {
        amrex::Print() << "\nLevel: " << lev << ", domain: " << composite_levels[lev].geom.Domain() << ", dx = " << composite_levels[lev].geom.CellSize()[0] << "\n";
        CheckResults(composite_levels[lev].pressure, composite_levels[lev].velocity, composite_levels[lev].geom);
    }

    // Compare fine pressure with all levels
    CompareMultiFabs(fine_level, composite_levels);

    // Sample and output results for all levels to a combined CSV file
    SamplePressureAlongLine(composite_levels, "pressure.csv");

    return 0;
}


/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/ 
