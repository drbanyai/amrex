/*
 *  Testbed implementing a single-level Gauss-Seidel pressure solver on a MAC
 * grid.
 */

/*--------------------------------------------------------------------
  standard includes
  --------------------------------------------------------------------*/
#include <AMReX.H>
#include <AMReX_ParmParse.H>

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
int main( int argc, char** argv )
{
  amrex::Initialize( argc, argv );
  const int ret = MyMain();
  amrex::Finalize();
  return ret;
}

int MyMain()
{
  // Parse command line parameters
  int nlevels = 2;
  int baseN = 4;
  bool dense = false;
  {
    amrex::ParmParse pp;
    pp.query( "n_levels", nlevels );
    pp.query( "base_n", baseN );
    pp.query( "dense", dense );
    amrex::Print()               //
      << "nlevels: " << nlevels  //
      << ", baseN: " << baseN    //
      << ", dense: " << dense    //
      << "\n";
  }

  constexpr amrex::Real domain_length = 1.0;  // meters
  constexpr amrex::Real tolerance = 1.0e-8;
  constexpr int max_iterations = 500;

  // Loop over levels: setup geometry, mesh, fields
  std::vector<LevelData> composite_levels = MakeSparseCompositeLevels(  //
    baseN,
    nlevels,
    domain_length );

  // Solve on the full composite mesh
  CompositeSolve( composite_levels, tolerance, max_iterations, baseN, nlevels );

  amrex::Print() << "\nChecking results for all levels\n";
  for ( int lev = 0; lev < nlevels; ++lev ) {
    amrex::Print()  //
      << "\nLevel: " << lev
      << ", domain: " << composite_levels[lev].geom.Domain()
      << ", dx = " << composite_levels[lev].geom.CellSize()[0] << "\n";
    CheckResults(  //
      composite_levels[lev].pressure,
      composite_levels[lev].geom,
      baseN,
      nlevels );
  }

  if ( dense ) {
    // First create and work with fine-level data
    LevelData fine_level = MakeDenseLevelData(  //
      domain_length,
      CalculateFineN( baseN, nlevels ),
      1,    // 1 level
      0 );  // level zero

    // Work with fine-level data
    amrex::Print()  //
      << "\nComplete fine solution, domain: " << fine_level.geom.Domain()
      << ", dx = " << fine_level.geom.CellSize()[0] << "\n";
    amrex::Print()                     //
      << "  Tolerance: " << tolerance  //
      << ", Max iterations: " << max_iterations << "\n";
    SingleLevelPressureSolve(  //
      fine_level.pressure,
      fine_level.velocity,
      fine_level.geom,
      tolerance,
      max_iterations,
      baseN,
      nlevels );
    CheckResults( fine_level.pressure, fine_level.geom, baseN, nlevels );

    // Compare fine pressure with all levels
    CompareMultiFabs( fine_level, composite_levels );

    // Sample and output results for all levels to a combined CSV file
    SamplePressureAlongLine(  //
      composite_levels,
      "pressure.csv",
      fine_level,
      baseN,
      nlevels );
  }

  return 0;
}

/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/
