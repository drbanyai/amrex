/*
 *  Testbed implementing a single-level Gauss-Seidel pressure solver on a MAC
 * grid.
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
void CompositeSolve(  //
  std::vector<LevelData>& composite_levels,
  amrex::Real tolerance,
  int max_iterations,
  int base_n,
  int nlevels,
  int level = 0 )
{
  const int fineN = base_n * ( 1 << ( nlevels - 1 ) );
  const int nLevels = composite_levels.size();
  amrex::Print()  //
    << "\nLevel: " << level
    << ", domain: " << composite_levels[level].geom.Domain()
    << ", dx = " << composite_levels[level].geom.CellSize()[0] << "\n";
  amrex::Print()  //
    << "  Tolerance: " << tolerance << ", Max iterations: " << max_iterations
    << "\n";

  if ( level > 0 ) {
    // Fill ghost cells in N using N-1 results
    FillPressureGhostCells(  //
      composite_levels[level],
      composite_levels[level - 1] );
  }

  // Single-level solve on N
  SolvePressure(  //
    composite_levels[level].pressure,
    composite_levels[level].velocity,
    composite_levels[level].geom,
    tolerance,
    max_iterations,
    fineN );

  if ( level < nLevels - 1 ) {
    // Composite solve on N+1 and above
    CompositeSolve(  //
      composite_levels,
      tolerance,
      max_iterations,
      base_n,
      nlevels,
      level + 1 );

    // Correction solve on N using N+1/N flux mismatch
    SolvePressureCorrection(  //
      composite_levels[level].pressure,
      composite_levels[level + 1].pressure,
      composite_levels[level].geom,
      composite_levels[level + 1].geom,
      tolerance,
      max_iterations,
      fineN );

    // Composite solve on N+1 and above, using corrected N
    CompositeSolve(  //
      composite_levels,
      tolerance,
      max_iterations,
      base_n,
      nlevels,
      level + 1 );
  }
}

int main( int argc, char** argv )
{
  amrex::Initialize( argc, argv );
  const int ret = MyMain();
  amrex::Finalize();
  return ret;
}

int MyMain()
{
  // Number of levels (AMR-ready, even if only single-level for now)
  constexpr int nlevels = 3;
  constexpr int baseN = 4;
  constexpr int fineN = baseN * ( 1 << ( nlevels - 1 ) );
  constexpr amrex::Real domain_length = 1.0;  // meters
  constexpr amrex::Real tolerance = 1.0e-8;
  constexpr int max_iterations = 500;

  // First create and work with fine-level data
  LevelData fine_level = MakeDenseLevelData(  //
    fineN,
    domain_length,
    baseN,
    nlevels );

  // Loop over levels: setup geometry, mesh, fields
  std::vector<LevelData> composite_levels = MakeSparseCompositeLevels(  //
    baseN,
    nlevels,
    domain_length );

  // Work with fine-level data
  amrex::Print()  //
    << "\nComplete fine solution, domain: " << fine_level.geom.Domain()
    << ", dx = " << fine_level.geom.CellSize()[0] << "\n";
  amrex::Print()                     //
    << "  Tolerance: " << tolerance  //
    << ", Max iterations: " << max_iterations << "\n";
  SolvePressure(  //
    fine_level.pressure,
    fine_level.velocity,
    fine_level.geom,
    tolerance,
    max_iterations,
    fineN );
  CheckResults( fine_level.pressure, fine_level.geom, fineN );

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
      fineN );
  }

  // Compare fine pressure with all levels
  CompareMultiFabs( fine_level, composite_levels );

  // Sample and output results for all levels to a combined CSV file
  SamplePressureAlongLine(  //
    composite_levels,
    "pressure.csv",
    fine_level,
    fineN );

  return 0;
}

/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/
