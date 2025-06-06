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
  int nlevels = 3;
  int baseN = 8;
  bool dense = true;
  amrex::Real domain_length = 1.0;  // meters
  amrex::Real tolerance = 1.0e-8;
  int max_iterations = 500;
  {
    amrex::ParmParse pp;
    pp.query( "n_levels", nlevels );
    pp.query( "base_n", baseN );
    pp.query( "dense", dense );
    pp.query( "domain_length", domain_length );
    pp.query( "tolerance", tolerance );
    pp.query( "max_iterations", max_iterations );
    amrex::Print()                                 //
      << "Equivalent command line parameters:\n "  //
      << " nlevels=" << nlevels                    //
      << " baseN=" << baseN                        //
      << " dense=" << dense                        //
      << " domain_length=" << domain_length        //
      << " tolerance=" << tolerance                //
      << " max_iterations=" << max_iterations      //
      << "\n";
  }

  // Create composite mesh
  std::vector<LevelData> composite_levels =  //
    MakeSparseCompositeLevels(               //
      baseN,
      nlevels,
      domain_length );

  // Solve on composite mesh
  CompositeSolve(  //
    composite_levels,
    tolerance,
    max_iterations,
    baseN,
    nlevels );

  for ( int lev = 0; lev < nlevels; ++lev ) {
    CheckResults(  //
      composite_levels[lev].pressure,
      composite_levels[lev].geom,
      baseN,
      nlevels );
  }

  if ( dense ) {
    // First create and work with fine-level data
    LevelData dense_level = MakeDenseLevelData(  //
      domain_length,
      CalculateFineN( baseN, nlevels ),
      1,    // Only 1 level
      0 );  // This is level zero

    // Work with fine-level data
    amrex::Print()  //
      << "\nDense fine solution, domain: " << dense_level.geom.Domain()
      << ", dx = " << dense_level.geom.CellSize()[0] << "\n";
    SingleLevelPressureSolve(  //
      dense_level.pressure,
      dense_level.velocity,
      dense_level.geom,
      tolerance,
      max_iterations,
      baseN,
      nlevels );

    CheckResults( dense_level.pressure, dense_level.geom, baseN, nlevels );

    // Compare dense results with sparse results
    CompareMultiFabs( dense_level, composite_levels );

    // Sample results and create plots
    SamplePressureAlongLine(  //
      composite_levels,
      dense_level,
      baseN,
      nlevels );
  }

  return 0;
}

/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/
