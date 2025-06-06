/*--------------------------------------------------------------------
  associated include
  --------------------------------------------------------------------*/
#include "helpers.H"

/*--------------------------------------------------------------------
  standard includes
  --------------------------------------------------------------------*/
#include <AMReX_BCUtil.H>
#include <AMReX_Config.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_Geometry.H>
#include <AMReX_Interpolater.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Print.H>

/*--------------------------------------------------------------------
  defines and static variables
  --------------------------------------------------------------------*/
// Helpers for indexing velocity components
static constexpr int U = 0;
static constexpr int V = 1;
static constexpr int W = 2;

/*--------------------------------------------------------------------
  private free function declarations
  --------------------------------------------------------------------*/
static amrex::Geometry DefineGeometry(  //
  int base_n,
  int level,
  amrex::Real domain_length );
static amrex::BoxArray DefineBoxArray( int base_n, int level );
static amrex::BoxArray DefineSparseBoxArray( int base_n, int level );
static amrex::DistributionMapping DefineDM( const amrex::BoxArray& ba );
static amrex::DistributionMapping DefineIOProcessorDM(
  const amrex::BoxArray& ba );
static void DefineFABs(  //
  amrex::MultiFab& pressure,
  std::array<amrex::MultiFab, 3>& velocity,
  const amrex::BoxArray& ba,
  const amrex::DistributionMapping& dm );
static void RecursiveCompositeSolve(  //
  std::vector<LevelData>& composite_levels,
  amrex::Real tolerance,
  int max_iterations,
  int base_n,
  int nLevels,
  int level );
static amrex::Real ExpectedPressure(  //
  const amrex::Geometry& geom,
  int i,
  int j,
  int k,
  int base_n,
  int nLevels );
static void GaussSeidelIteration(  //
  bool use_expected_BCs,
  amrex::MultiFab& pressure,
  const amrex::MultiFab& divergence,
  const amrex::Geometry& geom,
  int iteration,
  int base_n,
  int nLevels );
static amrex::Real ComputeResidual(  //
  const amrex::MultiFab& pressure,
  const amrex::MultiFab& divergence,
  const amrex::Geometry& geom );
static void ComputeDivergence(  //
  amrex::MultiFab& divergence,
  const std::array<amrex::MultiFab, 3>& velocity,
  const amrex::Geometry& geom );
static void SolvePressureIterations(  //
  bool use_expected_BCs,
  amrex::MultiFab& pressure,
  const amrex::MultiFab& divergence,
  const amrex::Geometry& geom,
  amrex::Real tolerance,
  int max_iterations,
  int base_n,
  int nLevels );
static amrex::Real Phi(  //
  amrex::Real r_x,
  amrex::Real r_y,
  amrex::Real r_z,
  amrex::Real ri_x,
  amrex::Real ri_y,
  amrex::Real ri_z,
  amrex::Real dx );

// Average Phi over a target cube of size dx centered at (x,y,z)
static amrex::Real AveragePhi(  //
  amrex::Real x,
  amrex::Real y,
  amrex::Real z,
  amrex::Real ri_x,
  amrex::Real ri_y,
  amrex::Real ri_z,
  amrex::Real dx )
{
  // Check if target cube overlaps with source cube
  const amrex::Real dr_x = x - ri_x;
  const amrex::Real dr_y = y - ri_y;
  const amrex::Real dr_z = z - ri_z;
  const amrex::Real r2 = dr_x * dr_x + dr_y * dr_y + dr_z * dr_z;

  // If target cube center is at the same cell center as source cube,
  // use analytic solution
  if ( r2 < 1.0e-8 ) {  // Effectively zero distance
    // Analytic solution for self-potential at the center of the cube
    return 2.0 * 1.516386 / ( 4.0 * M_PI * dx );
  }

  amrex::Real sum = 0.0;
  const amrex::Real half_dx = 0.5 * dx;
  constexpr amrex::Real sqrt3over5 = 0.7745966692414834;
  // Use a 3x3x3 quadrature rule (27 points) for better accuracy
  for ( int i = -1; i <= 1; i++ ) {
    for ( int j = -1; j <= 1; j++ ) {
      for ( int k = -1; k <= 1; k++ ) {
        // √(3/5)
        const amrex::Real x2 = x + i * half_dx * sqrt3over5;
        const amrex::Real y2 = y + j * half_dx * sqrt3over5;
        const amrex::Real z2 = z + k * half_dx * sqrt3over5;

        // Use Gauss-Legendre weights
        const amrex::Real w = ( i == 0 ? 0.8888888888888888
                                       : 0.5555555555555556 ) *
                              ( j == 0 ? 0.8888888888888888
                                       : 0.5555555555555556 ) *
                              ( k == 0 ? 0.8888888888888888
                                       : 0.5555555555555556 );

        sum += w * Phi( x2, y2, z2, ri_x, ri_y, ri_z, dx );
      }
    }
  }

  // The quadrature weights are normalized to sum to 8 (the volume of the cube)
  return sum / 8.0;
}

/*--------------------------------------------------------------------
  public free function definitions
  --------------------------------------------------------------------*/
void CompositeSolve(  //
  std::vector<LevelData>& composite_levels,
  amrex::Real tolerance,
  int max_iterations,
  int base_n,
  int nLevels )
{
  BL_PROFILE( "CompositeSolve" );

  RecursiveCompositeSolve(  //
    composite_levels,
    tolerance,
    max_iterations,
    base_n,
    nLevels,
    0 );
}

/*--------------------------------------------------------------------
  private free function definitions
  --------------------------------------------------------------------*/
void RecursiveCompositeSolve(  //
  std::vector<LevelData>& composite_levels,
  amrex::Real tolerance,
  int max_iterations,
  int base_n,
  int nLevels,
  int level )
{
  BL_PROFILE( "RecursiveCompositeSolve" );
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
  SingleLevelPressureSolve(  //
    composite_levels[level].pressure,
    composite_levels[level].velocity,
    composite_levels[level].geom,
    tolerance,
    max_iterations,
    base_n,
    nLevels );

  if ( level < nLevels - 1 ) {
    // Composite solve on N+1 and above
    RecursiveCompositeSolve(  //
      composite_levels,
      tolerance,
      max_iterations,
      base_n,
      nLevels,
      level + 1 );

    // Correction solve on N using N+1/N flux mismatch
    SolvePressureCorrection(  //
      composite_levels[level].pressure,
      composite_levels[level + 1].pressure,
      composite_levels[level].geom,
      composite_levels[level + 1].geom,
      tolerance,
      max_iterations,
      base_n,
      nLevels );

    // Composite solve on N+1 and above, using corrected N
    RecursiveCompositeSolve(  //
      composite_levels,
      tolerance,
      max_iterations,
      base_n,
      nLevels,
      level + 1 );
  }
}
static amrex::Real ExpectedPressure(  //
  const amrex::Geometry& geom,
  int i,
  int j,
  int k,
  int base_n,
  int nLevels )
{
  // Get the cell center coordinates for the target point
  const amrex::Real x = geom.CellCenter( i, U );
  const amrex::Real y = geom.CellCenter( j, V );
  const amrex::Real z = geom.CellCenter( k, W );

  // Calculate the source positions (centers of the source cubes)
  const int fineN = CalculateFineN( base_n, nLevels );
  const amrex::Real dx = 1.0 / fineN;  // Cube size
  const amrex::Real centerMinus = ( ( fineN / 2.0 ) - 0.5 ) /
                                  fineN;  // Left source cube center
  const amrex::Real centerPlus = ( ( fineN / 2.0 ) + 0.5 ) /
                                 fineN;  // Right source cube center

  // Compute the potential from each source cube, averaged over the target cube
  const amrex::Real phi1 =
    AveragePhi( x, y, z, centerMinus, centerPlus, centerPlus, dx );
  const amrex::Real phi2 =
    AveragePhi( x, y, z, centerPlus, centerPlus, centerPlus, dx );

  // Sum the contributions from the two source cubes
  const amrex::Real p = -1.0 * phi1 + phi2;

  return p;
}

class PressureBndryFunc
{
public:
  PressureBndryFunc( int base_n, int nLevels )
    : vBaseN( base_n )
    , vNLevels( nLevels )
  {
  }

  void operator()(  //
    amrex::Box const& bx,
    amrex::FArrayBox& data,
    const int dcomp,
    const int numcomp,
    amrex::Geometry const& geom,
    const amrex::Real time,
    const amrex::Vector<amrex::BCRec>& bcr,
    const int bcomp,
    const int scomp ) const
  {
    const auto lo = amrex::lbound( bx );
    const auto hi = amrex::ubound( bx );
    const auto arr = data.array();
    const amrex::Box& valid_box = geom.Domain();

    amrex::ParallelFor(  //
      bx,
      numcomp,
      [=] AMREX_GPU_DEVICE( int i, int j, int k, int n ) noexcept {
        // Only operate on external ghost cells
        if ( !valid_box.contains( i, j, k ) ) {
          arr( i, j, k, n + dcomp ) =  //
            ExpectedPressure( geom, i, j, k, vBaseN, vNLevels );
        }
      } );
  }

private:
  int vBaseN = 0;
  int vNLevels = 0;
};

amrex::Geometry DefineGeometry(  //
  int base_n,
  int level,
  amrex::Real domain_length )
{
  const int nx = CalculateNForLevel( base_n, level );
  const int ny = nx;
  const int nz = nx;
  const amrex::Real dx = domain_length / nx;
  const amrex::RealBox real_box( { 0.0, 0.0, 0.0 },
                                 { nx * dx, ny * dx, nz * dx } );
  constexpr amrex::CoordSys::CoordType coord =
    amrex::CoordSys::CoordType::cartesian;
  const amrex::IntArray is_periodic{ 0, 0, 0 };  // Non-periodic

  const amrex::Box domain( amrex::IntVect( 0, 0, 0 ),
                           amrex::IntVect( nx - 1, ny - 1, nz - 1 ) );
  return amrex::Geometry( domain, real_box, coord, is_periodic );
}

amrex::BoxArray DefineBoxArray( int base_n, int level )
{
  const int n = CalculateNForLevel( base_n, level );
  const amrex::Box domainBox( amrex::IntVect( 0, 0, 0 ),
                              amrex::IntVect( n - 1, n - 1, n - 1 ) );
  return amrex::BoxArray( domainBox );
}

amrex::BoxArray DefineSparseBoxArray( int base_n, int level )
{
  const int n = CalculateNForLevel( base_n, level );
  const int lo = ( n / 2 ) - base_n / 2;
  const int hi = ( n / 2 ) + base_n / 2 - 1;
  const amrex::Box sparseBox( amrex::IntVect( lo, lo, lo ),
                              amrex::IntVect( hi, hi, hi ) );
  return amrex::BoxArray( sparseBox );
}

amrex::DistributionMapping DefineDM( const amrex::BoxArray& ba )
{
  return amrex::DistributionMapping( ba );
}

static amrex::DistributionMapping DefineIOProcessorDM(  //
  const amrex::BoxArray& ba )
{
  return amrex::DistributionMapping( amrex::Vector<int>(  //
    ba.size(),
    amrex::ParallelDescriptor::IOProcessorNumber() ) );
}

void DefineFABs(  //
  amrex::MultiFab& pressure,
  std::array<amrex::MultiFab, 3>& velocity,
  const amrex::BoxArray& ba,
  const amrex::DistributionMapping& dm )
{
  constexpr int SingleComp = 1;
  constexpr int PressureGhosts = 1;
  constexpr int VelocityGhosts = 1;

  // Pressure is cell-centered
  pressure.define( ba, dm, SingleComp, PressureGhosts );
  pressure.setVal( 0.0 );  // Initialize all cells (including ghosts) to zero

  // Velocity components are face-centered
  const auto Uba = amrex::convert( ba,
                                   amrex::IntVect::TheDimensionVector( U ) );
  const auto Vba = amrex::convert( ba,
                                   amrex::IntVect::TheDimensionVector( V ) );
  const auto Wba = amrex::convert( ba,
                                   amrex::IntVect::TheDimensionVector( W ) );

  velocity[U].define( Uba, dm, SingleComp, VelocityGhosts );
  velocity[V].define( Vba, dm, SingleComp, VelocityGhosts );
  velocity[W].define( Wba, dm, SingleComp, VelocityGhosts );

  // Initialize velocity components to zero
  for ( int d = 0; d < 3; ++d ) {
    // Initialize all cells (including ghosts) to zero
    velocity[d].setVal( 0.0 );
  }
}

static void InitializeVelocity(  //
  std::array<amrex::MultiFab, 3>& velocity,
  const amrex::Geometry& geom,
  int base_n,
  int nLevels,
  int level )
{
  // Set all velocities to zero
  for ( int d = 0; d < 3; ++d ) {
    velocity[d].setVal( 0.0 );
  }

  if ( level == nLevels - 1 ) {
    const int halfN = geom.Domain().length( 0 ) / 2;
    const int i_face = halfN;
    const int j_face = halfN;
    const int k_face = halfN;

    // Set a single nonzero x-face at the specified face indices
    for ( amrex::MFIter mfi( velocity[U] ); mfi.isValid(); ++mfi ) {
      const amrex::Box& box = mfi.validbox();
      const auto& u_arr = velocity[U].array( mfi );
      if ( box.contains( i_face, j_face, k_face ) ) {
        const amrex::Real dx = geom.CellSize( 0 );
        u_arr( i_face, j_face, k_face ) = 1.0 / ( dx * dx );
      }
    }

    // Fill ghost cells
    for ( int d = 0; d < 3; ++d ) {
      velocity[d].FillBoundary( geom.periodicity() );
    }
  }
}

void SamplePressureAlongLine(  //
  const std::vector<LevelData>& level_data,
  const std::string& filename,
  const LevelData& fullFineSolution,
  int base_n,
  int nLevels )
{
  // Create a copy of fullFineSolution.pressure with all FABs on the IOProcessor
  const amrex::BoxArray& fine_ba = fullFineSolution.pressure.boxArray();
  const amrex::DistributionMapping fine_dm = DefineIOProcessorDM( fine_ba );
  amrex::MultiFab full_fine_pressure(  //
    fine_ba,
    fine_dm,
    fullFineSolution.pressure.nComp(),
    fullFineSolution.pressure.nGrow() );
  full_fine_pressure.ParallelCopy( fullFineSolution.pressure );

  const int finest_lev = static_cast<int>( level_data.size() ) - 1;
  const amrex::Geometry& fine_geom = level_data[finest_lev].geom;

  // Get centerline indices for y and z at finest level
  const amrex::Real center_y = 0.5 * ( fine_geom.ProbLo( 1 ) +
                                       fine_geom.ProbHi( 1 ) );
  const amrex::Real center_z = 0.5 * ( fine_geom.ProbLo( 2 ) +
                                       fine_geom.ProbHi( 2 ) );

  // Write raw level 0 pressure values to CSV for debugging
  if ( amrex::ParallelDescriptor::IOProcessor() ) {
    std::ofstream raw_outfile( "pressure_L0.csv" );
    raw_outfile << "x,raw_pressure_L0,analytic\n";

    const amrex::Geometry& level0_geom = level_data[0].geom;
    const amrex::Box& level0_domain = level0_geom.Domain();
    amrex::Real point[AMREX_SPACEDIM] = { 0.5, center_y, center_z };
    amrex::IntVect cell_idx = level0_geom.CellIndex( point );
    const int j0 = cell_idx[1];
    const int k0 = cell_idx[2];

    for ( int i = level0_domain.smallEnd( 0 ); i <= level0_domain.bigEnd( 0 );
          ++i ) {
      const amrex::Real x = level0_geom.CellCenter( i, 0 );
      amrex::Real val = 0.0;
      bool found = false;
      for ( amrex::MFIter mfi( level_data[0].pressure ); mfi.isValid();
            ++mfi ) {
        const amrex::Box& box = mfi.validbox();
        if ( box.contains( amrex::IntVect( i, j0, k0 ) ) ) {
          const auto& parr = level_data[0].pressure.array( mfi );
          val = parr( i, j0, k0 );
          found = true;
          break;
        }
      }
      if ( !found ) {
        amrex::Print() << "No raw value found for level 0 at (i, j, k) = (" << i
                       << ", " << j0 << ", " << k0 << ")\n";
      }
      const amrex::Real expected = ExpectedPressure( level0_geom,
                                                     i,
                                                     j0,
                                                     k0,
                                                     base_n,
                                                     nLevels );
      raw_outfile << x << "," << val << "," << expected << "\n";
    }
    raw_outfile.close();
  }

  // Create MultiFabs to store interpolated pressure at each level
  amrex::Vector<amrex::MultiFab> interpolated_pressure( level_data.size() );
  amrex::Vector<amrex::MultiFab> dense_level_pressure( level_data.size() );
  for ( int lev = 0; lev <= finest_lev; ++lev ) {
    const amrex::BoxArray& fine_ba = full_fine_pressure.boxArray();
    const amrex::DistributionMapping fine_dm = DefineIOProcessorDM( fine_ba );
    const int ncomp = fullFineSolution.pressure.nComp();
    const int nGrow = fullFineSolution.pressure.nGrow();
    // Define MultiFab with same structure as full fine pressure.
    interpolated_pressure[lev].define( fine_ba, fine_dm, ncomp, nGrow );
    interpolated_pressure.at( lev ).setVal( 0.0 );

    // Calculate refinement ratio between fine level and current level
    const amrex::Box& fine_domain = level_data[finest_lev].geom.Domain();
    const amrex::Box& current_domain = level_data[lev].geom.Domain();
    const amrex::IntVect fine_size = fine_domain.size();
    const amrex::IntVect current_size = current_domain.size();
    // Ratio should be uniform in all dimensions
    const int ratio = fine_size[0] / current_size[0];
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( ratio == fine_size[1] / current_size[1] &&
                                        ratio == fine_size[2] / current_size[2],
                                      "Refinement ratio must be uniform in all "
                                      "dimensions" );
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE( ratio >= 1,
                                      "Fine level must be refined version of "
                                      "current level" );
    const amrex::BoxArray& dense_ba = amrex::coarsen( fine_ba, ratio );
    // Use same IOProcessor DistributionMapping for dense level
    const amrex::DistributionMapping dense_dm = DefineIOProcessorDM( dense_ba );
    dense_level_pressure[lev].define( dense_ba, dense_dm, ncomp, nGrow );
    dense_level_pressure.at( lev ).setVal( 0.0 );
    dense_level_pressure.at( lev ).ParallelCopy( level_data[lev].pressure );

    // Print some debug info
    if ( amrex::ParallelDescriptor::IOProcessor() ) {
      amrex::Print() << "\nLevel " << lev << " domain info (ratio = " << ratio
                     << "):\n"
                     << "  Original domain: " << current_domain << "\n"
                     << "  Dense box array: " << dense_ba << "\n"
                     << "  Original box array: "
                     << level_data[lev].pressure.boxArray() << "\n";
    }
  }

  // Set up boundary conditions for pressure
  amrex::Vector<amrex::BCRec> bcs( 1 );  // One component for pressure
  for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim ) {
    bcs.at( 0 ).setLo( idim, amrex::BCType::ext_dir );
    bcs.at( 0 ).setHi( idim, amrex::BCType::ext_dir );
  }

  // Fill all levels with proper interpolation
  for ( int lev = 0; lev <= finest_lev; ++lev ) {
    // Create boundary condition functors for each level
    amrex::Vector<amrex::PhysBCFunct<PressureBndryFunc>> physbcs;
    for ( int thisLevel = 0; thisLevel <= lev; ++thisLevel ) {
      physbcs.emplace_back(  //
        level_data[thisLevel].geom,
        bcs,
        PressureBndryFunc( base_n, nLevels ) );
    }

    // Prepare data for FillPatchNLevels
    amrex::Vector<amrex::Vector<amrex::MultiFab*>> smf;
    amrex::Vector<amrex::Vector<amrex::Real>> st;
    amrex::Vector<amrex::Geometry> geom;
    amrex::Vector<amrex::IntVect> ratio;

    for ( int thisLevel = 0; thisLevel <= lev; ++thisLevel ) {
      smf.emplace_back(  //
        amrex::Vector<amrex::MultiFab*>{
          &dense_level_pressure.at( thisLevel ) } );
      st.emplace_back( amrex::Vector<amrex::Real>{ 0.0 } );
      geom.emplace_back( level_data[thisLevel].geom );

      // Calculate the correct ratio between this level and the finest level
      // For level L, ratio to finest level is 2^(finest_lev - L)
      const int level_ratio = 1 << ( finest_lev - thisLevel );
      ratio.emplace_back(
        amrex::IntVect( level_ratio, level_ratio, level_ratio ) );

      if ( amrex::ParallelDescriptor::IOProcessor() ) {
        amrex::Print() << "\nFillPatchNLevels for level " << thisLevel
                       << " using ratio " << level_ratio << " (2^"
                       << ( finest_lev - thisLevel ) << ")\n";
      }
    }

    auto& outMF = interpolated_pressure.at( lev );
    constexpr amrex::Real time = 0.0;
    constexpr int scomp = 0;
    constexpr int dcomp = 0;
    constexpr int ncomp = 1;
    constexpr int bccomp = 0;
    constexpr int bcrcomp = 0;
    amrex::Vector<amrex::BCRec> bcr( 1 );
    for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim ) {
      bcr[0].setLo( idim, amrex::BCType::ext_dir );
      bcr[0].setHi( idim, amrex::BCType::ext_dir );
    }
    amrex::FillPatchNLevels(  //
      outMF,
      lev,                  // level
      amrex::IntVect( 0 ),  // outMF.nGrowVect(),  // nghost // No difference
      time,                 // time
      smf,                  // source MultiFabs
      st,                   // source times
      scomp,                // source component
      dcomp,                // destination component
      ncomp,                // number of components
      geom,                 // geometries
      physbcs,              // boundary conditions
      bccomp,               // boundary condition component
      ratio,                // refinement ratios
      // Interpolator makes a significant difference here
      // &amrex::pc_interp,
      // &amrex::cell_bilinear_interp,
      &amrex::quadratic_interp,
      // &amrex::cell_quartic_interp,
      bcr,        // boundary conditions
      bcrcomp );  // boundary condition component
  }

  // Only rank 0 process should write output files
  if ( amrex::ParallelDescriptor::IOProcessor() ) {

    // Prepare output
    std::ofstream outfile( filename );
    outfile << "x";
    for ( int lev = 0; lev <= finest_lev; ++lev ) {
      outfile << ",pressure_L" << lev;
    }
    outfile << ",full_fine";  // Add column for full fine solution
    outfile << ",expected\n";

    // Sample along x-axis at center_y, center_z
    const amrex::Box& fine_domain = fine_geom.Domain();

    // For each level, sample pressure at (i, center_j, center_k)
    amrex::Real point[AMREX_SPACEDIM] = { 0.5, center_y, center_z };
    amrex::IntVect cell_idx = fine_geom.CellIndex( point );
    const int j = cell_idx[1];
    const int k = cell_idx[2];

    for ( int i = fine_domain.smallEnd( 0 ); i <= fine_domain.bigEnd( 0 );
          ++i ) {
      const amrex::Real x = fine_geom.CellCenter( i, 0 );
      outfile << x;

      // Sample from interpolated levels
      for ( int lev = 0; lev <= finest_lev; ++lev ) {
        amrex::Real val = 0.0;
        bool found = false;
        for ( amrex::MFIter mfi( interpolated_pressure[lev] ); mfi.isValid();
              ++mfi ) {
          const amrex::Box& box = mfi.validbox();
          if ( box.contains( amrex::IntVect( i, j, k ) ) ) {
            const auto& parr = interpolated_pressure[lev].array( mfi );
            val = parr( i, j, k );
            found = true;
            break;
          }
        }
        if ( !found ) {
          amrex::Print() << "No value found for level " << lev
                         << " at (i, j, k) = (" << i << ", " << j << ", " << k
                         << ")\n";
          // Print more info about the point we're trying to sample
          const amrex::Real x = fine_geom.CellCenter( i, 0 );
          const amrex::Real y = fine_geom.CellCenter( j, 1 );
          const amrex::Real z = fine_geom.CellCenter( k, 2 );
          amrex::Print() << "  Physical coordinates: (" << x << "," << y << ","
                         << z << ")\n";
          amrex::Print() << "  Fine domain: " << fine_geom.Domain() << "\n";
          amrex::Print() << "  Level " << lev
                         << " domain: " << level_data[lev].geom.Domain()
                         << "\n";
        }
        assert( found );
        outfile << "," << val;
      }

      // Sample from full fine solution
      amrex::Real full_fine_val = 0.0;
      bool found = false;
      for ( amrex::MFIter mfi( full_fine_pressure ); mfi.isValid(); ++mfi ) {
        const amrex::Box& box = mfi.validbox();
        if ( box.contains( amrex::IntVect( i, j, k ) ) ) {
          const auto& parr = full_fine_pressure.array( mfi );
          full_fine_val = parr( i, j, k );
          found = true;
          break;
        }
        if ( !found ) {
          amrex::Print()
            << "No value found for full fine solution at (i, j, k) = (" << i
            << ", " << j << ", " << k << ")\n";
        }
      }
      assert( found );
      outfile << "," << full_fine_val;

      // Add analytic solution as last column
      const amrex::Real expected =  //
        ExpectedPressure( fine_geom, i, j, k, base_n, nLevels );
      outfile << "," << expected;
      outfile << "\n";
    }
    outfile.close();

    // Create gnuplot script
    std::ofstream script( "plot_pressure.gp" );
    script << "set terminal png size 800,600\n";
    script << "set output 'pressure_profile.png'\n";
    script << "set xlabel 'x (m)'\n";
    script << "set ylabel 'Pressure (Pa)'\n";
    script << "set xzeroaxis\n";
    script << "set datafile separator ','\n";
    script << "set yrange [-1.5:1.5]\n";  // DEBUG
    script << "plot";
    script << " \\\n  '" << filename << "' using 1:" << ( finest_lev + 4 )
           << " title 'Analytic' with lines linetype -1 linewidth 3";
    script << " \\\n, 'pressure_L0.csv' using 1:2 title 'Raw Level 0'"
           << " with linespoints pointsize 2 linewidth 2";
    for ( int lev = 0; lev <= finest_lev; ++lev ) {
      script << " \\\n, '" << filename << "' using 1:" << ( lev + 2 )
             << " title 'Level " << lev << "'"
             << " with linespoints linewidth 2 pointsize 2";
    }
    script << " \\\n, '" << filename << "' using 1:" << ( finest_lev + 3 )
           << " title 'Dense' with linespoints linewidth 2 pointsize 2";
    script << "\n";

    // Add absolute pressure plot with log scale
    script << "set output 'pressure_profile_abs.png'\n";
    script << "set ylabel '|Pressure| (Pa)'\n";
    script << "set logscale y\n";
    script << "set yrange [*:*]\n";
    script << "plot";
    script << " \\\n  '" << filename << "' using 1:(abs($" << ( finest_lev + 4 )
           << ")) title '|Analytic|' with lines linetype -1 linewidth 3";
    script << " \\\n, 'pressure_L0.csv' using 1:(abs($2)) title '|Raw Level 0|'"
           << " with linespoints pointsize 2 linewidth 2";
    for ( int lev = 0; lev <= finest_lev; ++lev ) {
      script << " \\\n, '" << filename << "' using 1:(abs($" << ( lev + 2 )
             << ")) title '|Level " << lev << "|'"
             << " with linespoints linewidth 2 pointsize 2";
    }
    script << " \\\n, '" << filename << "' using 1:(abs($" << ( finest_lev + 3 )
           << ")) title '|Dense|' with linespoints linewidth 2 pointsize 2";
    script << "\n";

    script << "set output 'pressure_error_relative.png'\n";
    script << "set xlabel 'x (m)'\n";
    script << "set ylabel 'Pressure Error (unitless)'\n";
    script << "unset logscale y\n";
    script << "set yrange [*:*]\n";
    script << "set yrange [-0.5:0.5]\n";  // DEBUG
    script << "plot";
    // Error: full fine solution minus analytic
    script << " \\\n  '" << filename << "' using 1:(($" << ( finest_lev + 3 )
           << "-$" << ( finest_lev + 4 ) << ")/abs($" << ( finest_lev + 4 )
           << ")) title '(Dense - Analytic)/abs(Analytic)' with linespoints";
    // Error: finest level minus analytic
    script << " \\\n, '" << filename << "' using 1:(($" << ( finest_lev + 2 )
           << "-$" << ( finest_lev + 4 ) << ")/abs($" << ( finest_lev + 4 )
           << ")) title '(Fine - Analytic)/abs(Analytic)' with linespoints";
    // Error: finest level minus full fine
    script << " \\\n, '" << filename << "' using 1:(($" << ( finest_lev + 2 )
           << "-$" << ( finest_lev + 3 ) << ")/abs($" << ( finest_lev + 3 )
           << ")) title '(Fine - Dense)/abs(Dense)' with linespoints";
    // Error: raw level 0 minus analytic
    script << " \\\n, 'pressure_L0.csv' using 1:(($2-$3)/abs($3))"
           << " title '(Raw L0 - Analytic)/abs(Analytic)' with linespoints "
              "pointsize 2";
    script << " \\\n,  0.05 title '+/- 5%' lt 0";
    script << " \\\n, -0.05 title '' lt 0";
    script << "\n";
    script << "set output 'pressure_error_abs.png'\n";
    script << "set xlabel 'x (m)'\n";
    script << "set ylabel 'Pressure Error (Pa?)'\n";
    script << "set yrange [*:*]\n";
    script << "set yrange [-0.12:0.12]\n";  // DEBUG
    script << "plot";
    // Error: full fine solution minus analytic
    script << " \\\n  '" << filename << "' using 1:(($" << ( finest_lev + 3 )
           << "-$" << ( finest_lev + 4 )
           << ")) title 'Dense - Analytic' with linespoints";
    // Error: finest level minus analytic
    script << " \\\n, '" << filename << "' using 1:(($" << ( finest_lev + 2 )
           << "-$" << ( finest_lev + 4 )
           << ")) title 'Fine - Analytic' with linespoints";
    // Error: finest level minus full fine
    script << " \\\n, '" << filename << "' using 1:(($" << ( finest_lev + 2 )
           << "-$" << ( finest_lev + 3 )
           << ")) title 'Fine - Dense' with linespoints";
    // Error: raw level 0 minus analytic
    script << " \\\n, 'pressure_L0.csv' using 1:($2-$3)"
           << " title 'Raw L0 - Analytic' with linespoints pointsize 2";
    script << "\n";
    script.close();

    // Run gnuplot
    std::system( "gnuplot plot_pressure.gp" );
  }
}

void SingleLevelPressureSolve(  //
  amrex::MultiFab& pressure,
  const std::array<amrex::MultiFab, 3>& velocity,
  const amrex::Geometry& geom,
  amrex::Real tolerance,
  int max_iterations,
  int base_n,
  int nLevels )
{
  BL_PROFILE( "SingleLevelPressureSolve" );

  // Allocate and compute divergence
  amrex::MultiFab divergence( pressure.boxArray(),
                              pressure.DistributionMap(),
                              1,
                              0 );
  ComputeDivergence( divergence, velocity, geom );

  // Initialize pressure to zero
  pressure.setVal( 0.0 );

  // Solve for pressure using iterations
  constexpr bool use_expected_BCs = true;
  SolvePressureIterations(  //
    use_expected_BCs,
    pressure,
    divergence,
    geom,
    tolerance,
    max_iterations,
    base_n,
    nLevels );
}

void CheckResults(  //
  const amrex::MultiFab& pressure,
  const amrex::Geometry& geom,
  int base_n,
  int nLevels )
{
  amrex::Real max_error = 0.0;
  amrex::Real avg_error = 0.0;
  amrex::Real volume = 0.0;

  for ( amrex::MFIter mfi( pressure ); mfi.isValid(); ++mfi ) {
    const amrex::Box& box = mfi.validbox();
    const auto& p_arr = pressure.array( mfi );

    const auto lo = amrex::lbound( box );
    const auto hi = amrex::ubound( box );

    for ( int i = lo.x; i <= hi.x; ++i ) {
      for ( int j = lo.y; j <= hi.y; ++j ) {
        for ( int k = lo.z; k <= hi.z; ++k ) {
          const amrex::Real expected = ExpectedPressure(  //
            geom,
            i,
            j,
            k,
            base_n,
            nLevels );
          const amrex::Real error = std::abs( p_arr( i, j, k ) - expected );

          max_error = std::max( max_error, error );
          avg_error += error;
          volume += 1.0;
        }
      }
    }
  }

  // Sum across processors
  amrex::ParallelDescriptor::ReduceRealMax( max_error );
  amrex::ParallelDescriptor::ReduceRealSum( avg_error );
  amrex::ParallelDescriptor::ReduceRealSum( volume );

  avg_error /= volume;
  const amrex::Real error_tolerance = 1.0E-2;
  amrex::Print() << "Pressure solution verification:\n";
  amrex::Print() << "  Maximum error: " << max_error << "\n";
  amrex::Print() << "  Average error: " << avg_error << "\n";
  amrex::Print() << "  Error tolerance: " << error_tolerance << "\n";
  amrex::Print() << "  Average error relative to tolerance: "
                 << ( avg_error / error_tolerance ) << "\n";
}

void CompareMultiFabs(  //
  const LevelData& expected_level,
  const std::vector<LevelData>& level_data )
{
  const auto& expected_mf = expected_level.pressure;
  const auto& expected_geom = expected_level.geom;

  for ( int lev = 0; lev < level_data.size(); ++lev ) {
    const auto& level_mf = level_data[lev].pressure;
    const auto& level_geom = level_data[lev].geom;

    // Check real boxes match
    if ( !amrex::AlmostEqual( expected_geom.ProbDomain(),
                              level_geom.ProbDomain() ) ) {
      amrex::Print() << "\nReal box mismatch at level " << lev << ":\n"
                     << "  Expected: " << expected_geom.ProbDomain() << "\n"
                     << "  Level:    " << level_geom.ProbDomain() << "\n";
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(  //
      amrex::AlmostEqual( expected_geom.ProbDomain(), level_geom.ProbDomain() ),
      "Real boxes must match between expected and level geometries" );

    amrex::Print() << "\nComparing pressure at level " << lev << ":\n";

    // Create a copy of expected_mf with the same distribution mapping as
    // level_mf
    amrex::MultiFab expected_remapped( level_mf.boxArray(),
                                       level_mf.DistributionMap(),
                                       1,
                                       0 );

    if ( expected_geom.Domain() == level_geom.Domain() ) {
      // TODO: Can this also use average_down()?
      expected_remapped.ParallelCopy( expected_mf );
    } else {
      // Calculate refinement ratio between expected and level geometries
      amrex::IntVect ratio = expected_geom.Domain().size() /
                             level_geom.Domain().size();

      // Assert that expected_geom is a refined version of level_geom
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(  //
        ratio[0] >= 1 && ratio[1] >= 1 && ratio[2] >= 1,
        "Expected geometry must be refined version of level geometry" );
      AMREX_ALWAYS_ASSERT_WITH_MESSAGE(  //
        ratio[0] == ratio[1] && ratio[1] == ratio[2],
        "Refinement ratio must be uniform in all dimensions" );

      // Average down (restrict) from fine to coarse grid
      amrex::average_down( expected_mf, expected_remapped, 0, 1, ratio );

      amrex::Print()  //
        << "\nRestricted expected data from fine grid (ratio=" << ratio[0]
        << ") to coarse grid for level " << lev << "\n";
    }

    // Compute max absolute difference and L2 norm of difference
    amrex::Real max_diff = 0.0;
    amrex::Real l2_diff = 0.0;
    amrex::Real volume = 0.0;  // Add volume for proper L2 norm calculation

    for ( amrex::MFIter mfi( level_mf ); mfi.isValid(); ++mfi ) {
      const amrex::Box& bx = mfi.validbox();
      const auto& expected_fab = expected_remapped[mfi];
      const auto& level_fab = level_mf[mfi];

      for ( int i = bx.loVect()[0]; i <= bx.hiVect()[0]; ++i ) {
        for ( int j = bx.loVect()[1]; j <= bx.hiVect()[1]; ++j ) {
          for ( int k = bx.loVect()[2]; k <= bx.hiVect()[2]; ++k ) {
            amrex::Real diff = std::abs(
              expected_fab( amrex::IntVect( i, j, k ) ) -
              level_fab( amrex::IntVect( i, j, k ) ) );
            max_diff = std::max( max_diff, diff );
            l2_diff += diff * diff;
            volume += 1.0;
          }
        }
      }
    }

    // Reduce across all processes
    amrex::ParallelDescriptor::ReduceRealMax( max_diff );
    amrex::ParallelDescriptor::ReduceRealSum( l2_diff );
    amrex::ParallelDescriptor::ReduceRealSum( volume );

    // Compute normalized L2 norm
    l2_diff = std::sqrt( l2_diff / volume );

    amrex::Print() << "  Maximum absolute difference: " << max_diff << "\n";
    amrex::Print() << "  L2 norm of difference: " << l2_diff << "\n";
    amrex::Print() << "  Cell volume: " << volume << "\n";

    // Check if differences exceed tolerance
    constexpr amrex::Real diff_tolerance = 1.0e-3;
    if ( max_diff > diff_tolerance ) {
      amrex::Print() << "\nWARNING: Maximum difference exceeds tolerance ("
                     << diff_tolerance << ")!\n";
    }
    constexpr amrex::Real l2_tolerance = 1.0e-3;
    if ( l2_diff > l2_tolerance ) {
      amrex::Print() << "\nWARNING: L2 norm of difference exceeds tolerance ("
                     << l2_tolerance << ")!\n";
    }
  }
}

/*--------------------------------------------------------------------
  private free function definitions
  --------------------------------------------------------------------*/
static void ComputeDivergence(  //
  amrex::MultiFab& divergence,
  const std::array<amrex::MultiFab, 3>& velocity,
  const amrex::Geometry& geom )
{
  BL_PROFILE( "ComputeDivergence" );

  const amrex::Real dx = geom.CellSize( 0 );
  const amrex::Real dxinv = 1.0 / dx;

  // Track maximum divergence for diagnostics
  amrex::Real max_div = 0.0;

  for ( amrex::MFIter mfi( divergence ); mfi.isValid(); ++mfi ) {
    const amrex::Box& box = mfi.validbox();
    const auto& div_arr = divergence.array( mfi );
    const auto& u_arr = velocity[U].array( mfi );
    const auto& v_arr = velocity[V].array( mfi );
    const auto& w_arr = velocity[W].array( mfi );

    const auto lo = amrex::lbound( box );
    const auto hi = amrex::ubound( box );

    for ( int i = lo.x; i <= hi.x; ++i ) {
      for ( int j = lo.y; j <= hi.y; ++j ) {
        for ( int k = lo.z; k <= hi.z; ++k ) {
          // Compute divergence using central differences
          div_arr( i, j, k ) =                                   //
            dxinv * ( u_arr( i + 1, j, k ) - u_arr( i, j, k ) +  //
                      v_arr( i, j + 1, k ) - v_arr( i, j, k ) +  //
                      w_arr( i, j, k + 1 ) - w_arr( i, j, k ) );

          max_div = std::max( max_div, std::abs( div_arr( i, j, k ) ) );
        }
      }
    }
  }

  // Print maximum divergence for diagnostics
  amrex::ParallelDescriptor::ReduceRealMax( max_div );
  amrex::Print() << "Maximum divergence: " << max_div << "\n";
}

static void GaussSeidelIteration(  //
  bool use_expected_BCs,
  amrex::MultiFab& pressure,
  const amrex::MultiFab& divergence,
  const amrex::Geometry& geom,
  int iteration,
  int base_n,
  int nLevels )
{
  const amrex::Real dx = geom.CellSize( 0 );
  const amrex::Real dx2 = dx * dx;

  if ( use_expected_BCs ) {
    // Define boundary conditions
    amrex::Vector<amrex::BCRec> bc( 1 );

    // Set Dirichlet boundary conditions based on analytic solution
    for ( int n = 0; n < 1; ++n ) {
      for ( int dir = 0; dir < AMREX_SPACEDIM; ++dir ) {
        bc[n].setLo( dir, amrex::BCType::ext_dir );  // External Dirichlet
        bc[n].setHi( dir, amrex::BCType::ext_dir );  // External Dirichlet
      }
    }

    // Create boundary condition functor
    PressureBndryFunc pbf( base_n, nLevels );
    amrex::PhysBCFunct<PressureBndryFunc> physbc( geom, bc, pbf );

    // Fill ghost cells with boundary conditions
    const int start_comp = 0;          // Starting component
    const int num_comp = 1;            // Number of components
    const amrex::IntVect nghost( 1 );  // Ghost cell width
    const amrex::Real time = 0.0;      // Time
    const int bccomp = 0;  // Starting component for boundary conditions
    physbc.FillBoundary( pressure, start_comp, num_comp, nghost, time, bccomp );
  }

  // Now do standard Gauss-Seidel iteration for all cells in the domain
  for ( amrex::MFIter mfi( pressure ); mfi.isValid(); ++mfi ) {
    const amrex::Box& box = mfi.validbox();
    const auto& p_arr = pressure.array( mfi );
    const auto& div_arr = divergence.array( mfi );

    const auto lo = amrex::lbound( box );
    const auto hi = amrex::ubound( box );

    for ( int i = lo.x; i <= hi.x; ++i ) {
      for ( int j = lo.y; j <= hi.y; ++j ) {
        for ( int k = lo.z; k <= hi.z; ++k ) {
          // Gauss-Seidel update with under-relaxation
          const amrex::Real p_new =   //
            ( 1.0 / 6.0 ) *           //
            ( p_arr( i + 1, j, k ) +  //
              p_arr( i - 1, j, k ) +  //
              p_arr( i, j + 1, k ) +  //
              p_arr( i, j - 1, k ) +  //
              p_arr( i, j, k + 1 ) +  //
              p_arr( i, j, k - 1 ) -  //
              dx2 * div_arr( i, j, k ) );

          p_arr( i, j, k ) = p_new;
        }
      }
    }
  }

  // Fill internal ghost cells between patches
  pressure.FillBoundary( geom.periodicity() );
}

static amrex::Real ComputeResidual(  //
  const amrex::MultiFab& pressure,
  const amrex::MultiFab& divergence,
  const amrex::Geometry& geom )
{
  const amrex::Real dx = geom.CellSize( 0 );
  const amrex::Real dx2 = dx * dx;

  amrex::Real residual = 0.0;
  amrex::Real volume = 0.0;

  for ( amrex::MFIter mfi( pressure ); mfi.isValid(); ++mfi ) {
    const amrex::Box& box = mfi.validbox();
    const auto& p_arr = pressure.array( mfi );
    const auto& div_arr = divergence.array( mfi );

    const auto lo = amrex::lbound( box );
    const auto hi = amrex::ubound( box );

    for ( int i = lo.x; i <= hi.x; ++i ) {
      for ( int j = lo.y; j <= hi.y; ++j ) {
        for ( int k = lo.z; k <= hi.z; ++k ) {
          // Compute residual using central differences
          const amrex::Real lap_p =   //
            ( p_arr( i + 1, j, k ) +  //
              p_arr( i - 1, j, k ) +  //
              p_arr( i, j + 1, k ) +  //
              p_arr( i, j - 1, k ) +  //
              p_arr( i, j, k + 1 ) +  //
              p_arr( i, j, k - 1 ) -  //
              6.0 * p_arr( i, j, k ) ) /
            dx2;

          const amrex::Real res = lap_p - div_arr( i, j, k );
          residual += res * res;
          volume += 1.0;
        }
      }
    }
  }

  // Sum across processors
  amrex::ParallelDescriptor::ReduceRealSum( residual );
  amrex::ParallelDescriptor::ReduceRealSum( volume );

  return std::sqrt( residual / volume );
}

static void SolvePressureIterations(  //
  bool use_expected_BCs,
  amrex::MultiFab& pressure,
  const amrex::MultiFab& divergence,
  const amrex::Geometry& geom,
  amrex::Real tolerance,
  int max_iterations,
  int base_n,
  int nLevels )
{
  BL_PROFILE( "SolvePressureIterations" );

  amrex::Real residual = 1.0;
  int iteration = 0;

  while ( residual > tolerance && iteration < max_iterations ) {
    GaussSeidelIteration(  //
      use_expected_BCs,
      pressure,
      divergence,
      geom,
      iteration,
      base_n,
      nLevels );
    residual = ComputeResidual( pressure, divergence, geom );
    iteration++;

    if ( ( max_iterations < 100 ) || ( iteration % 10 == 0 ) ) {
      amrex::Print()  //
        << "Iteration " << iteration << ", residual = " << residual << "\n";
    }
  }

  amrex::Print()  //
    << "Final iteration " << iteration << ", residual = " << residual << "\n";
}

LevelData::LevelData( int base_n, int level, amrex::Real domain_length )
  : geom( DefineGeometry( base_n, level, domain_length ) )
{
}

LevelData MakeDenseLevelData(  //
  amrex::Real domain_length,
  int base_n,
  int nLevels,
  int level )
{
  LevelData level_data( base_n, level, domain_length );
  const amrex::BoxArray ba = DefineBoxArray( base_n, level );
  const amrex::DistributionMapping dm = DefineDM( ba );
  DefineFABs( level_data.pressure, level_data.velocity, ba, dm );
  InitializeVelocity(  //
    level_data.velocity,
    level_data.geom,
    base_n,
    nLevels,
    level );
  return level_data;
}

LevelData MakeSparseLevelData(  //
  amrex::Real domain_length,
  int base_n,
  int nLevels,
  int level )
{
  LevelData level_data( base_n, level, domain_length );
  const amrex::BoxArray ba = DefineSparseBoxArray( base_n, level );
  const amrex::DistributionMapping dm = DefineDM( ba );
  DefineFABs( level_data.pressure, level_data.velocity, ba, dm );
  InitializeVelocity(  //
    level_data.velocity,
    level_data.geom,
    base_n,
    nLevels,
    level );
  return level_data;
}

std::vector<LevelData> MakeSparseCompositeLevels(  //
  int base_n,
  int nLevels,
  amrex::Real domain_length )
{
  std::vector<LevelData> composite_levels;
  for ( int lev = 0; lev < nLevels; ++lev ) {
    if ( lev == 0 ) {
      composite_levels.push_back(
        MakeDenseLevelData( domain_length, base_n, nLevels, lev ) );
    } else {
      composite_levels.push_back(
        MakeSparseLevelData( domain_length, base_n, nLevels, lev ) );
    }
  }
  return composite_levels;
}

void FillPressureGhostCells(  //
  LevelData& fine_level,
  const LevelData& crse_level )
{
  // Unused time (passed to PhysBCFunc and otherwise not used?)
  constexpr double time_unused = 0.0;

  // Source, destination, and BC components
  constexpr int scomp = 0;
  constexpr int dcomp = 0;
  constexpr int ncomp = 1;
  constexpr int cbccomp = 0;
  constexpr int fbccomp = 0;
  constexpr int bcscomp = 0;

  // Refinement ratio, per dimension
  const amrex::IntVect ref_ratio( 2 );

  // Domain boundary conditions, per-component
  constexpr auto re = amrex::BCType::reflect_even;
  const amrex::BCRec bcrec( re, re, re, re, re, re );
  const amrex::Vector<amrex::BCRec> bcrecs{ bcrec };

  // Set up BC functors to handle external and coarse/fine ghosts.
  assert(
    !amrex::Gpu::inLaunchRegion() );  // Use GpuBndryFuncFab for GPU support?
  amrex::CpuBndryFuncFab null_bndry_func = nullptr;
  amrex::PhysBCFunct<amrex::CpuBndryFuncFab> cphysbc(  //
    crse_level.geom,
    bcrecs,
    null_bndry_func );
  amrex::PhysBCFunct<amrex::CpuBndryFuncFab> fphysbc(  //
    fine_level.geom,
    bcrecs,
    null_bndry_func );

  // Fill just the coarse/fine ghosts on the fine MultiFab using interpolated
  // values from the coarse MultiFab.
  // TODO: Does this also fill interior or exterior fine ghosts?
  amrex::FillPatchTwoLevels(  //
    fine_level.pressure,      // <- Destination
    time_unused,
    { const_cast<amrex::MultiFab*>( &crse_level.pressure ) },  // TODO: const?
    { time_unused },
    { &fine_level.pressure },  // <- Source, fine at multiple times
    { time_unused },
    scomp,
    dcomp,
    ncomp,
    crse_level.geom,
    fine_level.geom,
    cphysbc,
    cbccomp,
    fphysbc,
    fbccomp,
    ref_ratio,
    // Interpolator makes no difference here?
    // &amrex::pc_interp,
    // &amrex::cell_bilinear_interp,
    &amrex::quadratic_interp,
    bcrecs,
    bcscomp );
}

void SolvePressureCorrection(  //
  amrex::MultiFab& crse_pressure,
  const amrex::MultiFab& fine_pressure,
  const amrex::Geometry& crse_geom,
  const amrex::Geometry& fine_geom,
  amrex::Real tolerance,
  int max_iterations,
  int base_n,
  int nLevels )
{
  BL_PROFILE( "SolvePressureCorrection" );

  // Calculate refinement ratio
  amrex::IntVect ratio = fine_geom.Domain().size() / crse_geom.Domain().size();
  AMREX_ALWAYS_ASSERT( ratio[0] == ratio[1] &&
                       ratio[1] == ratio[2] );  // Uniform refinement

  const int ignoredFineLevel = -1;
  // Create a FluxRegister to handle flux mismatches
  amrex::FluxRegister flux_reg(  //
    fine_pressure.boxArray(),
    fine_pressure.DistributionMap(),
    ratio,
    ignoredFineLevel,
    1 );

  // Create temporary MultiFabs to store fluxes
  amrex::MultiFab crse_flux[AMREX_SPACEDIM];
  amrex::MultiFab fine_flux[AMREX_SPACEDIM];
  for ( int dir = 0; dir < AMREX_SPACEDIM; ++dir ) {
    // Create face-centered flux MultiFabs
    amrex::BoxArray ba_crse =
      amrex::convert( crse_pressure.boxArray(),
                      amrex::IntVect::TheDimensionVector( dir ) );
    amrex::BoxArray ba_fine =
      amrex::convert( fine_pressure.boxArray(),
                      amrex::IntVect::TheDimensionVector( dir ) );
    crse_flux[dir].define( ba_crse, crse_pressure.DistributionMap(), 1, 0 );
    fine_flux[dir].define( ba_fine, fine_pressure.DistributionMap(), 1, 0 );
  }

  // Compute fluxes on both levels
  // Flux = -Dp/dx
  // Face_Flux = -dp/dx * area = -dp/dx * (dx*dx) = -dp * dx

  // Compute coarse fluxes
  for ( amrex::MFIter mfi( crse_pressure ); mfi.isValid(); ++mfi ) {
    const amrex::Real scale = crse_geom.CellSize( 0 );
    const auto& pres_arr = crse_pressure.const_array( mfi );
    for ( int dir = 0; dir < AMREX_SPACEDIM; ++dir ) {
      const amrex::Box& bx = crse_flux[dir][mfi].box();
      const auto& flux_arr = crse_flux[dir].array( mfi );

      amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE( int i, int j, int k ) {
        // Compute flux using central differences
        if ( dir == 0 ) {
          flux_arr( i, j, k ) = scale * ( pres_arr( i, j, k ) -
                                          pres_arr( i - 1, j, k ) );
        } else if ( dir == 1 ) {
          flux_arr( i, j, k ) = scale * ( pres_arr( i, j, k ) -
                                          pres_arr( i, j - 1, k ) );
        } else {
          flux_arr( i, j, k ) = scale * ( pres_arr( i, j, k ) -
                                          pres_arr( i, j, k - 1 ) );
        }
      } );
    }
  }

  // Compute fine fluxes
  for ( amrex::MFIter mfi( fine_pressure ); mfi.isValid(); ++mfi ) {
    const amrex::Real scale = fine_geom.CellSize( 0 );
    const auto& pres_arr = fine_pressure.const_array( mfi );
    for ( int dir = 0; dir < AMREX_SPACEDIM; ++dir ) {
      const amrex::Box& bx = fine_flux[dir][mfi].box();
      const auto& flux_arr = fine_flux[dir].array( mfi );

      amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE( int i, int j, int k ) {
        // Compute flux using central differences
        if ( dir == 0 ) {
          flux_arr( i, j, k ) = scale * ( pres_arr( i, j, k ) -
                                          pres_arr( i - 1, j, k ) );
        } else if ( dir == 1 ) {
          flux_arr( i, j, k ) = scale * ( pres_arr( i, j, k ) -
                                          pres_arr( i, j - 1, k ) );
        } else {
          flux_arr( i, j, k ) = scale * ( pres_arr( i, j, k ) -
                                          pres_arr( i, j, k - 1 ) );
        }
      } );
    }
  }

  // Add fluxes to the register
  for ( int dir = 0; dir < AMREX_SPACEDIM; ++dir ) {
    flux_reg.CrseInit( crse_flux[dir], dir, 0, 0, 1, -1.0 );
    flux_reg.FineAdd( fine_flux[dir], dir, 0, 0, 1, 1.0 );
  }

  // Create correction MultiFab and divergence MultiFab
  amrex::MultiFab correction_solution(  //
    crse_pressure.boxArray(),
    crse_pressure.DistributionMap(),
    1,    // ncomp
    1 );  // nghost
  correction_solution.setVal( 0.0 );
  amrex::MultiFab correction_div(  //
    crse_pressure.boxArray(),
    crse_pressure.DistributionMap(),
    1,    // ncomp
    0 );  // nghost
  correction_div.setVal( 0.0 );

#if 0
  // Initialize ghosts????? Should not be used.
  correction_solution.ParallelCopy(  //
    crse_pressure,                   // source
    0,                               // source component
    0,                               // dest component
    1,                               // num comp
    crse_pressure.nGrow(),           // source nghost
    correction_solution.nGrow(),     // dest nghost
    amrex::Periodicity::NonPeriodic() );
#endif

  flux_reg.Reflux( correction_div, 1.0, 0, 0, 1, crse_geom );

  // Solve for the correction using the divergence as the right-hand side
  constexpr bool use_expected_BCs = false;
  SolvePressureIterations(  //
    use_expected_BCs,
    correction_solution,
    correction_div,
    crse_geom,
    tolerance,
    max_iterations,
    base_n,
    nLevels );

  // Add correction to coarse pressure
  amrex::MultiFab::Add( crse_pressure, correction_solution, 0, 0, 1, 0 );

  // Print statistics about the correction
  amrex::Real max_correction = 0.0;
  amrex::Real avg_correction = 0.0;
  amrex::Real volume = 0.0;

  for ( amrex::MFIter mfi( correction_solution ); mfi.isValid(); ++mfi ) {
    const amrex::Box& box = mfi.validbox();
    const auto& corr_arr = correction_solution.array( mfi );

    const auto lo = amrex::lbound( box );
    const auto hi = amrex::ubound( box );

    for ( int i = lo.x; i <= hi.x; ++i ) {
      for ( int j = lo.y; j <= hi.y; ++j ) {
        for ( int k = lo.z; k <= hi.z; ++k ) {
          const amrex::Real corr = std::abs( corr_arr( i, j, k ) );
          max_correction = std::max( max_correction, corr );
          avg_correction += corr;
          volume += 1.0;
        }
      }
    }
  }

  // Reduce across processors
  amrex::ParallelDescriptor::ReduceRealMax( max_correction );
  amrex::ParallelDescriptor::ReduceRealSum( avg_correction );
  amrex::ParallelDescriptor::ReduceRealSum( volume );

  avg_correction /= volume;

  amrex::Print()  //
    << "\nPressure correction statistics:\n"
    << "  Maximum correction: " << max_correction << "\n"
    << "  Average correction: " << avg_correction << "\n"
    << "  Number of cells: " << volume << "\n";
}

constexpr int CalculateFineN( int base_n, int nLevels )
{
  return CalculateNForLevel( base_n, nLevels - 1 );
}
constexpr int CalculateNForLevel( int base_n, int level )
{
  return base_n * ( 1 << level );
}

// Compute the potential at r due to a uniform source cube of size dx centered
// at ri This is (1/4π) * (1/dx³) * ∫∫∫_source_cube 1/|r-r'| dV' Note: r and ri
// are always at cell centers
static amrex::Real Phi(  //
  amrex::Real r_x,
  amrex::Real r_y,
  amrex::Real r_z,
  amrex::Real ri_x,
  amrex::Real ri_y,
  amrex::Real ri_z,
  amrex::Real dx )
{
  constexpr bool point_wise = false;
  {
    // Compute distance from r to center of source cube
    const amrex::Real dr_x = r_x - ri_x;
    const amrex::Real dr_y = r_y - ri_y;
    const amrex::Real dr_z = r_z - ri_z;
    const amrex::Real r2 = dr_x * dr_x + dr_y * dr_y + dr_z * dr_z;

    // If r is at the same cell center as ri, use analytic solution
    if ( r2 < 1.0e-8 ) {  // Effectively zero distance
      // Analytic solution for self-potential at the center of the cube
      return 2.0 * 1.516386 / ( 4.0 * M_PI * dx );
    } else if ( point_wise ) {
      return 1.0 / ( 4.0 * M_PI * std::sqrt( r2 ) );
    }
  }

  // For all other cell centers, use numerical integration
  amrex::Real sum = 0.0;
  const amrex::Real half_dx = 0.5 * dx;
  constexpr amrex::Real sqrt3over5 = 0.7745966692414834;
  // Use a 3x3x3 quadrature rule (27 points) for better accuracy
  for ( int i = -1; i <= 1; i++ ) {
    for ( int j = -1; j <= 1; j++ ) {
      for ( int k = -1; k <= 1; k++ ) {
        // √(3/5)
        const amrex::Real x = ri_x + i * half_dx * sqrt3over5;
        const amrex::Real y = ri_y + j * half_dx * sqrt3over5;
        const amrex::Real z = ri_z + k * half_dx * sqrt3over5;

        const amrex::Real dr_x = r_x - x;
        const amrex::Real dr_y = r_y - y;
        const amrex::Real dr_z = r_z - z;
        const amrex::Real r = std::sqrt( dr_x * dr_x + dr_y * dr_y +
                                         dr_z * dr_z );

        // Use Gauss-Legendre weights
        const amrex::Real w = ( i == 0 ? 0.8888888888888888
                                       : 0.5555555555555556 ) *
                              ( j == 0 ? 0.8888888888888888
                                       : 0.5555555555555556 ) *
                              ( k == 0 ? 0.8888888888888888
                                       : 0.5555555555555556 );

        sum += w / r;
      }
    }
  }

  // The quadrature weights are normalized to sum to 8 (the volume of the
  // cube) So we just need to multiply by 1/4π
  return ( 1.0 / ( 4.0 * M_PI ) ) * ( sum / 8.0 );
}
/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/
