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
static amrex::Real ExpectedPressure(  //
  const amrex::Geometry& geom,
  int i,
  int j,
  int k,
  int fineN )
{
  // Analytic solution for a dipole in infinite domain:
  // p(r) = -1/(4*pi) * (1/|r - r1| - 1/|r - r2|)
  const amrex::Real x = geom.CellCenter( i, U );
  const amrex::Real y = geom.CellCenter( j, V );
  const amrex::Real z = geom.CellCenter( k, W );

  // The nonzero x-face is at (coarse_face[0], coarse_face[1], coarse_face[2]),
  // between cells (coarse_face[0]-1, coarse_face[1], coarse_face[2]) and
  // (coarse_face[0], coarse_face[1], coarse_face[2])
  // Cell center of cell to the left of the face
  const amrex::Real centerMinus = ( ( fineN / 2.0 ) - 0.5 ) / fineN;
  // Cell center of cell to the right of the face
  const amrex::Real centerPlus = ( ( fineN / 2.0 ) + 0.5 ) / fineN;

  const amrex::Real r1_x = centerMinus;  // cell center to the left
  const amrex::Real r2_x = centerPlus;   // cell center to the right
  const amrex::Real r1_y = centerPlus;   // face center
  const amrex::Real r2_y = centerPlus;   // face center
  const amrex::Real r1_z = centerPlus;   // face center
  const amrex::Real r2_z = centerPlus;   // face center

  // Use cell-averaged inverse distance for source cells, pointwise otherwise
  const amrex::Real h = 1.0 / fineN;
  const amrex::Real avg_inv_r = 2.0 * 1.516386 / h;  // <1/r> over the cube

  const amrex::Real dist1 =                   //
    std::sqrt( ( x - r1_x ) * ( x - r1_x ) +  //
               ( y - r1_y ) * ( y - r1_y ) +  //
               ( z - r1_z ) * ( z - r1_z ) );
  const amrex::Real dist2 =                   //
    std::sqrt( ( x - r2_x ) * ( x - r2_x ) +  //
               ( y - r2_y ) * ( y - r2_y ) +  //
               ( z - r2_z ) * ( z - r2_z ) );

  // Use cell-averaged inverse distance if the evaluation point coincides with
  // the source, otherwise use pointwise inverse distance
  const amrex::Real tol = 1e-10;
  const amrex::Real inv_dist1 = ( dist1 < tol ) ? avg_inv_r : 1.0 / dist1;
  const amrex::Real inv_dist2 = ( dist2 < tol ) ? avg_inv_r : 1.0 / dist2;
  const amrex::Real p = -1.0 / ( 4.0 * M_PI ) * ( inv_dist1 - inv_dist2 );
  return p;
}

class PressureBndryFunc
{
public:
  PressureBndryFunc( int fineN ) : vFineN( fineN ) {}

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
          arr( i, j, k, n + dcomp ) = ExpectedPressure( geom, i, j, k, vFineN );
        }
      } );
  }

private:
  int vFineN = 0;
};

static void GaussSeidelIteration(  //
  amrex::MultiFab& pressure,
  const amrex::MultiFab& divergence,
  const amrex::Geometry& geom,
  int iteration,
  int fineN );

static amrex::Real ComputeResidual(  //
  const amrex::MultiFab& pressure,
  const amrex::MultiFab& divergence,
  const amrex::Geometry& geom );

static void ComputeDivergence(  //
  amrex::MultiFab& divergence,
  const std::array<amrex::MultiFab, 3>& velocity,
  const amrex::Geometry& geom );

static void SolvePressureIterations(  //
  amrex::MultiFab& pressure,
  const amrex::MultiFab& divergence,
  const amrex::Geometry& geom,
  amrex::Real tolerance,
  int max_iterations,
  int fineN );

/*--------------------------------------------------------------------
  public free function definitions
  --------------------------------------------------------------------*/
amrex::Geometry DefineGeometry( int nx, int ny, int nz, double dx )
{
  const amrex::RealBox real_box( { 0.0, 0.0, 0.0 },
                                 { nx * dx, ny * dx, nz * dx } );
  constexpr amrex::CoordSys::CoordType coord =
    amrex::CoordSys::CoordType::cartesian;
  const amrex::IntArray is_periodic{ 0, 0, 0 };  // Non-periodic

  const amrex::Box domain( amrex::IntVect( 0, 0, 0 ),
                           amrex::IntVect( nx - 1, ny - 1, nz - 1 ) );
  return amrex::Geometry( domain, real_box, coord, is_periodic );
}

amrex::BoxArray DefineBoxArray( int n )
{
  const amrex::Box domainBox( amrex::IntVect( 0, 0, 0 ),
                              amrex::IntVect( n - 1, n - 1, n - 1 ) );
  return amrex::BoxArray( domainBox );
}

amrex::BoxArray DefineSparseBoxArray( int n )
{
  const int lo = ( n / 2 ) - 2;
  const int hi = ( n / 2 ) + 1;
  const amrex::Box sparseBox( amrex::IntVect( lo, lo, lo ),
                              amrex::IntVect( hi, hi, hi ) );
  return amrex::BoxArray( sparseBox );
}

amrex::DistributionMapping DefineDM( const amrex::BoxArray& ba )
{
  return amrex::DistributionMapping( ba );
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

void InitializeVelocity(  //
  std::array<amrex::MultiFab, 3>& velocity,
  const amrex::Geometry& geom,
  int fineN )
{
  // Set all velocities to zero
  for ( int d = 0; d < 3; ++d ) {
    velocity[d].setVal( 0.0 );
  }

  const int halfN = fineN / 2;
  if ( geom.Domain().length( 0 ) == fineN ) {
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
  int fineN )
{
  const int finest_lev = static_cast<int>( level_data.size() ) - 1;
  const amrex::Geometry& fine_geom = level_data[finest_lev].geom;

  // Get centerline indices for y and z at finest level
  const amrex::Real center_y = 0.5 * ( fine_geom.ProbLo( 1 ) +
                                       fine_geom.ProbHi( 1 ) );
  const amrex::Real center_z = 0.5 * ( fine_geom.ProbLo( 2 ) +
                                       fine_geom.ProbHi( 2 ) );

  // Create MultiFabs to store interpolated pressure at each level
  amrex::Vector<amrex::MultiFab> interpolated_pressure( level_data.size() );
  for ( int lev = 0; lev <= finest_lev; ++lev ) {
    // Define MultiFab with same structure as full fine pressure.
    interpolated_pressure[lev].define(  //
      fullFineSolution.pressure.boxArray(),
      fullFineSolution.pressure.DistributionMap(),
      fullFineSolution.pressure.nComp(),
      fullFineSolution.pressure.nGrow() );
    interpolated_pressure.at( lev ).setVal( 0.0 );
  }

  // Set up boundary conditions for pressure
  amrex::Vector<amrex::BCRec> bcs( 1 );  // One component for pressure
  for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim ) {
    bcs.at( 0 ).setLo( idim, amrex::BCType::ext_dir );
    bcs.at( 0 ).setHi( idim, amrex::BCType::ext_dir );
  }

  // Create boundary condition functors for each level
  amrex::Vector<amrex::PhysBCFunct<PressureBndryFunc>> physbcs;
  for ( int lev = 0; lev <= finest_lev; ++lev ) {
    physbcs.emplace_back(  //
      level_data[lev].geom,
      bcs,
      PressureBndryFunc( fineN ) );
  }

  // Prepare data for FillPatchNLevels
  amrex::Vector<amrex::Vector<amrex::MultiFab*>> smf( level_data.size() );
  amrex::Vector<amrex::Vector<amrex::Real>> st( level_data.size() );
  amrex::Vector<amrex::Geometry> geom( level_data.size() );
  amrex::Vector<amrex::IntVect> ratio( level_data.size() );

  for ( int lev = 0; lev <= finest_lev; ++lev ) {
    smf.at( lev ) =  //
      { const_cast<amrex::MultiFab*>( &level_data[lev].pressure ) };
    st.at( lev ) = { 0.0 };
    geom.at( lev ) = level_data[lev].geom;
#if 1
    if ( lev < finest_lev ) {
      ratio.at( lev ) = level_data[lev + 1].geom.Domain().size() /
                        level_data[lev].geom.Domain().size();
    } else {
      // TODO: How do ratios work?
      ratio.at( lev ) = amrex::IntVect::TheUnitVector();
    }
#else
    // TODO: How do ratios work?
    ratio.at( lev ) = amrex::IntVect( 2, 2, 2 );
#endif
  }

  // Fill all levels with proper interpolation
  for ( int lev = 0; lev <= finest_lev; ++lev ) {
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
      lev,                           // level
      outMF.nGrowVect(),             // nghost
      time,                          // time
      smf,                           // source MultiFabs
      st,                            // source times
      scomp,                         // source component
      dcomp,                         // destination component
      ncomp,                         // number of components
      geom,                          // geometries
      physbcs,                       // boundary conditions
      bccomp,                        // boundary condition component
      ratio,                         // refinement ratios
      &amrex::cell_bilinear_interp,  // interpolation operator
      bcr,                           // boundary conditions
      bcrcomp );                     // boundary condition component
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
    const int j = static_cast<int>( ( center_y - fine_geom.ProbLo( 1 ) ) /
                                    fine_geom.CellSize( 1 ) );
    const int k = static_cast<int>( ( center_z - fine_geom.ProbLo( 2 ) ) /
                                    fine_geom.CellSize( 2 ) );

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
        }
        assert( found );
        outfile << "," << val;
      }

      // Sample from full fine solution
      amrex::Real full_fine_val = 0.0;
      bool found = false;
      for ( amrex::MFIter mfi( fullFineSolution.pressure ); mfi.isValid();
            ++mfi ) {
        const amrex::Box& box = mfi.validbox();
        if ( box.contains( amrex::IntVect( i, j, k ) ) ) {
          const auto& parr = fullFineSolution.pressure.array( mfi );
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
      amrex::Real expected = ExpectedPressure( fine_geom, i, j, k, fineN );
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
    script << "set yrange [-1:1]\n";  // DEBUG
    script << "plot '" << filename << "' using 1:" << ( finest_lev + 4 )
           << " title 'Analytic' with lines linetype -1 linewidth 3";
    for ( int lev = 0; lev <= finest_lev; ++lev ) {
      script << ", '" << filename << "' using 1:" << ( lev + 2 )
             << " title 'Level " << lev << "'"
             << " with linespoints linewidth 2 pointsize 2";
    }
    script << ", '" << filename << "' using 1:" << ( finest_lev + 3 )
           << " title 'Full Fine' with linespoints linewidth 2 pointsize 2";
    script << "\n";
    script << "set output 'pressure_error.png'\n";
    script << "set xlabel 'x (m)'\n";
    script << "set ylabel 'Pressure Error (unitless)'\n";
    script << "set yrange [*:*]\n";
    // Error: finest level minus analytic
    script << "plot '" << filename << "' using 1:(($" << ( finest_lev + 2 )
           << "-$" << ( finest_lev + 4 ) << ")/$" << ( finest_lev + 4 )
           << ") title '(Finest - Analytic)/Analytic' with linespoints";
    // Add error for full fine solution
    script << ", '" << filename << "' using 1:(($" << ( finest_lev + 3 ) << "-$"
           << ( finest_lev + 4 ) << ")/$" << ( finest_lev + 4 )
           << ") title '(Full Fine - Analytic)/Analytic' with linespoints\n";
    script.close();

    // Run gnuplot
    std::system( "gnuplot plot_pressure.gp" );
  }
}

void SolvePressure(  //
  amrex::MultiFab& pressure,
  const std::array<amrex::MultiFab, 3>& velocity,
  const amrex::Geometry& geom,
  amrex::Real tolerance,
  int max_iterations,
  int fineN )
{
  // Allocate and compute divergence
  amrex::MultiFab divergence( pressure.boxArray(),
                              pressure.DistributionMap(),
                              1,
                              0 );
  ComputeDivergence( divergence, velocity, geom );

  // Initialize pressure to zero
  pressure.setVal( 0.0 );

  // Solve for pressure using iterations
  SolvePressureIterations(  //
    pressure,
    divergence,
    geom,
    tolerance,
    max_iterations,
    fineN );
}

void CheckResults(  //
  const amrex::MultiFab& pressure,
  const amrex::Geometry& geom,
  int fineN )
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
            fineN );
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
  amrex::MultiFab& pressure,
  const amrex::MultiFab& divergence,
  const amrex::Geometry& geom,
  int iteration,
  int fineN )
{
  const amrex::Real dx = geom.CellSize( 0 );
  const amrex::Real dx2 = dx * dx;

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
  PressureBndryFunc pbf( fineN );
  amrex::PhysBCFunct<PressureBndryFunc> physbc( geom, bc, pbf );

  // Fill ghost cells with boundary conditions
  const int start_comp = 0;          // Starting component
  const int num_comp = 1;            // Number of components
  const amrex::IntVect nghost( 1 );  // Ghost cell width
  const amrex::Real time = 0.0;      // Time
  const int bccomp = 0;  // Starting component for boundary conditions
  physbc.FillBoundary( pressure, start_comp, num_comp, nghost, time, bccomp );

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
  amrex::MultiFab& pressure,
  const amrex::MultiFab& divergence,
  const amrex::Geometry& geom,
  amrex::Real tolerance,
  int max_iterations,
  int fineN )
{
  amrex::Real residual = 1.0;
  int iteration = 0;

  while ( residual > tolerance && iteration < max_iterations ) {
    GaussSeidelIteration(  //
      pressure,
      divergence,
      geom,
      iteration,
      fineN );
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

LevelData::LevelData( int n, amrex::Real domain_length )
  : geom( DefineGeometry( n, n, n, domain_length / n ) )
{
}

LevelData MakeDenseLevelData( int n, amrex::Real domain_length, int fineN )
{
  LevelData level_data( n, domain_length );
  const amrex::BoxArray ba = DefineBoxArray( n );
  const amrex::DistributionMapping dm = DefineDM( ba );
  DefineFABs( level_data.pressure, level_data.velocity, ba, dm );
  InitializeVelocity( level_data.velocity, level_data.geom, fineN );
  return level_data;
}

LevelData MakeSparseLevelData( int n, amrex::Real domain_length, int fineN )
{
  LevelData level_data( n, domain_length );
  const amrex::BoxArray ba = DefineSparseBoxArray( n );
  const amrex::DistributionMapping dm = DefineDM( ba );
  DefineFABs( level_data.pressure, level_data.velocity, ba, dm );
  InitializeVelocity( level_data.velocity, level_data.geom, fineN );
  return level_data;
}

std::vector<LevelData> MakeSparseCompositeLevels(  //
  int base_n,
  int nlevels,
  amrex::Real domain_length,
  int fineN )
{
  std::vector<LevelData> composite_levels;
  for ( int lev = 0; lev < nlevels; ++lev ) {
    int n = base_n * ( 1 << lev );
    if ( lev == 0 ) {
      composite_levels.push_back(
        MakeDenseLevelData( n, domain_length, fineN ) );
    } else {
      composite_levels.push_back(
        MakeSparseLevelData( n, domain_length, fineN ) );
    }
  }
  return composite_levels;
}

void FillPressureGhostCells(  //
  LevelData& fine_level,
  const LevelData& crse_level,
  int fineN )
{
  // Set up boundary conditions for pressure
  amrex::Vector<amrex::BCRec> bcs( 1 );  // One component for pressure
  for ( int idim = 0; idim < AMREX_SPACEDIM; ++idim ) {
    bcs[0].setLo( idim, amrex::BCType::ext_dir );
    bcs[0].setHi( idim, amrex::BCType::ext_dir );
  }

  // Create boundary condition functors
  PressureBndryFunc cbc( fineN );
  PressureBndryFunc fbc( fineN );
  amrex::PhysBCFunct<PressureBndryFunc> cphysbc( crse_level.geom, bcs, cbc );
  amrex::PhysBCFunct<PressureBndryFunc> fphysbc( fine_level.geom, bcs, fbc );

  // Create or reuse FillPatcher if not already initialized
  if ( !fine_level.fillpatcher ) {
    const amrex::IntVect nghost( 1 );  // Number of ghost cells to fill
    const int ncomp = 1;               // One component for pressure
    fine_level
      .fillpatcher = std::make_unique<amrex::FillPatcher<amrex::MultiFab>>(  //
      fine_level.pressure.boxArray(),
      fine_level.pressure.DistributionMap(),
      fine_level.geom,
      crse_level.pressure.boxArray(),
      crse_level.pressure.DistributionMap(),
      crse_level.geom,
      nghost,
      ncomp,
      &amrex::pc_interp );
  }

  // Fill ghost cells using FillPatcher
  const amrex::IntVect nghost( 1 );
  const amrex::Real time = 0.0;  // Time is not used in this case
  amrex::Vector<amrex::MultiFab*> cmf = { const_cast<amrex::MultiFab*>(
    &crse_level.pressure ) };
  amrex::Vector<amrex::Real> ct = { time };
  amrex::Vector<amrex::MultiFab*> fmf = { &fine_level.pressure };
  amrex::Vector<amrex::Real> ft = { time };

  fine_level.fillpatcher->fill(  //
    fine_level.pressure,         // Destination
    nghost,                      // Number of ghost cells to fill
    time,                        // Time
    cmf,                         // Coarse level data
    ct,                          // Coarse level times
    fmf,                         // Fine level data
    ft,                          // Fine level times
    0,                           // Source component
    0,                           // Destination component
    1,                           // Number of components
    cphysbc,                     // Coarse level boundary conditions
    0,                           // Coarse level boundary condition component
    fphysbc,                     // Fine level boundary conditions
    0,                           // Fine level boundary condition component
    bcs,                         // Boundary conditions
    0                            // Boundary condition component
  );
}

void SolvePressureCorrection(  //
  amrex::MultiFab& crse_pressure,
  const amrex::MultiFab& fine_pressure,
  const amrex::Geometry& crse_geom,
  const amrex::Geometry& fine_geom,
  amrex::Real tolerance,
  int max_iterations,
  int fineN )
{
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
  // Initialize ghosts?????
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
  SolvePressureIterations(  //
    correction_solution,
    correction_div,
    crse_geom,
    tolerance,
    max_iterations,
    fineN );

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
/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/
