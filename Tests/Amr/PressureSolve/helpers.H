#pragma once

/*--------------------------------------------------------------------
  standard includes
  --------------------------------------------------------------------*/
#include <AMReX_BCRec.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <array>
#include <vector>

struct LevelData
{
  LevelData( int base_n, int level, amrex::Real domain_length );
  const amrex::Geometry geom;
  amrex::MultiFab pressure;
  std::array<amrex::MultiFab, 3> velocity;

  /*--------------------------------------------------------------------
    public free function declarations
    --------------------------------------------------------------------*/
};

// Entry point for recursive composite-mesh pressure solve
void CompositeSolve(  //
  std::vector<LevelData>& composite_levels,
  amrex::Real tolerance,
  int max_iterations,
  int base_n,
  int nlevels );

// Sinle-level pressure solve
void SingleLevelPressureSolve(  //
  amrex::MultiFab& pressure,
  const std::array<amrex::MultiFab, 3>& velocity,
  const amrex::Geometry& geom,
  amrex::Real tolerance,
  int max_iterations,
  int base_n,
  int nlevels );

// Verification and comparison
void CheckResults(  //
  const amrex::MultiFab& pressure,
  const amrex::Geometry& geom,
  int base_n,
  int nlevels );
void CompareMultiFabs(  //
  const LevelData& expected_level,
  const std::vector<LevelData>& level_data );

// Plotting
void SamplePressureAlongLine(  //
  const std::vector<LevelData>& level_data,
  const LevelData& fullFineSolution,
  int base_n,
  int nlevels );

// Dense or sparse/composite mesh defintion
LevelData MakeDenseLevelData(  //
  amrex::Real domain_length,
  int base_n,
  int nlevels,
  int level );
std::vector<LevelData> MakeSparseCompositeLevels(  //
  int base_n,
  int nlevels,
  amrex::Real domain_length );

constexpr int CalculateFineN( int base_n, int nlevels );
constexpr int CalculateNForLevel( int base_n, int level );

/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/
