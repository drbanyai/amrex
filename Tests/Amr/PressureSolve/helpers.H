#pragma once

/*--------------------------------------------------------------------
  standard includes
  --------------------------------------------------------------------*/
#include <AMReX_BCRec.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FillPatcher.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <array>
#include <memory>
#include <vector>

struct LevelData
{
  LevelData( int n, amrex::Real domain_length );
  const amrex::Geometry geom;
  amrex::MultiFab pressure;
  std::array<amrex::MultiFab, 3> velocity;
  std::unique_ptr<amrex::FillPatcher<amrex::MultiFab>> fillpatcher;

  /*--------------------------------------------------------------------
    public free function declarations
    --------------------------------------------------------------------*/
};

// Geometry and mesh setup
amrex::Geometry DefineGeometry( int nx, int ny, int nz, double dx );
amrex::BoxArray DefineBoxArray( int n );
amrex::BoxArray DefineSparseBoxArray( int n );
amrex::DistributionMapping DefineDM( const amrex::BoxArray& ba );

// Data structure setup
void DefineFABs(  //
  amrex::MultiFab& pressure,
  std::array<amrex::MultiFab, 3>& velocity,
  const amrex::BoxArray& ba,
  const amrex::DistributionMapping& dm );

// Main pressure solver function
void SolvePressure(  //
  amrex::MultiFab& pressure,
  const std::array<amrex::MultiFab, 3>& velocity,
  const amrex::Geometry& geom,
  amrex::Real tolerance,
  int max_iterations,
  int base_n,
  int nlevels );

// Verification
// Multi-level sampling and CSV output
/**
 * @brief Sample pressure along the center line at the finest resolution,
 * interpolating all levels to the fine grid, and write a CSV with columns for
 * each level.
 * @param level_data Vector of LevelData objects (coarse to fine)
 * @param filename Output CSV filename
 */
void SamplePressureAlongLine(  //
  const std::vector<LevelData>& level_data,
  const std::string& filename,
  const LevelData& fullFineSolution,
  int base_n,
  int nlevels );

void CheckResults(  //
  const amrex::MultiFab& pressure,
  const amrex::Geometry& geom,
  int base_n,
  int nlevels );

// Comparison utilities
/**
 * @brief Compare a reference LevelData against a vector of LevelData at
 * different AMR levels
 * @param expected_level Reference LevelData to compare against
 * @param level_data Vector of LevelData at different AMR levels
 */
void CompareMultiFabs(  //
  const LevelData& expected_level,
  const std::vector<LevelData>& level_data );

LevelData MakeDenseLevelData(  //
  int n,
  amrex::Real domain_length,
  int base_n,
  int nlevels );
std::vector<LevelData> MakeSparseCompositeLevels(  //
  int base_n,
  int nlevels,
  amrex::Real domain_length );

// Fill ghost cells for pressure field between AMR levels
void FillPressureGhostCells(  //
  LevelData& fine_level,
  const LevelData& crse_level );

// Solve for pressure correction between AMR levels
void SolvePressureCorrection(  //
  amrex::MultiFab& crse_pressure,
  const amrex::MultiFab& fine_pressure,
  const amrex::Geometry& crse_geom,
  const amrex::Geometry& fine_geom,
  amrex::Real tolerance,
  int max_iterations,
  int base_n,
  int nlevels );

/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/
