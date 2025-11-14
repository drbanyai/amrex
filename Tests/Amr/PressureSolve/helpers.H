#pragma once

/*--------------------------------------------------------------------
  standard includes
  --------------------------------------------------------------------*/
#include "stencil.H"
#include <AMReX_BCRec.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Vector.H>
#include <array>
#include <memory>
#include <vector>

// Forward declarations
struct LevelData;

struct LevelData {
  LevelData(int base_n, int level, amrex::Real domain_length);
  const amrex::Geometry geom;
  amrex::MultiFab pressure;
  std::array<amrex::MultiFab, 3> velocity;

  /*--------------------------------------------------------------------
    public free function declarations
    --------------------------------------------------------------------*/
};

// Entry point for recursive composite-mesh pressure solve
void CompositeSolve( //
    std::vector<LevelData> &composite_levels, amrex::Real tolerance,
    int max_iterations, int base_n, int nLevels,
    PressureSolve::StencilType stencil_type =
        PressureSolve::StencilType::Standard5Point);

// Sinle-level pressure solve
void SingleLevelPressureSolve( //
    amrex::MultiFab &pressure, const std::array<amrex::MultiFab, 3> &velocity,
    const amrex::Geometry &geom, amrex::Real tolerance, int max_iterations,
    int base_n, int nLevels, PressureSolve::StencilType stencil_type);

void AmrexPressureSolve( //
    std::vector<LevelData> &levelData, int nLevels);

// void WriteAmrexResults(std::vector<LevelData> &levelData, int nLevels);

// Verification and comparison
void CheckResults( //
    const amrex::MultiFab &pressure, const amrex::Geometry &geom, int base_n,
    int nLevels);
void CompareMultiFabs( //
    const LevelData &expected_level, const std::vector<LevelData> &level_data);

// Plotting
void SamplePressureAlongLine( //
    const std::vector<LevelData> &level_data, const LevelData &fullFineSolution,
    int base_n, int nLevels);

// Dense or sparse/composite mesh defintion
LevelData MakeDenseLevelData( //
    amrex::Real domain_length, int base_n, int nLevels, int level);
std::vector<LevelData> MakeSparseCompositeLevels( //
    int base_n, int nlevels, amrex::Real domain_length);

static int CalculateNForLevel(int base_n, int level) {
  return base_n * (1 << level);
}

static int CalculateFineN(int base_n, int nlevels) {
  return CalculateNForLevel(base_n, nlevels - 1);
}

/*--------------------------------------------------------------------
  End of file
  --------------------------------------------------------------------*/
