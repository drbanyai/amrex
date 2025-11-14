#include "stencil.H"
#include <memory>

namespace PressureSolve {

std::unique_ptr<StencilBase> createStencil( StencilType type )
{
  switch ( type ) {
    case StencilType::Standard5Point:
      return std::make_unique<StandardStencil>();
    case StencilType::HOC9Point:
      return std::make_unique<HOCStencil>();
    default:
      amrex::Abort( "Unknown stencil type" );
      return nullptr;
  }
}

}  // namespace PressureSolve