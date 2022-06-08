// Copyright 2020 Intel Corporation
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "pmlc/dialect/pxa/ir/matchers.h"
#include "pmlc/dialect/pxa/ir/ops.h"
#include "pmlc/dialect/pxa/transforms/autotile.h"
#include "pmlc/dialect/pxa/transforms/stencil.h"
#include "pmlc/dialect/stdx/ir/ops.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/target/x86/passes.h"
#include "pmlc/util/logging.h"
#include "pmlc/util/matchers.h"
#include "pmlc/util/util.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir; // NOLINT

namespace pmlc::target::x86 {

namespace pxa = dialect::pxa;
namespace stdx = dialect::stdx;

namespace {

class TppCombineImpl {
private:
  bool maybeCaptureTopLevel(AffineParallelOp op) {
    using matchers::m_Any;
    // Check for unary /binary ops back to back within the affineparallel op

    Value *gemmOp = new Value();
    Value *identityOp = new Value();

    auto opPattern = m_Op<AffineYieldOp>(
        arith::AtomicRMWKind::assign,
        m_Op<stdx::ReluOp>(
            m_Capture(gemmOp, m_Op<pxa::PxaGenericOp>(m_Capture(
                                  identityOp, m_Op<pxa::PxaGenericOp>())))),
        m_Any());

    auto affineYield = op.getBody()->getTerminator();
    if (!matchPattern(affineYield, opPattern)) {
      return false;
    }
    pxa::PxaGenericOp pxaGemmOp =
        dyn_cast<pxa::PxaGenericOp>(gemmOp->getDefiningOp());
    pxa::PxaGenericOp pxaIdentityOp =
        dyn_cast<pxa::PxaGenericOp>(identityOp->getDefiningOp());
    if (pxaGemmOp.kernel().str() != "tpp_gemm" ||
        pxaIdentityOp.kernel().str() != "tpp_identity") {
      return false;
    }
    return true;
  }
};
} // namespace

struct TppCombinePass : public TppCombineBase<TppCombinePass> {
  void runOnOperation() final {
    getOperation().walk(
        [](AffineParallelOp op) { TppCombineImpl combineImpl; });
  }
};

std::unique_ptr<Pass> createTppCombinePass() {
  return std::make_unique<TppCombinePass>();
}

} // namespace pmlc::target::x86
