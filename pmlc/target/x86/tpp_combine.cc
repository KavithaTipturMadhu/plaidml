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
public:
  bool maybeCaptureTopLevel(AffineParallelOp op) {
    using matchers::m_Any;
    Value *gemmOp = new Value();
    Value *reluOp = new Value();
    auto opPattern =
        m_Op<AffineYieldOp>(m_Capture(reluOp, m_Op<pxa::PxaGenericOp>()));
    auto affineYield = op.getBody()->getTerminator();
    if (!matchPattern(affineYield, opPattern)) {
      return false;
    }
    pxa::PxaGenericOp pxaReluOp =
        dyn_cast<pxa::PxaGenericOp>(reluOp->getDefiningOp());
    if (pxaReluOp.kernel().str() == "tpp_relu") {
      auto pxaGemmOp = pxaReluOp.getOperand(0);
      if (dyn_cast<pxa::PxaGenericOp>(pxaGemmOp.getDefiningOp()) &&
          dyn_cast<pxa::PxaGenericOp>(pxaGemmOp.getDefiningOp())
                  .kernel()
                  .str() == "tpp_gemm") {
        auto pxaIdOp =
            dyn_cast<pxa::PxaGenericOp>(pxaGemmOp.getDefiningOp()).outputs()[0];
        bool identityOp = false;
        for (auto inst = op.getBody()->begin(); inst != op.getBody()->end();
             inst++) {
          // TODO check for broadcast calls: row major
          if (dyn_cast<pxa::PxaGenericOp>(inst) &&
              dyn_cast<pxa::PxaGenericOp>(inst).kernel().str() ==
                  "tpp_identity") {
            if (&*inst == pxaIdOp.getDefiningOp()) {
              identityOp = true;
              break;
            }
          }
        }
        if (identityOp) {
          std::cout << "identity op found\n";
          return true;
        }
      }
    }
    return false;
  }
};
} // namespace

struct TppCombinePass : public TppCombineBase<TppCombinePass> {
  void runOnOperation() final {
    getOperation().walk([](AffineParallelOp op) {
      TppCombineImpl combineImpl;
      combineImpl.maybeCaptureTopLevel(op);
    });
  }
};

std::unique_ptr<Pass> createTppCombinePass() {
  return std::make_unique<TppCombinePass>();
}

} // namespace pmlc::target::x86
