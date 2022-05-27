// Copyright 2021 Intel Corporation
#include <fstream>
#include <list>
#include <sstream>
#include <utility>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "pmlc/dialect/pxa/ir/matchers.h"
#include "pmlc/target/x86/pass_detail.h"
#include "pmlc/util/env.h"
#include "pmlc/util/matchers.h"
#include "pmlc/util/tags.h"
using namespace mlir; // NOLINT[build/namespaces]

namespace pmlc::target::x86 {

namespace {

namespace pxa = dialect::pxa;

struct ShapeAnalysisPass : public ShapeAnalysisBase<ShapeAnalysisPass> {
  Value *matchTppPattern(AffineParallelOp op) {
    using matchers::m_Any;
    Value *tpp = new Value();
    Operation *yield = op.getBody()->getTerminator();
    if (matchPattern(yield, m_Op<AffineYieldOp>(
                                m_Capture(tpp, m_Op<pxa::PxaGenericOp>())))) {
      return tpp;
    }

    return NULL;
  }

  void getTppDetails(
      pxa::PxaGenericOp op, std::list<Operation *> parentOps,
      std::stringstream &opString,
      std::map<std::pair<Block *, int>, std::pair<int, int>> &blockToIndexMap) {
    static std::string prefix[5] = {"M", "N", "K", "KBatch", "Others"};
    if (op.getLoc().dyn_cast<FileLineColLoc>()) {
      opString << "(Line=" << op.getLoc().dyn_cast<FileLineColLoc>().getLine()
               << ")" << op.kernel().str() << ":";
    }
    int i = 0;
    int brgemmCount = 1;
    for (auto tileIndex : op.tile()) {
      if (i <= 4) {
        opString << prefix[i] << "=";
      }
      opString << tileIndex.dyn_cast<IntegerAttr>().getInt();
      if (i < op.tile().size() - 1) {
        opString << ",";
      }
      if (i >= 3) {
        brgemmCount *= tileIndex.dyn_cast<IntegerAttr>().getInt();
      }
      i++;
    }

    if (op.kernel().str() == "tpp_gemm") {
      for (auto parentOp : parentOps) {
        AffineParallelOp parentOpAffine = dyn_cast<AffineParallelOp>(parentOp);
        if (!parentOpAffine) {
          continue;
        }

        for (auto iv : parentOpAffine.getIVs()) {
          auto indices = op.inputIndices();
          size_t prefix = 0;

          for (int i = 0; i < op.getNumInputs(); i++) {
            Attribute accessMap = op.inputAccessMaps()[i];
            AffineMapAttr accessMapAttr = accessMap.cast<AffineMapAttr>();
            size_t count = accessMapAttr.getValue().getNumInputs();
            auto valueRangeOp = indices.slice(prefix, count);
            AffineMap accessMapVal = accessMapAttr.getValue();
            int index = -1;
            for (int j = 0; j < accessMapVal.getNumResults(); j++) {
              std::list<AffineExpr> exprList;
              exprList.push_back(accessMapVal.getResults()[j]);
              while (!exprList.empty()) {
                auto tempExpr = exprList.front();
                exprList.pop_front();
                if (tempExpr.getKind() == AffineExprKind::DimId) {
                  unsigned pos = tempExpr.cast<AffineDimExpr>().getPosition();
                  if (valueRangeOp[pos] == iv) {
                    index = j;
                    break;
                  }
                } else if (tempExpr.dyn_cast<AffineBinaryOpExpr>()) {
                  exprList.push_back(
                      tempExpr.dyn_cast<AffineBinaryOpExpr>().getLHS());
                  exprList.push_back(
                      tempExpr.dyn_cast<AffineBinaryOpExpr>().getRHS());
                }
              }
              if (index > -1) {
                break;
              }
            }
            if (index > -1) {
              blockToIndexMap.insert(std::make_pair(
                  std::make_pair(iv.getOwner(), iv.getArgNumber()),
                  std::make_pair(i, index)));
              break;
            }
            prefix += count;
          }
        }
      }
      opString << ";BRGEMM: " << brgemmCount;
    }
    opString << "\n";
  }

  void runOnOperation() final {
    std::string weightPrefixes[6] = {"K", "C", "R", "S", "C'", "K'"};
    std::string inputPrefixes[5] = {"N", "H", "W", "C"};
    std::string reorderedInputPrefixes[5] = {"N", "C", "H", "W", "C'"};

    if (!util::getEnvVar("PLAIDML_SHAPE_ANALYSIS_OUTPUT").empty()) {
      auto outputFileName = util::getEnvVar("PLAIDML_SHAPE_ANALYSIS_OUTPUT");
      std::ofstream outputFile(outputFileName, std::ofstream::binary);
      getOperation().walk([&](AffineParallelOp op) {
        auto tppPattern = matchTppPattern(op);
        // Find every tpp that's in the block
        if (tppPattern != NULL) {
          auto pxaTppOp =
              dyn_cast<pxa::PxaGenericOp>(tppPattern->getDefiningOp());
          if (pxaTppOp.kernel().str() != "tpp_identity") {
            Operation *tempOp = op;
            std::list<Operation *> parentOps;
            parentOps.push_front(tempOp);
            while (tempOp->getParentOp() != NULL) {
              parentOps.push_front(tempOp->getParentOp());
              tempOp = tempOp->getParentOp();
            }
            std::list<pxa::PxaGenericOp> opList;
            pxa::PxaGenericOp gemmOp;
            opList.push_back(pxaTppOp);
            std::stringstream opString;
            std::map<std::pair<Block *, int>, std::pair<int, int>>
                blockArgAndIndex;
            while (!opList.empty()) {
              auto tppOperator = opList.front();
              if (tppOperator.kernel().str() == "tpp_gemm") {
                gemmOp = tppOperator;
              }
              opList.pop_front();
              getTppDetails(tppOperator, parentOps, opString, blockArgAndIndex);
              for (int i = 0; i < tppOperator->getNumOperands(); i++) {
                auto tppOp = tppOperator->getOperand(i).getDefiningOp();
                if (tppOp != NULL && dyn_cast<pxa::PxaGenericOp>(tppOp)) {
                  opList.push_back(dyn_cast<pxa::PxaGenericOp>(tppOp));
                }
              }
            }
            outputFile << opString.str();
            for (auto itr : parentOps) {
              auto opItr = dyn_cast<AffineParallelOp>(itr);
              if (!opItr)
                continue;
              if (hasUnitTag(opItr, "cpuThread")) {
                outputFile << "PARALLEL";
              } else {
                outputFile << "SERIAL";
              }
              outputFile << " LowerBounds=(";
              for (int i = 0; i < opItr.lowerBoundsMap().getNumResults(); i++) {
                auto blockArg = opItr.getBody()->getArguments()[i];
                for (int j = 0; j < opItr.getLowerBoundMap(i).getNumResults();
                     j++) {
                  auto argMap = blockArgAndIndex[std::make_pair(
                      blockArg.getOwner(), blockArg.getArgNumber())];
                  int inputIndex = argMap.first;
                  int inputOffset = argMap.second;
                  std::string prefix;
                  if (inputIndex == 0) {
                    assert(inputOffset < 5);
                    if (!util::getEnvVar("PLAIDML_REORDER").empty()) {
                      prefix = reorderedInputPrefixes[inputOffset];
                    } else {
                      prefix = inputPrefixes[inputOffset];
                    }
                  } else {
                    assert(inputIndex == 1 && inputOffset < 6);
                    prefix = weightPrefixes[inputOffset];
                  }

                  outputFile << prefix << "="
                             << opItr.getLowerBoundMap(i)
                                    .getResult(j)
                                    .cast<AffineConstantExpr>()
                                    .getValue();
                  if (j < opItr.getLowerBoundMap(i).getNumResults() - 1)
                    outputFile << ",";
                }
                if (i < opItr.lowerBoundsMap().getNumResults() - 1)
                  outputFile << ",";
              }
              outputFile << ") UpperBounds=(";

              for (int i = 0; i < opItr.upperBoundsMap().getNumResults(); i++) {
                auto blockArg = opItr.getBody()->getArguments()[i];
                for (int j = 0; j < opItr.getUpperBoundMap(i).getNumResults();
                     j++) {
                  auto argMap = blockArgAndIndex[std::make_pair(
                      blockArg.getOwner(), blockArg.getArgNumber())];
                  int inputIndex = argMap.first;
                  int inputOffset = argMap.second;
                  std::string prefix;
                  if (inputIndex == 0) {
                    assert(inputOffset < 5);
                    if (!util::getEnvVar("PLAIDML_REORDER").empty()) {
                      prefix = reorderedInputPrefixes[inputOffset];
                    } else {
                      prefix = inputPrefixes[inputOffset];
                    }
                  } else {
                    assert(inputIndex == 1 && inputOffset < 6);
                    prefix = weightPrefixes[inputOffset];
                  }

                  outputFile << prefix << "="
                             << opItr.getUpperBoundMap(i)
                                    .getResult(j)
                                    .cast<AffineConstantExpr>()
                                    .getValue();
                  if (j < opItr.getUpperBoundMap(i).getNumResults() - 1)
                    outputFile << ",";
                }
                if (i < opItr.upperBoundsMap().getNumResults() - 1)
                  outputFile << ",";
              }
              outputFile << ") Steps=(";

              for (int i = 0; i < opItr.getSteps().size(); i++) {
                outputFile << opItr.getSteps()[i];
                if (i < opItr.getSteps().size() - 1) {
                  outputFile << ",";
                }
              }
              outputFile << ")\n";
            }

            outputFile << "\n-----------\n";
          }
        }
        return WalkResult::skip();
      });
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createShapeAnalysisPass() {
  return std::make_unique<ShapeAnalysisPass>();
}

} // namespace pmlc::target::x86
