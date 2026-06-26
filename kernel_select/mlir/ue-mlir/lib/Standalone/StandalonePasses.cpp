//===- StandalonePasses.cpp - Standalone passes -----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Standalone/StandaloneOps.h"
#include "Standalone/StandalonePasses.h"

#include <cmath>

namespace mlir::standalone {
#define GEN_PASS_DEF_STANDALONESWITCHBARFOO
#define GEN_PASS_DEF_UERAISEWINDOWS
#include "Standalone/StandalonePasses.h.inc"

namespace {
//===----------------------------------------------------------------------===//
// standalone-switch-bar-foo (skeleton example, unchanged)
//===----------------------------------------------------------------------===//
class StandaloneSwitchBarFooRewriter : public OpRewritePattern<func::FuncOp> {
public:
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const final {
    if (op.getSymName() == "bar") {
      rewriter.modifyOpInPlace(op, [&op]() { op.setSymName("foo"); });
      return success();
    }
    return failure();
  }
};

class StandaloneSwitchBarFoo
    : public impl::StandaloneSwitchBarFooBase<StandaloneSwitchBarFoo> {
public:
  using impl::StandaloneSwitchBarFooBase<
      StandaloneSwitchBarFoo>::StandaloneSwitchBarFooBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<StandaloneSwitchBarFooRewriter>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};

//===----------------------------------------------------------------------===//
// ue-raise-windows
//
// Anchor on the FINAL ue.reshape of a (reshape -> permute -> reshape) chain and
// raise it to ue.window_partition / ue.window_reverse ONLY when the signature is
// exactly the Swin window idiom. Otherwise leave it (head-split etc. fall back to
// generic ops). This is the "guard on shape, not structure" rule.
//===----------------------------------------------------------------------===//
static bool readPerm(PermuteOp p, SmallVectorImpl<int64_t> &out) {
  for (Attribute a : p.getPerm())
    if (auto i = dyn_cast<IntegerAttr>(a))
      out.push_back(i.getInt());
    else
      return false;
  return true;
}

// integer sqrt; returns -1 if n is not a perfect square.
static int64_t isqrtExact(int64_t n) {
  if (n < 0) return -1;
  int64_t r = (int64_t)std::llround(std::sqrt((double)n));
  for (int64_t c = r - 1; c <= r + 1; ++c)
    if (c >= 0 && c * c == n) return c;
  return -1;
}

class RaiseWindowPattern : public OpRewritePattern<ReshapeOp> {
public:
  using OpRewritePattern<ReshapeOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ReshapeOp finalRs,
                                PatternRewriter &rewriter) const final {
    // final.input must be a permute; permute.input must be a reshape.
    auto perm = finalRs.getInput().getDefiningOp<PermuteOp>();
    if (!perm) return failure();
    auto firstRs = perm.getInput().getDefiningOp<ReshapeOp>();
    if (!firstRs) return failure();

    // The window permute signature: exactly [0,1,3,2,4,5].
    SmallVector<int64_t> pv;
    if (!readPerm(perm, pv)) return failure();
    const int64_t want[6] = {0, 1, 3, 2, 4, 5};
    if (pv.size() != 6) return failure();
    for (int i = 0; i < 6; ++i)
      if (pv[i] != want[i]) return failure();   // e.g. head-split [2,0,3,1,4] -> bail

    auto inTy = dyn_cast<RankedTensorType>(firstRs.getInput().getType());
    auto outTy = dyn_cast<RankedTensorType>(finalRs.getResult().getType());
    if (!inTy || !outTy) return failure();

    Value src = firstRs.getInput();
    Location loc = finalRs.getLoc();

    // Direction by rank. window dim = sqrt of the (ws*ws) axis on the rank-3 side.
    if (inTy.getRank() == 4 && outTy.getRank() == 3) {
      int64_t ws = isqrtExact(outTy.getDimSize(1));
      if (ws <= 0) return failure();
      rewriter.replaceOpWithNewOp<WindowPartitionOp>(
          finalRs, outTy, src, rewriter.getI64IntegerAttr(ws));
    } else if (inTy.getRank() == 3 && outTy.getRank() == 4) {
      int64_t ws = isqrtExact(inTy.getDimSize(1));
      if (ws <= 0) return failure();
      rewriter.replaceOpWithNewOp<WindowReverseOp>(
          finalRs, outTy, src, rewriter.getI64IntegerAttr(ws));
    } else {
      return failure();
    }

    // Clean up the now-dead permute/reshape if nothing else uses them.
    if (perm.use_empty()) rewriter.eraseOp(perm);
    if (firstRs.use_empty()) rewriter.eraseOp(firstRs);
    return success();
  }
};

class UERaiseWindows : public impl::UERaiseWindowsBase<UERaiseWindows> {
public:
  using impl::UERaiseWindowsBase<UERaiseWindows>::UERaiseWindowsBase;
  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    patterns.add<RaiseWindowPattern>(&getContext());
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPatternsAndFoldGreedily(getOperation(), patternSet)))
      signalPassFailure();
  }
};
} // namespace
} // namespace mlir::standalone
