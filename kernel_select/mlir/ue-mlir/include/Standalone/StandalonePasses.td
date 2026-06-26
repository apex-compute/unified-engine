//===- StandalonePsss.td - Standalone dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_PASS
#define STANDALONE_PASS

include "mlir/Pass/PassBase.td"

def UERaiseWindows: Pass<"ue-raise-windows", "::mlir::ModuleOp"> {
  let summary = "Raise reshape+permute+reshape window idioms to ue.window_partition/reverse.";
  let description = [{
    Pattern-raise (NOT structural): only fires on the EXACT Swin window signature
    — perm == [0,1,3,2,4,5] wrapping a rank-4<->rank-3 reshape pair. A generic
    reshape+permute+reshape (e.g. attention head-split, perm [2,0,3,1,4]) is left
    untouched and lowers via the generic ops. Direction is by rank:
      rank4 -> rank3  =>  ue.window_partition
      rank3 -> rank4  =>  ue.window_reverse
  }];
}

def StandaloneSwitchBarFoo: Pass<"standalone-switch-bar-foo", "::mlir::ModuleOp"> {
  let summary = "Switches the name of a FuncOp named `bar` to `foo` and folds.";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    func.func @bar() {
      return
    }
    // Gets transformed to:
    func.func @foo() {
      return
    }
    ```
  }];
}

#endif // STANDALONE_PASS
