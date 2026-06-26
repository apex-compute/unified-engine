//===- StandaloneDialect.td - Standalone dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_DIALECT
#define STANDALONE_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Standalone dialect definition.
//===----------------------------------------------------------------------===//

def Standalone_Dialect : Dialect {
    let name = "ue";
    let summary = "Unified-engine hardware op dialect.";
    let description = [{
        The `ue` dialect models unified-engine FPGA accelerator ops 1:1 with the
        hlo.py UE_HLO kernel catalog. Programs are ordered op blocks; the order of
        ops in a block is the execution schedule fed to the device.
    }];
    let cppNamespace = "::mlir::standalone";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class Standalone_Op<string mnemonic, list<Trait> traits = []> :
        Op<Standalone_Dialect, mnemonic, traits>;

#endif // STANDALONE_DIALECT
