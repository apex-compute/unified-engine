//===- StandaloneOps.td - ue dialect ops -------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STANDALONE_OPS
#define STANDALONE_OPS

include "Standalone/StandaloneTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// All ue ops below operate on ranked tensors and use a uniform generic
// assembly format:  %res = ue.<op> <operands> {attrs} : (in-types) -> out-type
//
// NOTE on ordering: these ops are intentionally NOT marked `Pure`. They model
// device kernels that write DRAM, so MLIR must treat them as opaque/ordered and
// must never reorder, fold, or DCE them. The block order = the execution order
// fed to the device. (A MemWrite memory-effect trait will be added once outputs
// move to destination-passing memref operands; for the round-trip harness the
// data-dependency chain already pins order.)

class UE_Op<string mnemonic, list<Trait> traits = []> :
        Op<Standalone_Dialect, mnemonic, traits>;

def UE_GenericFmt {
    string fmt = "operands attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Patch embedding (vision encoder front).  -> patching_core
//===----------------------------------------------------------------------===//
def UE_PatchingOp : UE_Op<"patching"> {
    let summary = "im2col + matmul patch embedding (vision encoder input).";
    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$weight,
                         I64Attr:$patch_size);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

//===----------------------------------------------------------------------===//
// Normalization.  -> layer_norm_core_dram / layer_norm_core_dram_post_add
//===----------------------------------------------------------------------===//
def UE_LayerNormOp : UE_Op<"layer_norm"> {
    let summary = "LayerNorm with affine gamma/beta.";
    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$gamma,
                         AnyRankedTensor:$beta);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

def UE_LayerNormPostAddOp : UE_Op<"layer_norm_post_add"> {
    let summary = "Fused residual-add then LayerNorm.";
    let arguments = (ins AnyRankedTensor:$input,
                         AnyRankedTensor:$residual,
                         AnyRankedTensor:$gamma,
                         AnyRankedTensor:$beta);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

//===----------------------------------------------------------------------===//
// Matmul (optionally fused activation).  -> matmat_mul_core / quantized_matmat_core
//===----------------------------------------------------------------------===//
def UE_MatmulOp : UE_Op<"matmul"> {
    let summary = "A @ B^T with optional fused gelu/silu activation.";
    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         DefaultValuedAttr<StrAttr, "\"none\"">:$act);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

//===----------------------------------------------------------------------===//
// Elementwise.  -> eltwise_core_dram
//===----------------------------------------------------------------------===//
def UE_EltwiseOp : UE_Op<"eltwise"> {
    let summary = "Elementwise add/mul of two tensors.";
    let arguments = (ins AnyRankedTensor:$a,
                         AnyRankedTensor:$b,
                         DefaultValuedAttr<StrAttr, "\"add\"">:$mode);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

//===----------------------------------------------------------------------===//
// Swin windowing (dedicated kernels, order-sensitive).
//   -> window_partition_dram / window_reverse_dram
//===----------------------------------------------------------------------===//
def UE_WindowPartitionOp : UE_Op<"window_partition"> {
    let summary = "Reshape [H*W, C] into non-overlapping windows.";
    let arguments = (ins AnyRankedTensor:$input, I64Attr:$window_size);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

def UE_WindowReverseOp : UE_Op<"window_reverse"> {
    let summary = "Inverse of window_partition: windows back to [H*W, C].";
    let arguments = (ins AnyRankedTensor:$input, I64Attr:$window_size);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

//===----------------------------------------------------------------------===//
// Generic axis permute (head split / transpose).  -> smart_bf16_permute_core
//===----------------------------------------------------------------------===//
def UE_PermuteOp : UE_Op<"permute"> {
    let summary = "ND axis permute / transpose via DMA gather.";
    let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$perm);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

//===----------------------------------------------------------------------===//
// Pure shape ops (faithful 1:1 with ATen view/reshape/select). A later ue->ue
// fusion pass raises reshape+permute idioms into window_partition/window_reverse.
//===----------------------------------------------------------------------===//
def UE_ReshapeOp : UE_Op<"reshape"> {
    let summary = "Logical reshape (no data movement); result type carries new shape.";
    let arguments = (ins AnyRankedTensor:$input);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

def UE_SelectOp : UE_Op<"select"> {
    let summary = "Select one index along a dim (rank-reducing), e.g. qkv[0].";
    let arguments = (ins AnyRankedTensor:$input, I64Attr:$dim, I64Attr:$index);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

//===----------------------------------------------------------------------===//
// Prefill (bidirectional / encoder) flash attention. -> prefill_flash_attention_core
//===----------------------------------------------------------------------===//
def UE_FlashAttnPfOp : UE_Op<"flash_attn_pf"> {
    let summary = "Prefill flash attention over Q,K,V; causal=false for encoders.";
    let arguments = (ins AnyRankedTensor:$q, AnyRankedTensor:$k, AnyRankedTensor:$v,
                         DefaultValuedAttr<BoolAttr, "false">:$causal,
                         I64Attr:$heads);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

//===----------------------------------------------------------------------===//
// Shifted-window ops + vision head. roll -> cyclic_shift_dram, patch_merge ->
// patch_merging_gather_dram (2x2 gather, the following layer_norm+matmul reduce),
// mean -> sequence mean-pool (host), pad -> spatial pad (identity when all-zero).
//===----------------------------------------------------------------------===//
def UE_RollOp : UE_Op<"roll"> {
    let summary = "Cyclic shift (torch.roll) of a spatial [1,H,W,C] map.";
    let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$shifts, I64ArrayAttr:$dims);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

def UE_PatchMergeOp : UE_Op<"patch_merge"> {
    let summary = "2x2 spatial gather (H,W,C)->(H/2,W/2,4C); concat 2x2 neighbours.";
    let arguments = (ins AnyRankedTensor:$input);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

def UE_MeanOp : UE_Op<"mean"> {
    let summary = "Sequence mean-pool over tokens (final vision head).";
    let arguments = (ins AnyRankedTensor:$input);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

def UE_PadOp : UE_Op<"pad"> {
    let summary = "Spatial pad (identity when all pads are zero).";
    let arguments = (ins AnyRankedTensor:$input, I64ArrayAttr:$pads);
    let results   = (outs AnyRankedTensor:$res);
    let assemblyFormat = UE_GenericFmt.fmt;
}

#endif // STANDALONE_OPS
