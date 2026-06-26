// RUN: standalone-opt %s | standalone-opt | FileCheck %s
//
// One Swin (W-MSA) transformer block expressed in the `ue` dialect, in the
// exact execution order the device runs. Round-tripping this through
// standalone-opt twice must reproduce the same ordered op list = the
// order-verification harness against k_select's chainer.

// CHECK-LABEL: func.func @swin_block
func.func @swin_block(
    %x:    tensor<3136x96xbf16>,   // block input  [H*W=56*56, C=96]
    %g1:   tensor<96xbf16>, %b1: tensor<96xbf16>,
    %wqkv: tensor<288x96xbf16>,
    %wo:   tensor<96x96xbf16>,
    %g2:   tensor<96xbf16>, %b2: tensor<96xbf16>,
    %w1:   tensor<384x96xbf16>,    // MLP up
    %w2:   tensor<96x384xbf16>     // MLP down
  ) -> tensor<3136x96xbf16> {

  // CHECK: ue.layer_norm
  %n1   = ue.layer_norm %x, %g1, %b1
          : (tensor<3136x96xbf16>, tensor<96xbf16>, tensor<96xbf16>) -> tensor<3136x96xbf16>

  // CHECK: ue.window_partition
  %win  = ue.window_partition %n1 {window_size = 7 : i64}
          : (tensor<3136x96xbf16>) -> tensor<3136x96xbf16>

  // CHECK: ue.matmul
  %qkv  = ue.matmul %win, %wqkv
          : (tensor<3136x96xbf16>, tensor<288x96xbf16>) -> tensor<3136x288xbf16>

  // CHECK: ue.flash_attn_pf
  %attn = ue.flash_attn_pf %qkv {causal = false, heads = 3 : i64}
          : (tensor<3136x288xbf16>) -> tensor<3136x96xbf16>

  // CHECK: ue.matmul
  %proj = ue.matmul %attn, %wo
          : (tensor<3136x96xbf16>, tensor<96x96xbf16>) -> tensor<3136x96xbf16>

  // CHECK: ue.window_reverse
  %mrg  = ue.window_reverse %proj {window_size = 7 : i64}
          : (tensor<3136x96xbf16>) -> tensor<3136x96xbf16>

  // CHECK: ue.layer_norm_post_add
  %r1   = ue.layer_norm_post_add %x, %mrg, %g2, %b2
          : (tensor<3136x96xbf16>, tensor<3136x96xbf16>, tensor<96xbf16>, tensor<96xbf16>) -> tensor<3136x96xbf16>

  // CHECK: ue.matmul {{.*}}gelu
  %fc1  = ue.matmul %r1, %w1 {act = "gelu"}
          : (tensor<3136x96xbf16>, tensor<384x96xbf16>) -> tensor<3136x384xbf16>

  // CHECK: ue.matmul
  %fc2  = ue.matmul %fc1, %w2
          : (tensor<3136x384xbf16>, tensor<96x384xbf16>) -> tensor<3136x96xbf16>

  // CHECK: ue.eltwise
  %out  = ue.eltwise %r1, %fc2 {mode = "add"}
          : (tensor<3136x96xbf16>, tensor<3136x96xbf16>) -> tensor<3136x96xbf16>

  return %out : tensor<3136x96xbf16>
}
