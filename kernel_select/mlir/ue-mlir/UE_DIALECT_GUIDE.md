# Building the `ue` MLIR dialect — where each piece lives

This repo is the upstream **`mlir/examples/standalone`** out-of-tree template,
still carrying the name **"Standalone"** everywhere. Renaming `Standalone -> UE`
is step 0. Below is the map from each MLIR *concept* (the ones we keep talking
about — ops, verifiers, patterns, passes, the `ue-opt` engine, emission) to the
exact file in this tree where you define it.

The mental model that matters:

> MLIR never "recognizes" a `ue` op. **You** write the op definition (+ verifier)
> and the rewrite patterns. MLIR just runs them to fixpoint, verifying between
> stages. Every `if op == "matmul":` in `loom_ir.py` becomes a typed
> `OpRewritePattern<ue::MatmulOp>` here.

---

## File map: concept -> where to define it

| Concept | File | What you write |
|---|---|---|
| **Dialect registration** (the `ue` namespace) | `include/Standalone/StandaloneDialect.td` | rename to `UE_Dialect`, mnemonic `"ue"` |
| **Op definitions** (`ue.matmul`, `ue.layer_norm`, `ue.flash_attn`, `ue.conv1d`, …) | `include/Standalone/StandaloneOps.td` | one `def UE_*Op` per op: operands, results, attrs |
| **Verifiers** ("is this op well-formed?" — the 64-ALU floor, K-match) | same `.td` (`let hasVerifier = 1;`) + impl in `lib/Standalone/StandaloneOps.cpp` | the `::verify()` body |
| **Custom types** (if `ue` needs a tile/tensor type) | `include/Standalone/StandaloneTypes.td` | usually fine to reuse builtin tensor/memref |
| **Lowering passes** (torch/linalg -> ue, ue -> backend) | `include/Standalone/StandalonePasses.td` (declare) | pass name, summary, options |
| **Rewrite patterns** (the actual `matchAndRewrite` recognizers) | `lib/Standalone/StandalonePasses.cpp` | `OpRewritePattern<...>` structs + populate them |
| **The `ue-opt` engine** (PassManager CLI driver) | `standalone-opt/standalone-opt.cpp` | registers dialect + passes; this becomes `ue-opt` |
| **Emission / codegen** (ue IR -> FPGA descriptors) | `standalone-translate/standalone-translate.cpp` | the "translate" hook = where `loom_ir.py run()` logic goes |
| **Tests** (`.mlir` in, expected IR out) | `test/Standalone/*.mlir` | FileCheck tests; `swin_block.mlir` already here |

---

## The solidified steps (do them in this order)

### Step 0 — Rename `Standalone` -> `UE`
Mechanical sweep across filenames, `.td` defs, C++ symbols, CMake targets, and the
`standalone` mnemonic. After this, `ninja` should still build and
`standalone-opt` (now `ue-opt`) should still run the template's dummy pass.

### Step 1 — Define the ops in `StandaloneOps.td`
For every op in the `loom_ir.py` dispatch vocabulary, add a `def`. Example shape:

```tablegen
def UE_MatmulOp : UE_Op<"matmul", [Pure]> {
  let summary = "tiled matmul with optional bias + gelu epilogue";
  let arguments = (ins AnyRankedTensor:$a,
                       AnyRankedTensor:$b,
                       Optional<AnyRankedTensor>:$bias,
                       DefaultValuedAttr<BoolAttr, "false">:$gelu);
  let results   = (outs AnyRankedTensor:$out);
  let hasVerifier = 1;          // turn on the well-formedness check
  let assemblyFormat = "$a `,` $b (`bias` $bias^)? attr-dict `:` "
                       "functional-type(operands, results)";
}
```

Starter op set (from the current vocabulary): `matmul`, `layer_norm`,
`flash_attn`, `permute`, `eltwise`, `reshape`, `select`, `roll`, `patch_merge`.
Parakeet later adds: `conv1d`, `conv2d`, `batch_norm`, `glu`, `silu`, `tanh`.

### Step 2 — Write the verifier in `StandaloneOps.cpp`
This is the emission gate — the equivalent of `loom_ir.py`'s shape asserts /
`raise ValueError`. It runs automatically at parse and between passes.

```cpp
LogicalResult ue::MatmulOp::verify() {
  auto aTy = cast<RankedTensorType>(getA().getType());
  auto bTy = cast<RankedTensorType>(getB().getType());
  if (aTy.getRank() != 2 || bTy.getRank() != 2)
    return emitOpError("operands must be 2D");
  if (aTy.getDimSize(1) != bTy.getDimSize(0))
    return emitOpError("K mismatch: ") << aTy.getDimSize(1)
                                       << " vs " << bTy.getDimSize(0);
  return success();
}
```

### Step 3 — Write the lowering patterns in `StandalonePasses.cpp`
This is where `if op == "matmul":` becomes a typed pattern. Two pass families:

- **Raising / import** — incoming dialect (`torch`/`linalg`, or your own importer
  output) -> `ue`. The attention/window idiom recognizers from `_analyze()`
  live here as patterns that match a *subgraph* and replace it with one `ue` op.
- **Backend lowering** — `ue` op -> backend descriptors (Step 5).

```cpp
struct LinalgMatmulToUE : OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ue::MatmulOp>(
        op, op.getType(0), op.getInputs()[0], op.getInputs()[1],
        /*bias=*/Value(), /*gelu=*/false);
    return success();
  }
};
// populate into the pass's RewritePatternSet; the driver runs it to fixpoint.
```

### Step 4 — Register the passes + dialect in `ue-opt`
`standalone-opt.cpp` is the `PassManager` CLI. Register the `ue` dialect and your
passes so they're reachable as pipeline flags:

```bash
ue-opt model.mlir \
   --convert-linalg-to-ue \
   --ue-raise-attention \
   --ue-bufferize \
   -o lowered.mlir
```

### Step 5 — Emission via `ue-translate`
`standalone-translate.cpp` is the **codegen / "translate out of MLIR"** hook.
This is the new home for the back half of `loom_ir.py run()`: walk the final
`ue` module, bind weights, allocate DRAM, emit the FPGA program. Difference from
today: the walk is over a *verified* `ue` ModuleOp, not a parsed dict list.

### Step 6 — Tests in `test/Standalone/`
Drop `ue.matmul` IR into a `.mlir`, add FileCheck lines for the lowered output,
wire it into `lit`. `swin_block.mlir` is already here as a real fixture.

---

## How a model gets in (the frontend — unchanged decision)

MLIR/`ue-opt` only ever consumes `.mlir` text. Producing it is the frontend's job:
- **torch-mlir / torch.export** -> `torch` dialect -> lower to `ue`, **or**
- **your own importer** — `torch_to_ue.py` already does exactly this. Keep it;
  point it at emitting text that parses against the Step-1 ops. It plays the role
  of torch-mlir + the import passes, and is the part you keep almost as-is.

---

## Mapping back to what already works (`loom_ir.py`)

| `loom_ir.py` (Python, today) | `ue-mlir` (here, target) |
|---|---|
| `torch_to_ue.lower()` | frontend/importer -> `.mlir` (keep) |
| `parse_mlir()` dict list | MLIR parser -> verified `ModuleOp` |
| `_analyze()` idiom recognizers | raising `OpRewritePattern`s (Step 3) |
| `if op == "matmul": ...` dispatch | backend lowering patterns + verifier |
| `alloc()` bump allocator + zeroing | MLIR bufferization framework |
| `for o in self.ops:` loop | `PassManager` driving passes to fixpoint |
| shape asserts / `ValueError` | op verifiers (Step 2), run automatically |
| `run()` FPGA emit | `ue-translate` (Step 5) |

The logic already exists and is proven on hardware. Porting = **moving each
hand-rolled rule into its typed MLIR slot above**, so a `PassManager` drives them
with verification between every stage — not teaching MLIR to understand `ue`.
