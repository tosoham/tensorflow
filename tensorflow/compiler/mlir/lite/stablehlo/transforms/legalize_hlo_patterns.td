/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This is the legalization pattern definition file for HLO to TF.

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Func/IR/FuncOps.td"
include "stablehlo/dialect/ChloOps.td"
include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td"
include "mhlo/IR/hlo_ops.td"

// Check if broadcasting is compatible with TF ops.
def IsLegalNumpyRankedBroadcast :
    Constraint<CPred<"hlo::isLegalNumpyRankedBroadcast($0, $1, $2 ? $2.asArrayRef() : ArrayRef<int64_t>{})">,
    "broadcasting should be compatible with TF ops">;

// Return a constant op that carries the shape of the given value.
def ShapeToConst : NativeCodeCall<"ShapeToConst($_builder, $0)">;

// Check if broadcast dimensions match Tensorflow convention.
def IsTFStyleBroadcast : Constraint<CPred<"IsTFStyleBroadcast($0, $1)">,
    "new dimensions are added as prefix">;

def IsNotWithinScatterRegion : Constraint<CPred<"!IsWithinOpRegion<mhlo::ScatterOp>($0)">,
    "binary ops within scatter regions are not converted">;

// Check if broadcast dimensions do not match Tensorflow convention.
def IsNotTFStyleBroadcast : Constraint<Neg<CPred<"IsTFStyleBroadcast($0, $1)">>,
    "new dimensions are inserted in intermediate positions">;

// Return intermediate shape before broadcasting, wrapped in a constant op.
def ExpandedShape : NativeCodeCall<"ExpandedShape($_builder, $0, $1, $2)">;
def ExpandedDynamicShape : NativeCodeCall<"ExpandedDynamicShape($_builder, $0, $1, $2)">;

def : Pat<(MHLO_ConstantOp:$output $value), (TF_ConstOp $value),
          [(TF_Tensor $output)]>;

//===----------------------------------------------------------------------===//
// Binary op patterns.
// Note that these are legalized from chlo.broadcast_* ops, since those are
// semantically compatible with the corresponding TF ops. Depending on
// context, getting to these ops may require some raising.
//===----------------------------------------------------------------------===//

foreach fromToBinPair = [[MHLO_AddOp, CHLO_BroadcastAddOp, TF_AddV2Op],
                         [MHLO_DivOp, CHLO_BroadcastDivOp, TF_DivOp],
                         [MHLO_ShiftLeftOp, CHLO_BroadcastShiftLeftOp, TF_LeftShiftOp],
                         [MHLO_MaxOp, CHLO_BroadcastMaxOp, TF_MaximumOp],
                         [MHLO_MinOp, CHLO_BroadcastMinOp, TF_MinimumOp],
                         [MHLO_MulOp, CHLO_BroadcastMulOp, TF_MulOp],
                         [MHLO_PowOp, CHLO_BroadcastPowOp, TF_PowOp],
                         [MHLO_SubtractOp, CHLO_BroadcastSubOp, TF_SubOp],
                         [MHLO_Atan2Op, CHLO_BroadcastAtan2Op, TF_Atan2Op]] in {
  def : Pat<(fromToBinPair[0]:$result $l, $r),
            (fromToBinPair[2] $l, $r),
            [(IsNotWithinScatterRegion $result)]>;
  def : Pat<(fromToBinPair[1] $l, $r, $broadcast_dimensions),
            (fromToBinPair[2] $l, $r),
            [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;
  def : Pat<(fromToBinPair[0] $l,
                              (MHLO_BroadcastInDimOp:$output
                                $bcast_operand,
                                $broadcast_dimensions)),
            (fromToBinPair[2] $l, $bcast_operand),
            [(IsTFStyleBroadcast $broadcast_dimensions, $output)]>;
  def : Pat<(fromToBinPair[0] (MHLO_BroadcastInDimOp:$output
                                $bcast_operand,
                                $broadcast_dimensions),
                              $r),
            (fromToBinPair[2] $bcast_operand, $r),
            [(IsTFStyleBroadcast $broadcast_dimensions, $output)]>;
}

foreach pair  = [[MHLO_AndOp, CHLO_BroadcastAndOp, TF_BitwiseAndOp],
                 [MHLO_OrOp, CHLO_BroadcastOrOp, TF_BitwiseOrOp],
                 [MHLO_XorOp, CHLO_BroadcastXorOp, TF_BitwiseXorOp]] in {
  def : Pat<(pair[0] TF_IntTensor:$l, TF_IntTensor:$r), (pair[2] $l, $r)>;
  def : Pat<(pair[1] TF_IntTensor:$l, TF_IntTensor:$r, $broadcast_dimensions),
            (pair[2] $l, $r),
            [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;
  def : Pat<(pair[0] (MHLO_BroadcastInDimOp:$output
                                            TF_IntTensor:$bcast_operand,
                                            $broadcast_dimensions),
                      TF_IntTensor:$r),
            (pair[2] $bcast_operand, $r),
            [(IsTFStyleBroadcast $broadcast_dimensions, $output)]>;
  def : Pat<(pair[0] TF_IntTensor:$l,
                     (MHLO_BroadcastInDimOp:$output
                                            TF_IntTensor:$bcast_operand,
                                            $broadcast_dimensions)),
            (pair[2] $l, $bcast_operand),
            [(IsTFStyleBroadcast $broadcast_dimensions, $output)]>;
}

foreach pair  = [[MHLO_AndOp, CHLO_BroadcastAndOp, TF_LogicalAndOp],
                 [MHLO_OrOp, CHLO_BroadcastOrOp, TF_LogicalOrOp]] in {
  def : Pat<(pair[0] I1Tensor:$l, I1Tensor:$r), (pair[2] $l, $r)>;
  def : Pat<(pair[1] I1Tensor:$l, I1Tensor:$r, $broadcast_dimensions),
            (pair[2] $l, $r),
            [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;
}

def : Pat<(MHLO_ShiftRightArithmeticOp $l, $r), (TF_RightShiftOp $l, $r)>;
def : Pat<(CHLO_BroadcastShiftRightArithmeticOp $l, $r,
                                                     $broadcast_dimensions),
          (TF_RightShiftOp $l, $r),
          [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;
def : Pat<(MHLO_ShiftRightLogicalOp $l, $r), (TF_RightShiftOp $l, $r)>;
def : Pat<(CHLO_BroadcastShiftRightLogicalOp $l, $r,
                                                  $broadcast_dimensions),
          (TF_RightShiftOp $l, $r),
          [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;

def : Pat<(MHLO_FloorOp (MHLO_DivOp $l, $r)), (TF_FloorDivOp $l, $r)>;
def : Pat<(MHLO_FloorOp (CHLO_BroadcastDivOp $l, $r,
                                                 $broadcast_dimensions)),
          (TF_FloorDivOp $l, $r),
          [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;

def : Pat<(MHLO_ComplexOp $r, $i), (TF_ComplexOp $r, $i)>;

def : Pat<(MHLO_RemOp TF_FpOrI32OrI64Tensor:$l, TF_FpOrI32OrI64Tensor:$r), (TF_ModOp $l, $r)>;
def : Pat<(CHLO_BroadcastRemOp TF_FpOrI32OrI64Tensor:$l, TF_FpOrI32OrI64Tensor:$r, $broadcast_dimensions),
          (TF_ModOp $l, $r),
          [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;

//===----------------------------------------------------------------------===//
// Unary op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(MHLO_ConvertOp MHLO_Tensor:$operand),
          (TF_CastOp $operand, ConstBoolAttrFalse)>;

foreach Mapping = [[MHLO_AbsOp, TF_AbsOp],
                   [MHLO_BitcastConvertOp, TF_BitcastOp],
                   [MHLO_CeilOp, TF_CeilOp],
                   [MHLO_FloorOp, TF_FloorOp],
                   [MHLO_ImagOp, TF_ImagOp],
                   [MHLO_IsFiniteOp, TF_IsFiniteOp],
                   [MHLO_NegOp, TF_NegOp],
                   [MHLO_RealOp, TF_RealOp],
                   [MHLO_SignOp, TF_SignOp]] in
  def : Pat<(Mapping[0] TF_IntOrFpTensor:$input), (Mapping[1] $input)>;

def ConstDefaultResultAccuracyAttr :
  ConstantAttr<MHLO_ResultAccuracyAttr, "::mlir::mhlo::ResultAccuracyMode::DEFAULT">;

foreach Mapping = [[MHLO_CosineOp, TF_CosOp],
                   [MHLO_Expm1Op, TF_Expm1Op],
                   [MHLO_ExpOp, TF_ExpOp],
                   [MHLO_LogOp, TF_LogOp],
                   [MHLO_Log1pOp, TF_Log1pOp],
                   [MHLO_LogisticOp, TF_SigmoidOp],
                   [MHLO_RsqrtOp, TF_RsqrtOp],
                   [MHLO_SineOp, TF_SinOp],
                   [MHLO_SqrtOp, TF_SqrtOp],
                   [MHLO_TanhOp, TF_TanhOp]] in {
  def : Pat<(Mapping[0] $input, ConstDefaultResultAccuracyAttr),
           (Mapping[1] MHLO_Tensor:$input)>;
}

class GetRankedScalarAttr<string prefix, int width, string signed, string value> :
  NativeCodeCall<"DenseElementsAttr::get<" # prefix # "int" # width # "_t>("
    "RankedTensorType::get({}, $_builder.getIntegerType("
      # width # signed # "))," # value # ")">;

def : Pat<(MHLO_NotOp TF_Int8Tensor:$input),
            (TF_BitwiseXorOp $input,
              (TF_ConstOp (GetRankedScalarAttr<"", 8, "", "-1">)))>;
def : Pat<(MHLO_NotOp TF_Int16Tensor:$input),
            (TF_BitwiseXorOp $input,
              (TF_ConstOp (GetRankedScalarAttr<"", 16, "", "-1">)))>;
def : Pat<(MHLO_NotOp TF_Int32Tensor:$input),
            (TF_BitwiseXorOp $input,
              (TF_ConstOp (GetRankedScalarAttr<"", 32, "", "-1">)))>;
def : Pat<(MHLO_NotOp TF_Int64Tensor:$input),
            (TF_BitwiseXorOp $input,
              (TF_ConstOp (GetRankedScalarAttr<"", 64, "", "-1">)))>;

def : Pat<(MHLO_NotOp TF_Uint8Tensor:$input),
            (TF_BitwiseXorOp $input,
              (TF_ConstOp (GetRankedScalarAttr<"u", 8, ", false", "0xFFU">)))>;
def : Pat<(MHLO_NotOp TF_Uint16Tensor:$input),
            (TF_BitwiseXorOp $input,
              (TF_ConstOp (GetRankedScalarAttr<"u", 16, ", false",
                                               "0xFFFFU">)))>;
def : Pat<(MHLO_NotOp TF_Uint32Tensor:$input),
            (TF_BitwiseXorOp $input,
              (TF_ConstOp (GetRankedScalarAttr<"u", 32, ", false",
                                               "0xFFFFFFFFUL">)))>;
def : Pat<(MHLO_NotOp TF_Uint64Tensor:$input),
            (TF_BitwiseXorOp $input,
              (TF_ConstOp (GetRankedScalarAttr<"u", 64, ", false",
                                               "0xFFFFFFFFFFFFFFFFULL">)))>;

def : Pat<(MHLO_NotOp TF_BoolTensor:$input), (TF_LogicalNotOp $input)>;
def : Pat<(MHLO_AbsOp TF_ComplexTensor:$arg), (TF_ComplexAbsOp $arg)>;

def : Pat<(MHLO_BroadcastOp $arg, $shape),
          (TF_BroadcastToOp $arg, (TF_ConstOp $shape))>;
def : Pat<(MHLO_BroadcastInDimOp:$output $input, $broadcast_dimensions),
          (TF_BroadcastToOp $input, (ShapeToConst $output)),
          [(IsTFStyleBroadcast $broadcast_dimensions, $output)]>;
def : Pat<(MHLO_BroadcastInDimOp:$output $input, $broadcast_dimensions),
          (TF_BroadcastToOp
              (TF_ReshapeOp
                  $input,
                  (ExpandedShape $input, $broadcast_dimensions, $output)),
              (ShapeToConst $output)),
          [(IsNotTFStyleBroadcast $broadcast_dimensions, $output)]>;
// Dynamism op
def : Pat<(MHLO_DynamicBroadcastInDimOp:$output $input, $output_dimensions,
               $broadcast_dimensions, $expanding_dimensions_unused, $nonexpanding_dimensions_unused),
          (TF_BroadcastToOp $input, $output_dimensions),
          [(IsTFStyleBroadcast $broadcast_dimensions, $output)]>;
def : Pat<(MHLO_DynamicBroadcastInDimOp:$output $input, $output_dimensions,
               $broadcast_dimensions, $expanding_dimensions_unused, $nonexpanding_dimensions_unused),
          (TF_BroadcastToOp (ExpandedDynamicShape $input, $broadcast_dimensions, $output), $output_dimensions),
          [(IsNotTFStyleBroadcast $broadcast_dimensions, $output)]>;


def : Pat<(MHLO_TransposeOp $arg, $permutation),
          (TF_TransposeOp $arg, (TF_ConstOp $permutation))>;
def : Pat<(MHLO_ReverseOp $op, $dims), (TF_ReverseV2Op $op, (TF_ConstOp $dims))>;
def : Pat<(MHLO_ReshapeOp:$output $input),
          (TF_ReshapeOp $input, (ShapeToConst $output))>;
def : Pat<(MHLO_DynamicReshapeOp $input, $shape),
          (TF_ReshapeOp $input, $shape)>;

// Both implement the Banker's rounding.
def : Pat<(MHLO_RoundNearestEvenOp $input), (TF_RoundOp $input)>;

//===----------------------------------------------------------------------===//
// Ternary op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(MHLO_ClampOp $min, $arg, $max),
          (TF_MaximumOp (TF_MinimumOp $arg, $max), $min)>;
def : Pat<(MHLO_SelectOp $cond, $t, $e), (TF_SelectOp $cond, $t, $e)>;
def : Pat<(MHLO_SelectOp (MHLO_BroadcastInDimOp:$output
                           $bcast_cond,
                           $broadcast_dimensions),
                          $t, $e),
          (TF_SelectV2Op $bcast_cond, $t, $e),
          [(IsTFStyleBroadcast $broadcast_dimensions, $output)]>;
def : Pat<(MHLO_SelectOp $cond, $t,
                                (MHLO_BroadcastInDimOp:$output
                                  $bcast_operand,
                                  $broadcast_dimensions)),
          (TF_SelectV2Op $cond, $t, $bcast_operand),
          [(IsTFStyleBroadcast $broadcast_dimensions, $output)]>;
def : Pat<(MHLO_SelectOp $cond, (MHLO_BroadcastInDimOp:$output
                                  $bcast_operand,
                                  $broadcast_dimensions),
                                $e),
          (TF_SelectV2Op $cond, $bcast_operand, $e),
          [(IsTFStyleBroadcast $broadcast_dimensions, $output)]>;

//===----------------------------------------------------------------------===//
// Variadic op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(MHLO_ConcatenateOp $inputs, $dim),
          (TF_ConcatV2Op $inputs, (TF_ConstOp $dim))>;

//===----------------------------------------------------------------------===//
// Compare op patterns.
// Note that these are legalized from chlo.broadcast_* ops, since those are
// semantically compatible with the corresponding TF ops. Depending on
// context, getting to these ops may require some raising.
//===----------------------------------------------------------------------===//

class HasChloCompareType<string value> :
    CPred<"llvm::cast<::mlir::chlo::ComparisonTypeAttr>($_self).getValue() == " # value>;

// Attribute value should be such that it matches the comparison used by
// TensorFlow, if the attribute is present.
def IsChloTFCompareType : AttrConstraint<
   Or<[CPred<"!$_self">, HasChloCompareType<"::mlir::chlo::ComparisonType::FLOAT">,
   HasChloCompareType<"::mlir::chlo::ComparisonType::SIGNED">,
   HasChloCompareType<"::mlir::chlo::ComparisonType::UNSIGNED">,
   HasChloCompareType<"::mlir::chlo::ComparisonType::NOTYPE">]>,
   "compare type supported by TensorFlow">;

class CHLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<CHLO_ComparisonDirectionAttr, "::mlir::chlo::ComparisonDirection::" # enumStr>;

class HasMhloCompareType<string value> :
    CPred<"llvm::cast<::mlir::mhlo::ComparisonTypeAttr>($_self).getValue() == " # value>;

// Attribute value should be such that it matches the comparison used by
// TensorFlow, if the attribute is present.
def IsMhloTFCompareType : AttrConstraint<
   Or<[CPred<"!$_self">, HasMhloCompareType<"::mlir::mhlo::ComparisonType::FLOAT">,
   HasMhloCompareType<"::mlir::mhlo::ComparisonType::SIGNED">,
   HasMhloCompareType<"::mlir::mhlo::ComparisonType::UNSIGNED">,
   HasMhloCompareType<"::mlir::mhlo::ComparisonType::NOTYPE">]>,
   "compare type supported by TensorFlow">;

class MHLO_ComparisonDirectionValue<string enumStr> :
  ConstantAttr<MHLO_ComparisonDirectionAttr, "::mlir::mhlo::ComparisonDirection::" # enumStr>;

foreach p = [[TF_EqualOp, CHLO_ComparisonDirectionValue<"EQ">],
             [TF_NotEqualOp, CHLO_ComparisonDirectionValue<"NE">]] in {
  def : Pat<(CHLO_BroadcastCompareOp $l, $r, $broadcast_dimensions, p[1],
                                          IsChloTFCompareType:$type),
            (p[0] $l, $r, ConstBoolAttrTrue),
            [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;
}

foreach p = [[TF_EqualOp, MHLO_ComparisonDirectionValue<"EQ">],
             [TF_NotEqualOp, MHLO_ComparisonDirectionValue<"NE">]] in {
  def : Pat<(MHLO_CompareOp $l, $r, p[1], IsMhloTFCompareType:$type),
            (p[0] $l, $r, ConstBoolAttrTrue)>;
  def : Pat<(MHLO_CompareOp (MHLO_BroadcastInDimOp:$output
                              $bcast_operand,
                              $broadcast_dimensions),
                            $r,
                            p[1],
                            IsMhloTFCompareType:$type),
            (p[0] $bcast_operand, $r, ConstBoolAttrTrue),
            [(IsTFStyleBroadcast $broadcast_dimensions, $output)]>;
  def : Pat<(MHLO_CompareOp $l,
                            (MHLO_BroadcastInDimOp:$output
                              $bcast_operand,
                              $broadcast_dimensions),
                            p[1],
                            IsMhloTFCompareType:$type),
            (p[0] $l, $bcast_operand, ConstBoolAttrTrue),
            [(IsTFStyleBroadcast $broadcast_dimensions, $output)]>;
}

foreach p = [[TF_GreaterEqualOp, CHLO_ComparisonDirectionValue<"GE">],
             [TF_GreaterOp, CHLO_ComparisonDirectionValue<"GT">],
             [TF_LessEqualOp, CHLO_ComparisonDirectionValue<"LE">],
             [TF_LessOp, CHLO_ComparisonDirectionValue<"LT">]] in {
  def : Pat<(CHLO_BroadcastCompareOp $l, $r, $broadcast_dimensions,
                                          p[1], IsChloTFCompareType:$type),
            (p[0] $l, $r),
            [(IsLegalNumpyRankedBroadcast $l, $r, $broadcast_dimensions)]>;
}

foreach p = [[TF_GreaterEqualOp, MHLO_ComparisonDirectionValue<"GE">],
             [TF_GreaterOp, MHLO_ComparisonDirectionValue<"GT">],
             [TF_LessEqualOp, MHLO_ComparisonDirectionValue<"LE">],
             [TF_LessOp, MHLO_ComparisonDirectionValue<"LT">]] in {
  def : Pat<(MHLO_CompareOp $l, $r, p[1], IsMhloTFCompareType:$type),
            (p[0] $l, $r)>;
  def : Pat<(MHLO_CompareOp (MHLO_BroadcastInDimOp:$output
                              $bcast_operand,
                              $broadcast_dimensions),
                            $r,
                            p[1],
                            IsMhloTFCompareType:$type),
            (p[0] $bcast_operand, $r),
            [(IsTFStyleBroadcast $broadcast_dimensions, $output)]>;
  def : Pat<(MHLO_CompareOp $l,
                            (MHLO_BroadcastInDimOp:$output
                              $bcast_operand,
                              $broadcast_dimensions),
                            p[1],
                            IsMhloTFCompareType:$type),
            (p[0] $l, $bcast_operand),
            [(IsTFStyleBroadcast $broadcast_dimensions, $output)]>;
}

def ConvertDotOp : NativeCodeCall<"ConvertDotOp($_builder, "
                                               "$0.getDefiningOp())">;
def : Pat<(MHLO_DotOp:$old_value StaticShapeTensorOf<[TF_ElementType]>:$lhs,
               StaticShapeTensorOf<[TF_ElementType]>:$rhs, $precision_config),
          (ConvertDotOp $old_value)>;

def ConvertDotGeneralOp : NativeCodeCall<"ConvertDotGeneralOp($_builder, "
                                               "$0.getDefiningOp())">;
def : Pat<(MHLO_DotGeneralOp:$old_value
               RankedTensorOf<[TF_ElementType]>:$lhs,
               RankedTensorOf<[TF_ElementType]>:$rhs,
               $dot_dimension_numbers, $precision_config, $algorithm),
          (ConvertDotGeneralOp $old_value)>;


def IsZero : Constraint<CPred<
  "$0.isSplat() && $0.getSplatValue<APInt>() == 0">>;
def ConvertPadOp : NativeCodeCall<
  "ConvertPadOp($_builder, $0.getDefiningOp())">;
def : Pat<(MHLO_PadOp:$old_value
               StaticShapeTensorOf<[TF_ElementType]>:$input,
               $pad_value, $pad_low, $pad_high, $pad_interior),
          (ConvertPadOp $old_value),
          [(IsZero $pad_interior)]>;

class ValueEquals<string val> : Constraint<CPred<"ValueEquals($0, " # val # ")">>;
def ValueGreaterThanZero : Constraint<CPred<"ValueGreaterThanZero($0)">>;
def ValueIsReciprocal : Constraint<CPred<"ValueIsReciprocal($0, $1)">>;
def TensorIsSign : Constraint<CPred<"TensorIsSign($_builder, $0, $1)">>;
def SameValue : Constraint<CPred<"$0 == $1">>;
def FloatOrDefaultCompare : Constraint<CPred<
  "!$0 || $0.getValue() == ::mlir::mhlo::ComparisonType::FLOAT">>;
def SameTypeOrDefaultCompare : Constraint<CPred<
  "SameTypeOrDefaultCompare($0, $1)">>;

// Converts a dag of HLOs representing banker rounding (round x.5 to nearest
// even) to tf.round. This only supports float types because mhlo.floor only
// supports float types. tf.round with integer input type will become an
// identity op, so we will never face an mhlo.floor with an integer input type.
// The pattern matched executes the following computation:
// frac = x - floor(x)
// to_even = (floor(x) - 2 * floor(0.5 * x)) == 1
// if frac > 0.5 || (frac == 0.5 && to_even)
//   return floor(x) + 1
// else
//   return floor(x)
def : Pat<(MHLO_SelectOp
            (MHLO_OrOp
              (MHLO_CompareOp (MHLO_SubtractOp:$frac
                               $input,
                               (MHLO_FloorOp:$floor $input)),
                             (MHLO_ConstantOp $half),
                             MHLO_ComparisonDirectionValue<"GT">,
                             $compare_type0),
              (MHLO_AndOp
                (MHLO_CompareOp
                  $frac1,
                  (MHLO_ConstantOp $half1),
                  MHLO_ComparisonDirectionValue<"EQ">,
                  $compare_type1),
                (MHLO_CompareOp
                  (MHLO_SubtractOp
                    $floor1,
                    (MHLO_MulOp
                      (MHLO_FloorOp (MHLO_MulOp $input, (MHLO_ConstantOp $half2))),
                      (MHLO_ConstantOp $two))),
                  (MHLO_ConstantOp $one1),
                  MHLO_ComparisonDirectionValue<"EQ">,
                  $compare_type2))),
            (MHLO_AddOp $floor2, (MHLO_ConstantOp $one)),
            $floor3),
          (TF_RoundOp $input),
          [(ValueEquals<"1.0"> $one),
           (ValueEquals<"1.0"> $one1),
           (ValueEquals<"2.0"> $two),
           (ValueEquals<"0.5"> $half),
           (ValueEquals<"0.5"> $half1),
           (ValueEquals<"0.5"> $half2),
           (SameValue $floor, $floor1),
           (SameValue $floor, $floor2),
           (SameValue $floor, $floor3),
           (SameValue $frac, $frac1),
           (FloatOrDefaultCompare $compare_type0),
           (FloatOrDefaultCompare $compare_type1),
           (FloatOrDefaultCompare $compare_type2)]>;

// Converts a dag of HLOs representing floor_mod to tf.FloorMod.
// The pattern matched executes the following computation:
//
// rem = remainder(arg0, arg1)
// for i in 0 to len(arg1):
//    if ((rem[i] < 0) != (arg0[i] < 0) && arg0[i] != 0)
//       rem[i] += arg1[i]
// return rem
def : Pat<(MHLO_SelectOp
            (MHLO_AndOp
              (MHLO_CompareOp
                (MHLO_CompareOp:$rltz
                  (MHLO_RemOp:$rem $arg, $arg1),
                  (MHLO_ConstantOp $cst),
                  MHLO_ComparisonDirectionValue<"LT">,
                  $compare_type),
                (MHLO_CompareOp:$arg1ltz $arg1, (MHLO_ConstantOp $cst1), MHLO_ComparisonDirectionValue<"LT">, $compare_type1),
                MHLO_ComparisonDirectionValue<"NE">,
                $compare_type2),
              (MHLO_CompareOp:$rnz $rem1, (MHLO_ConstantOp $cst2), MHLO_ComparisonDirectionValue<"NE">, $compare_type3)),
            (MHLO_AddOp $rem2, $arg1),
            $rem3),
          (TF_FloorModOp $arg, $arg1),
          [(ValueEquals<"0.0"> $cst),
           (ValueEquals<"0.0"> $cst1),
           (ValueEquals<"0.0"> $cst2),
           (SameValue $rem, $rem1),
           (SameValue $rem, $rem2),
           (SameValue $rem, $rem3),
           (SameTypeOrDefaultCompare $compare_type, $cst),
           (SameTypeOrDefaultCompare $compare_type1, $cst1)]>;

// Converts a dag of HLOs representing floor_mod with a constant to
// tf.FloorMod. The pattern matched executes the following computation:
//
// cst = value that is > 0
// rem = remainder(arg0, cst)
// for i in 0 to len(arg1):
//    if (rem[i] < 0 && rem[i] != 0)
//       rem[i] += cst
// return rem
def : Pat<(MHLO_SelectOp
            (MHLO_AndOp
              (MHLO_CompareOp:$rltz
                (MHLO_RemOp:$rem $arg, (MHLO_ConstantOp $cst)),
                (MHLO_ConstantOp $cst1),
                MHLO_ComparisonDirectionValue<"LT">,
                $compare_type),
              (MHLO_CompareOp:$rnz $rem1, (MHLO_ConstantOp $cst2), MHLO_ComparisonDirectionValue<"NE">, $compare_type3)),
            (MHLO_AddOp $rem2, (MHLO_ConstantOp $cst3)),
            $rem3),
          (TF_FloorModOp $arg, (TF_ConstOp $cst3)),
          [(ValueGreaterThanZero $cst),
           (ValueEquals<"0.0"> $cst1),
           (ValueEquals<"0.0"> $cst2),
           (SameValue $cst, $cst3),
           (SameValue $rem, $rem1),
           (SameValue $rem, $rem2),
           (SameValue $rem, $rem3),
           (SameTypeOrDefaultCompare $compare_type, $cst1),
           (SameTypeOrDefaultCompare $compare_type3, $cst2)]>;

// Converts a dag of HLOs representing floor_div to tf.FloorDiv.
// The pattern matched executes the following computation:
//
// rem = remainder(arg0, arg1)
// for i in 0 to len(arg1):
//    rem[i] = arg0[i] - rem[i] / arg1[i]
//    if (rem[i] != 0 && sign(arg1[i]) != sign(rem[i]))
//       rem[i] -= 1.0
// return round_nearest_afz(rem)
// As a dag this looks like the following:
//                       round
//                         |
//              -------- select
//              |          |    \
//             &&          +    div
//           /   |        / \
//        !=    !=       div -1
//       / |   / |      / |
//    rem 0.0 sn sn1    -  $1
//   / |      |  |    / |
// $0 $1     $1 rem  $0 rem
// Note that named operators like 'sn' and 'sn1' are different values produced by
// the same function in this case the sign function. Named values like 'div'
// refer to the same value produced by the same function, in this case division.
// Mathematical symbols do not indicate a re-use of the value.
def : Pat<(MHLO_RoundOp
            (MHLO_SelectOp
              (MHLO_AndOp
                (MHLO_CompareOp
                  (MHLO_RemOp:$rem $arg0, $arg1),
                  (MHLO_ConstantOp $cst),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (MHLO_CompareOp
                  (MHLO_SignOp $arg1),
                  (MHLO_SignOp $rem1),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (MHLO_AddOp
                (MHLO_DivOp:$div
                  (MHLO_SubtractOp $arg0, $rem2),
                  $arg1b),
                (MHLO_ConstantOp $cst_neg1)),
              $div1)),
          (TF_FloorDivOp $arg0, $arg1),
          [(ValueEquals<"0.0"> $cst),
           (ValueEquals<"-1.0"> $cst_neg1),
           (SameValue $div, $div1),
           (SameValue $rem, $rem1),
           (SameValue $rem, $rem2),
           (FloatOrDefaultCompare $compare_type, $cst),
           (FloatOrDefaultCompare $compare_type1, $cst)]>;

// Converts a dag of HLOs representing floor_div with a splat constant to
// tf.FloorDiv. The pattern matched executes the following computation:
// This particular pattern matches multiplication with the reciprocal of the
// constant instead of dividing by the constant.
// rem = remainder(arg0, cst)
// for i in 0 to len(arg0):
//    rem[i] = (arg0[i] - rem[i]) * 1 / cst
//    if (rem[i] != 0 && sign(cst) != sign(rem[i]))
//       rem[i] += -1.0
// return round_nearest_afz(rem)
// As a dag this looks like the following:
//                       round
//                         |
//              -------- select
//              |           |     \
//             &&           +     mul
//           /   |         / \
//        !=    !=        mul -1
//       / |   /  |       / |
//    rem 0.0 cs1 sn1   -  cs2
//   / |           |   / |
// $0 cst         rem $0 rem
// cs1 == sign(cst)
// cs2 = 1 / cst i.e. the reciprocal
// Note that named operators like 'sn' and 'sn1' are different values produced by
// the same function in this case the sign function. Named values like 'div'
// refer to the same value produced by the same function, in this case division.
// Mathematical symbols do not indicate a re-use of the value.
def : Pat<(MHLO_RoundOp
            (MHLO_SelectOp
              (MHLO_AndOp
                (MHLO_CompareOp
                  (MHLO_RemOp:$rem $arg0, (MHLO_ConstantOp:$cst $cstv)),
                  (MHLO_ConstantOp $cst_zero),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (MHLO_CompareOp
                  (MHLO_ConstantOp $cst_sgn),
                  (MHLO_SignOp $rem1),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (MHLO_AddOp
                (MHLO_MulOp:$mul
                  (MHLO_SubtractOp $arg0, $rem2),
                  (MHLO_ConstantOp $cst_recip)),
                (MHLO_ConstantOp $cst_neg1)),
              $mul1)),
          (TF_FloorDivOp $arg0, $cst),
          [(ValueEquals<"0.0"> $cst_zero),
           (ValueEquals<"-1.0"> $cst_neg1),
           (TensorIsSign $cstv, $cst_sgn),
           (ValueIsReciprocal $cstv, $cst_recip),
           (SameValue $mul, $mul1),
           (SameValue $rem, $rem1),
           (SameValue $rem, $rem2),
           (FloatOrDefaultCompare $compare_type, $cst_zero),
           (FloatOrDefaultCompare $compare_type1, $cst_sgn)]>;

// Converts a dag of HLOs representing floor_div with a splat constant to
// tf.FloorDiv. The pattern matched executes the following computation:
// This particular pattern matches division with the constant.
// .
// rem = remainder(arg0, cst)
// for i in 0 to len(arg0):
//    rem[i] = (arg0[i] - rem[i]) / cst
//    if (rem[i] != 0 && sign(cst) != sign(rem[i]))
//       rem[i] -= 1.0
// return round_nearest_afz(rem)
// As a dag this looks like the following:
//                       round
//                         |
//              -------- select
//              |           |     \
//             &&           +     div
//           /   |         / \
//        !=    !=        div -1
//       / |   /  |       / |
//    rem 0.0 cs1 sn1   -  cs2
//   / |           |   / |
// $0 cst         rem $0 rem
// cs1 == sign(cst)
// cs2 = 1 / cst i.e. the reciprocal
// Note that named operators like 'sn' and 'sn1' are different values produced by
// the same function in this case the sign function. Named values like 'div'
// refer to the same value produced by the same function, in this case division.
// Mathematical symbols do not indicate a re-use of the value.
def : Pat<(MHLO_RoundOp
            (MHLO_SelectOp
              (MHLO_AndOp
                (MHLO_CompareOp
                  (MHLO_RemOp:$rem $arg0, (MHLO_ConstantOp:$cst $cstv)),
                  (MHLO_ConstantOp $cst_zero),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (MHLO_CompareOp
                  (MHLO_ConstantOp $cst_sgn),
                  (MHLO_SignOp $rem1),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (MHLO_AddOp
                (MHLO_DivOp:$div
                  (MHLO_SubtractOp $arg0, $rem2),
                  (MHLO_ConstantOp $cstv1)),
                (MHLO_ConstantOp $cst_neg1)),
              $div1)),
          (TF_FloorDivOp $arg0, $cst),
          [(ValueEquals<"0.0"> $cst_zero),
           (ValueEquals<"-1.0"> $cst_neg1),
           (TensorIsSign $cstv, $cst_sgn),
           (SameValue $div, $div1),
           (SameValue $rem, $rem1),
           (SameValue $rem, $rem2),
           (SameValue $cstv1, $cstv),
           (FloatOrDefaultCompare $compare_type, $cst_zero),
           (FloatOrDefaultCompare $compare_type1, $cst_sgn)]>;

// Converts a dag of HLOs representing floor_div with a broadcasted vector
// constant to tf.FloorDiv. The pattern matched executes the following
// computation:
// scs = sign(cst)
// bcst = broadcast(cst)
// rem = remainder(arg0, bcst)
// for i in 0 to len(arg0):
//    rem[i] = arg0[i] - rem[i] * / bcst
//    if (rem[i] != 0 && scs != sign(rem[i]))
//       rem[i] -= 1.0
// return round_nearest_afz(rem)
// Where scs is a splat constant folded sign on the unbroadcasted tensor.
//
// As a dag this looks like the following:
//                       round
//                         |
//              -------- select
//              |           |     \
//             &&           +     div
//           /   |         / \
//        !=    !=        div -1
//       / |   /  |       / |
//    rem 0.0 scs sn1   -  bcst
//   / |           |   / |
// $0 bcst        rem $0 rem
//     |
//    cst
// scs == sign(cst) == sign(bcst)
// Note that named operators like 'sn' and 'sn1' are different values produced by
// the same function in this case the sign function. Named values like 'div'
// refer to the same value produced by the same function, in this case division.
// Mathematical symbols do not indicate a re-use of the value.
def : Pat<(MHLO_RoundOp
            (MHLO_SelectOp
              (MHLO_AndOp
                (MHLO_CompareOp
                  (MHLO_RemOp:$rem $arg0,
                    (MHLO_BroadcastInDimOp:$bcst
                      (MHLO_ConstantOp $cstv),
                      $broadcast_dimension)),
                  (MHLO_ConstantOp $cst_zero),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type),
                (MHLO_CompareOp
                  (MHLO_ConstantOp $cst_sgn),
                  (MHLO_SignOp $rem1),
                  MHLO_ComparisonDirectionValue<"NE">,
                  $compare_type1)),
              (MHLO_AddOp
                (MHLO_DivOp:$div
                  (MHLO_SubtractOp $arg0, $rem2),
                  $bcst1),
                (MHLO_ConstantOp $cst_neg1)),
              $div1)),
          (TF_FloorDivOp $arg0, $bcst),
          [(ValueEquals<"0.0"> $cst_zero),
           (ValueEquals<"-1.0"> $cst_neg1),
           (TensorIsSign $cstv, $cst_sgn),
           (SameValue $bcst, $bcst1),
           (SameValue $div, $div1),
           (SameValue $rem, $rem1),
           (SameValue $rem, $rem2),
           (FloatOrDefaultCompare $compare_type, $cst_zero),
           (FloatOrDefaultCompare $compare_type1, $cst_sgn)]>;

//===----------------------------------------------------------------------===//
// TorchIndexSelect op patterns.
//===----------------------------------------------------------------------===//

def : Pat<(MHLO_TorchIndexSelectOp $params, $indices, $axis, $batch_dims),
          (TF_GatherV2Op $params, $indices, (TF_ConstOp $axis), $batch_dims)>;

