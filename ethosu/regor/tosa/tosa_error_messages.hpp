//
// SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the License); you may
// not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

static const char *ERRORIF_ArgmaxOutputRankMismatch = "Mismatch between output shape provided and expected output shape";
static const char *ERRORIF_ArgmaxOutputShapeMismatch = "Mismatch between output shape provided and expected output shape";
static const char *ERRORIF_AxisLargerRank = "Axis larger than rank";
static const char *ERRORIF_AxisSmallerZero = "Axis smaller than zero";
static const char *ERRORIF_BatchMismatch = "Input batch size not equal to output batch size";
static const char *ERRORIF_BorderLargerEqualMax = "Border value larger than or equal to maximum value";
static const char *ERRORIF_BorderSmallerMin = "Border value smaller than minimum value";
static const char *ERRORIF_BroadcastShapesMismatch = "Broadcast shape calculation failed";
static const char *ERRORIF_ChannelMismatch = "Input channel size not equal to output channel size";
static const char *ERRORIF_ConcatInputDimMismatch = "Input dimensions differ on too many axes";
static const char *ERRORIF_ConcatInputRankMismatch = "Input ranks are not identical";
static const char *ERRORIF_ConcatNoInputList = "Concat is given an empty list of inputs";
static const char *ERRORIF_ConcatShapeSumMismatch = "Sum of dimensions on axis not equal to output dimension";
static const char *ERRORIF_CondGraphOutputNotMatchingBool = "Cond graph output is not a match list of booleans";
static const char *ERRORIF_CondGraphOutputShapeNotSizeOne = "Cond graph output is not a shape of size one";
static const char *ERRORIF_CondIfCondNotMatchingBool = "Conditional tensor does not match bool type";
static const char *ERRORIF_CondIfCondShapeNotSizeOne = "Conditional tensor is not equal to a size of one";
static const char *ERRORIF_CondIfInputListElseGraphMismatch = "Input list shape does not match else-graph shape";
static const char *ERRORIF_CondIfInputListThenGraphMismatch = "Input list shape does not match then-graph shape";
static const char *ERRORIF_CondIfOutputListElseGraphMismatch = "Output list shape does not match else-graph shape";
static const char *ERRORIF_CondIfOutputListThenGraphMismatch = "Output list shape does not match then-graph shape";
static const char *ERRORIF_ConvBiasShapeMismatch = "Convolution bias shape mismatch";
static const char *ERRORIF_ConvOutputShapeMismatch = "Mismatch between output shape provided and expected output shape";
static const char *ERRORIF_ConvOutputShapeNonInteger = "Parameters do not yield exact integer output dimensions";
static const char *ERRORIF_DilationSmallerOne = "At least one dilation is smaller than one";
static const char *ERRORIF_DimensionMismatch = "Input Dimensions do not match output";
static const char *ERRORIF_IndexOutsideBounds = "Index outside of allowed bounds";
static const char *ERRORIF_IndexUsedTwice = "Index used multiple times";
static const char *ERRORIF_InputListBodyGraphInputMismatch = "Input list does not match body graph input";
static const char *ERRORIF_InputListBodyGraphOutputMismatch = "Input list does not match body graph output";
static const char *ERRORIF_InputListCondGraphMismatch = "Input list does not match cond graph";
static const char *ERRORIF_InputListOutputListMismatch = "Input list does not match output list";
static const char *ERRORIF_InputRank1WrongRank = "Unsupported input rank when rank 1 expected";
static const char *ERRORIF_InputSizeStartLengthMismatch = "rank of input not equal to length of start or size";
static const char *ERRORIF_InputZeroPointNotZero = "Input DType not INT8 and zero point not 0";
static const char *ERRORIF_KernelSmallerOne = "At least one kernel dimension is smaller than one";
static const char *ERRORIF_MaxDimExceeded = "At least one maximum dimension is greater than or equal to 16384";
static const char *ERRORIF_MaxSmallerMin = "Max value smaller than min value";
static const char *ERRORIF_OffsetLargerEqualMax = "Offset value larger than or equal to maximum value";
static const char *ERRORIF_OffsetSmallerMin = "Offset value smaller than minimum value";
static const char *ERRORIF_OutputZeroPointNotZero = "Output DType not INT8 and zero point not 0";
static const char *ERRORIF_PadLargerEqualKernel = "At least one pad is larger than kernel dimension";
static const char *ERRORIF_PadOutputShapeMismatch = "Pad output shape mismatch for requested padding";
static const char *ERRORIF_PadSmallerZero = "At least one pad is smaller than zero";
static const char *ERRORIF_PoolingOutputShapeMismatch = "Mismatch between output shape provided and expected output shape";
static const char *ERRORIF_PoolingOutputShapeNonInteger = "Parameters do not yield exact integer output dimensions";
static const char *ERRORIF_RankMismatch = "Input Rank does not match output rank";
static const char *ERRORIF_RescaleI32InputOutputUnsigned = "INT32 input and output_unsigned set true";
static const char *ERRORIF_RescaleI32InputUnsigned = "INT32 input and input_unsigned set true";
static const char *ERRORIF_RescaleI32OutputInputUnsigned = "INT32 output and input_unsigned set true";
static const char *ERRORIF_RescaleI32OutputUnsigned = "INT32 output and output_unsigned set true";
static const char *ERRORIF_RescaleI48InputOutputUnsigned = "INT48 input and output_unsigned set true";
static const char *ERRORIF_RescaleI48InputUnsigned = "INT48 input and input_unsigned set true";
static const char *ERRORIF_RescaleInputUnsignedOutputUnsigned = "both input_unsigned and output_unsigned set true";
static const char *ERRORIF_RescalePerChannelRank0 = "Per channel set for rank 0";
static const char *ERRORIF_ResizeOutputShapeMismatch = "Mismatch between output shape provided and expected output shape";
static const char *ERRORIF_ResizeOutputShapeNonInteger = "Parameters do not yield exact integer output dimensions";
static const char *ERRORIF_ScaleDLargerMax = "Scale D value larger than maximum value";
static const char *ERRORIF_ScaleNLargerMax = "Scale N value larger than maximum value";
static const char *ERRORIF_ScaleNotTrue = "Scale set to false but double round set to true";
static const char *ERRORIF_ScaleSmallerEqualZero = "Scale value smaller than or equal to zero";
static const char *ERRORIF_ScaleTrue = "Scale set to true but input type is INT48";
static const char *ERRORIF_ShapeOfAxisNotOne = "shape[axis] is not equal to 1";
static const char *ERRORIF_SizeOutputShapeMismatch = "Size does not match output dimension";
static const char *ERRORIF_SizeSmallerEqualZero = "Size smaller than or equal to zero";
static const char *ERRORIF_StartSizeOutsideBounds = "starting point plus size larger than input dimension";
static const char *ERRORIF_StartSmallerZero = "Starting point smaller than zero";
static const char *ERRORIF_StrideSmallerOne = "At least one stride dimension is smaller than one";
static const char *ERRORIF_TensorSizeInputOutputMismatch = "Input tensor size does not match output tensor size";
static const char *ERRORIF_TileMultiplesOutputShapeMismatch = "Multiples do not match output shape";
static const char *ERRORIF_TransposePermsOutputShapeMismatch = "Permutations do not match output shape";
static const char *ERRORIF_U16InputZeroPointNotValid = "Input DType is UINT16 and zero point not 0 or 32768";
static const char *ERRORIF_U16OutputZeroPointNotValid = "Output DType is UINT16 and zero point not 0 or 32768";
static const char *ERRORIF_WeightZeroPointNotZero = "Weight DType not INT8 and zero point not 0";
static const char *ERRORIF_WrongAccumulatorType = "An unsupported accumulator data type was requested";
static const char *ERRORIF_WrongBiasType = "An unsupported bias data type was requested";
static const char *ERRORIF_WrongInputList = "Op input list does not match expected input";
static const char *ERRORIF_WrongInputType = "Input data type not supported for this operator";
static const char *ERRORIF_WrongOutputList = "Op output list does not match expected output";
static const char *ERRORIF_WrongOutputType = "Output data type not supported for this configuration of operator";
static const char *ERRORIF_WrongRank = "Rank not supported for this operator";
