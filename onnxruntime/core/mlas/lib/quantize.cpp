/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    quantize.cpp

Abstract:

    This module implements routines to quantize buffers.

    For quantization formula as specified in the ONNX operator documentation is:

        Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)

--*/
#include "mlasi.h"
#include <cmath>

#if defined(MLAS_NEON64_INTRINSICS) || defined(MLAS_SSE2_INTRINSICS)

//
// QuantizeLinear implementation using NEON or SSE2 intrinsics.
//

MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearVector(
    MLAS_FLOAT32X4 FloatVector,
    MLAS_FLOAT32X4 ScaleVector,
    MLAS_FLOAT32X4 MinimumValueVector,
    MLAS_FLOAT32X4 MaximumValueVector,
    MLAS_INT32X4 ZeroPointVector
    )
{
    //
    // Scale the input vector and clamp the values to the minimum and maximum
    // range (adjusted by the zero point value).
    //

    FloatVector = MlasDivideFloat32x4(FloatVector, ScaleVector);

#if defined(MLAS_NEON64_INTRINSICS)
    // N.B. FMINNM and FMAXNM returns the numeric value if either of the values
    // is a NaN.
    FloatVector = vmaxnmq_f32(FloatVector, MinimumValueVector);
    FloatVector = vminnmq_f32(FloatVector, MaximumValueVector);
#else
    // N.B. MINPS and MAXPS returns the value from the second vector if the
    // value from the first vector is a NaN.
    FloatVector = _mm_max_ps(FloatVector, MinimumValueVector);
    FloatVector = _mm_min_ps(FloatVector, MaximumValueVector);
#endif

    //
    // Convert the float values to integer using "round to nearest even" and
    // then shift the output range using the zero point value.
    //

#if defined(MLAS_NEON64_INTRINSICS)
    auto IntegerVector = vcvtnq_s32_f32(FloatVector);
    IntegerVector = vaddq_s32(IntegerVector, ZeroPointVector);
#else
    // N.B. Assumes MXCSR has been configured with the default rounding mode of
    // "round to nearest even".
    auto IntegerVector = _mm_cvtps_epi32(FloatVector);
    IntegerVector = _mm_add_epi32(IntegerVector, ZeroPointVector);
#endif

    return IntegerVector;
}

template<typename OutputType>
MLAS_INT32X4
MlasQuantizeLinearPackBytes(
    MLAS_INT32X4 IntegerVector
    );

#if defined(MLAS_NEON64_INTRINSICS)

template<typename OutputType>
MLAS_INT32X4
MlasQuantizeLinearPackBytes(
    MLAS_INT32X4 IntegerVector
    )
{
    //
    // Swizzle the least significant byte from each int32_t element to the
    // bottom four bytes of the vector register.
    //

    uint16x8_t WordVector = vreinterpretq_u16_s32(IntegerVector);
    WordVector = vuzp1q_u16(WordVector, WordVector);
    uint8x16_t ByteVector = vreinterpretq_u8_u16(WordVector);
    ByteVector = vuzp1q_u8(ByteVector, ByteVector);

    return vreinterpretq_s32_u8(ByteVector);
}

#else

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearPackBytes<uint8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
    IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);
    IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);

    return IntegerVector;
}

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearPackBytes<int8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
    IntegerVector = _mm_packs_epi16(IntegerVector, IntegerVector);
    IntegerVector = _mm_packs_epi16(IntegerVector, IntegerVector);

    return IntegerVector;
}

#endif

template<typename OutputType>
void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    )
/*++

Routine Description:

    This routine quantizes the input buffer using the supplied quantization
    parameters.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

    auto ScaleVector = MlasBroadcastFloat32x4(Scale);
    auto MinimumValueVector = MlasBroadcastFloat32x4(float(MinimumValue - ZeroPoint));
    auto MaximumValueVector = MlasBroadcastFloat32x4(float(MaximumValue - ZeroPoint));
    auto ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);

    while (N >= 4) {

        auto FloatVector = MlasLoadFloat32x4(Input);
        auto IntegerVector = MlasQuantizeLinearVector(FloatVector, ScaleVector,
            MinimumValueVector, MaximumValueVector, ZeroPointVector);

        IntegerVector = MlasQuantizeLinearPackBytes<OutputType>(IntegerVector);

#if defined(MLAS_NEON64_INTRINSICS)
        vst1q_lane_s32((int32_t*)Output, IntegerVector, 0);
#else
        *((int32_t*)Output) = _mm_cvtsi128_si32(IntegerVector);
#endif

        Input += 4;
        Output += 4;
        N -= 4;
    }

    for (size_t n = 0; n < N; n++) {

#if defined(MLAS_NEON64_INTRINSICS)
        auto FloatVector = vld1q_dup_f32(Input + n);
#else
        auto FloatVector = _mm_load_ss(Input + n);
#endif
        auto IntegerVector = MlasQuantizeLinearVector(FloatVector, ScaleVector,
            MinimumValueVector, MaximumValueVector, ZeroPointVector);

#if defined(MLAS_NEON64_INTRINSICS)
        vst1q_lane_u8((uint8_t*)Output + n, vreinterpretq_u8_s32(IntegerVector), 0);
#else
        *((uint8_t*)Output + n) = (uint8_t)_mm_cvtsi128_si32(IntegerVector);
#endif
    }
}

MLAS_FORCEINLINE
MLAS_FLOAT32X4
MlasDequantizeLinearVector(
    MLAS_INT32X4 IntVector,
    MLAS_FLOAT32X4 ScaleVector,
    MLAS_INT32X4 ZeroPointVector
    )
{
#if defined(MLAS_NEON64_INTRINSICS)
    return MlasMultiplyFloat32x4(vcvtq_f32_s32(vsubq_s32(IntVector, ZeroPointVector)), ScaleVector);
#else
    return MlasMultiplyFloat32x4(_mm_cvtepi32_ps(_mm_sub_epi32(IntVector, ZeroPointVector)), ScaleVector);
#endif
}

template<typename DataType>
MLAS_INT32X4
MlasQuantizeLinearUnpackBytes(
    MLAS_INT32X4 IntegerVector
    );

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearUnpackBytes<uint8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
#if defined(MLAS_NEON64_INTRINSICS)
    uint16x8_t vl = vmovl_u8(vget_low_u8(vreinterpretq_u8_s32(IntegerVector)));
    IntegerVector = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vl)));
#else
    IntegerVector = _mm_unpacklo_epi8(IntegerVector, IntegerVector);
    IntegerVector = _mm_unpacklo_epi16(IntegerVector, IntegerVector);
    IntegerVector = _mm_srli_epi32(IntegerVector, 24);
#endif
    return IntegerVector;
}

template<>
MLAS_FORCEINLINE
MLAS_INT32X4
MlasQuantizeLinearUnpackBytes<int8_t>(
    MLAS_INT32X4 IntegerVector
    )
{
#if defined(MLAS_NEON64_INTRINSICS)
    int16x8_t vl = vmovl_s8(vget_low_s8(vreinterpretq_s8_s32(IntegerVector)));
    IntegerVector = vmovl_s16(vget_low_s16(vl));
#else
    IntegerVector = _mm_unpacklo_epi8(IntegerVector, IntegerVector);
    IntegerVector = _mm_unpacklo_epi16(IntegerVector, IntegerVector);
    IntegerVector = _mm_srai_epi32(IntegerVector, 24);
#endif
    return IntegerVector;
}

template<typename DataType>
void
MLASCALL
MlasQLinearAddKernel(
    const DataType* InputA,
    float ScaleA,
    DataType ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    DataType ZeroPointB,
    float ScaleC,
    DataType ZeroPointC,
    DataType* OutputC,
    size_t LengthA,
    size_t LengthB
    )
{
    constexpr int32_t MinimumValue = std::numeric_limits<DataType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<DataType>::max();

    const auto ScaleVectorA = MlasBroadcastFloat32x4(ScaleA);
    const auto ScaleVectorB = MlasBroadcastFloat32x4(ScaleB);
    const auto ScaleVectorC = MlasBroadcastFloat32x4(ScaleC);
    const auto ZeroPointVectorA = MlasBroadcastInt32x4((int32_t)ZeroPointA);
    const auto ZeroPointVectorB = MlasBroadcastInt32x4((int32_t)ZeroPointB);
    const auto ZeroPointVectorC = MlasBroadcastInt32x4((int32_t)ZeroPointC);
    const auto MinimumValueVectorC = MlasBroadcastFloat32x4(float(MinimumValue - ZeroPointC));
    const auto MaximumValueVectorC = MlasBroadcastFloat32x4(float(MaximumValue - ZeroPointC));

    size_t N = std::max(LengthA, LengthB);
    if (LengthA == 1) {
        auto IntegerVectorA = MlasBroadcastInt32x4((int32_t)*InputA));
        auto FloatVectorA = MlasDequantizeLinearVector(IntegerVectorA, ScaleVectorA, ZeroPointVectorA);
        while (N >= 4) {
            auto IntegerVectorB = MlasQuantizeLinearUnpackBytes<DataType>(MlasBroadcastInt32x4(*((const int32_t*)InputB)));
            auto FloatVectorC = MlasAddFloat32x4(
                    FloatVectorA,
                    MlasDequantizeLinearVector(IntegerVectorB, ScaleVectorB, ZeroPointVectorB));
            auto IntegerVectorC = MlasQuantizeLinearVector(FloatVectorC, ScaleVectorC,
                    MinimumValueVectorC, MaximumValueVectorC, ZeroPointVectorC);
            IntegerVectorC = MlasQuantizeLinearPackBytes<DataType>(IntegerVectorC);

#if defined(MLAS_NEON64_INTRINSICS)
            vst1q_lane_s32((int32_t*)OutputC, IntegerVectorC, 0);
#else
            *((int32_t*)OutputC) = _mm_cvtsi128_si32(IntegerVectorC);
#endif

            InputB += 4;
            OutputC += 4;
            N -= 4;
        }
    }
    else if (LengthB == 1) {
        auto IntegerVectorB = MlasBroadcastInt32x4((int32_t)*InputB));
        auto FloatVectorB = MlasDequantizeLinearVector(IntegerVectorB, ScaleVectorB, ZeroPointVectorB);
        while (N >= 4) {
            auto IntegerVectorA = MlasQuantizeLinearUnpackBytes<DataType>(MlasBroadcastInt32x4(*((const int32_t*)InputA)));
            auto FloatVectorC = MlasAddFloat32x4(
                    MlasDequantizeLinearVector(IntegerVectorA, ScaleVectorA, ZeroPointVectorA),
                    FloatVectorB);
            auto IntegerVectorC = MlasQuantizeLinearVector(FloatVectorC, ScaleVectorC,
                    MinimumValueVectorC, MaximumValueVectorC, ZeroPointVectorC);
            IntegerVectorC = MlasQuantizeLinearPackBytes<DataType>(IntegerVectorC);

#if defined(MLAS_NEON64_INTRINSICS)
            vst1q_lane_s32((int32_t*)OutputC, IntegerVectorC, 0);
#else
            *((int32_t*)OutputC) = _mm_cvtsi128_si32(IntegerVectorC);
#endif

            InputA += 4;
            OutputC += 4;
            N -= 4;
        }
    }
    else {
        while (N >= 4) {
            auto IntegerVectorA = MlasQuantizeLinearUnpackBytes<DataType>(MlasBroadcastInt32x4(*((const int32_t*)InputA)));
            auto IntegerVectorB = MlasQuantizeLinearUnpackBytes<DataType>(MlasBroadcastInt32x4(*((const int32_t*)InputB)));
            auto FloatVectorC = MlasAddFloat32x4(
                    MlasDequantizeLinearVector(IntegerVectorA, ScaleVectorA, ZeroPointVectorA),
                    MlasDequantizeLinearVector(IntegerVectorB, ScaleVectorB, ZeroPointVectorB));
            auto IntegerVectorC = MlasQuantizeLinearVector(FloatVectorC, ScaleVectorC,
                    MinimumValueVectorC, MaximumValueVectorC, ZeroPointVectorC);
            IntegerVectorC = MlasQuantizeLinearPackBytes<DataType>(IntegerVectorC);

#if defined(MLAS_NEON64_INTRINSICS)
            vst1q_lane_s32((int32_t*)OutputC, IntegerVectorC, 0);
#else
            *((int32_t*)OutputC) = _mm_cvtsi128_si32(IntegerVectorC);
#endif

            InputA += 4;
            InputB += 4;
            OutputC += 4;
            N -= 4;
        }

    }

    if (N > 0) {
        auto IntegerVectorA = (LengthA == 1) ?
                MlasBroadcastInt32x4((int32_t)*InputA)) :
                MlasQuantizeLinearUnpackBytes<DataType>(MlasBroadcastInt32x4(*((const int32_t*)InputA)));
        auto IntegerVectorB = (LengthB == 1) ?
                MlasBroadcastInt32x4((int32_t)*InputB)) :
                MlasQuantizeLinearUnpackBytes<DataType>(MlasBroadcastInt32x4(*((const int32_t*)InputB)));
        auto FloatVectorC = MlasAddFloat32x4(
                MlasDequantizeLinearVector(IntegerVectorA, ScaleVectorA, ZeroPointVectorA),
                MlasDequantizeLinearVector(IntegerVectorB, ScaleVectorB, ZeroPointVectorB));
        auto IntegerVectorC = MlasQuantizeLinearVector(FloatVectorC, ScaleVectorC,
                MinimumValueVectorC, MaximumValueVectorC, ZeroPointVectorC);
        IntegerVectorC = MlasQuantizeLinearPackBytes<DataType>(IntegerVectorC);

        uint32_t PackedValueC = 0;
#if defined(MLAS_NEON64_INTRINSICS)
        vst1q_lane_s32((int32_t*)&PackedValueC, IntegerVectorC, 0);
#else
        *((int32_t*)&PackedValueC) = _mm_cvtsi128_si32(IntegerVectorC);
#endif

        for (size_t n = 0; n < N; ++n) {
            *((uint8_t*)OutputC + n) = (uint8_t)PackedValueC;
            PackedValueC >>= 8;
        }
    }
}

#else

//
// QuantizeLinear implementation using the C++ runtime.
//
template<typename OutputType, int32_t MinimumValue, int32_t MaximumValue>
MLAS_FORCEINLINE
void
MlasQuantizeLinearKernel(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    int32_t ZeroPoint
    )
{
    for (size_t n = 0; n < N; n++) {

        float FloatValue = std::nearbyintf(Input[n] / Scale) + float(ZeroPoint);
        FloatValue = std::max(FloatValue, float(MinimumValue));
        FloatValue = std::min(FloatValue, float(MaximumValue));
        Output[n] = (OutputType)(int32_t)FloatValue;
    }
}


template<typename OutputType>
void
MLASCALL
MlasQuantizeLinear(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint
    )
/*++

Routine Description:

    This routine quantizes the input buffer using the supplied quantization
    parameters.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

    for (size_t n = 0; n < N; n++) {

        float FloatValue = std::nearbyintf(Input[n] / Scale) + float(ZeroPoint);
        FloatValue = std::max(FloatValue, float(MinimumValue));
        FloatValue = std::min(FloatValue, float(MaximumValue));
        Output[n] = (OutputType)(int32_t)FloatValue;
    }
}


template<typename DataType>
void
MLASCALL
MlasQLinearAddKernel(
    const DataType* InputA,
    float ScaleA,
    DataType ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    DataType ZeroPointB,
    float ScaleC,
    DataType ZeroPointC,
    DataType* OutputC,
    size_t LengthA,
    size_t LengthB
    )
{
    constexpr int32_t MinimumValue = std::numeric_limits<DataType>::min();
    constexpr int32_t MaximumValue = std::numeric_limits<DataType>::max();

    size_t N = std::max(LengthA, LengthB);
    if (LengthA == 1) {
        float ValueA = ScaleA * (int(InputA[0]) - int(ZeroPointA));
        for (size_t n = 0; n < N; n++) {
            float ValueB = ScaleB * (int(InputB[n]) - int(ZeroPointB));
            int32_t IntValueC = (int32_t)std::nearbyintf((ValueA + ValueB) / ScaleC) + int32_t(ZeroPointC);
            IntValueC = std::max(IntValueC, MinimumValue);
            IntValueC = std::min(IntValueC, MaximumValue);
            OutputC[n] = (DataType)IntValueC;
        }
    }
    else if (LengthB == 1) {
        float ValueB = ScaleB * (int(InputB[0]) - int(ZeroPointB));
        for (size_t n = 0; n < N; n++) {
            float ValueA = ScaleA * (int(InputA[n]) - int(ZeroPointA));
            int32_t IntValueC = (int32_t)std::nearbyintf((ValueA + ValueB) / ScaleC) + int32_t(ZeroPointC);
            IntValueC = std::max(IntValueC, MinimumValue);
            IntValueC = std::min(IntValueC, MaximumValue);
            OutputC[n] = (DataType)IntValueC;
        }
    }
    else {
        for (size_t n = 0; n < N; n++) {
            float ValueA = ScaleA * (int(InputA[n]) - int(ZeroPointA));
            float ValueB = ScaleB * (int(InputB[n]) - int(ZeroPointB));
            int32_t IntValueC = (int32_t)std::nearbyintf((ValueA + ValueB) / ScaleC) + int32_t(ZeroPointC);
            IntValueC = std::max(IntValueC, MinimumValue);
            IntValueC = std::min(IntValueC, MaximumValue);
            OutputC[n] = (DataType)IntValueC;
        }
    }
}

#endif

template
void
MLASCALL
MlasQuantizeLinear<int8_t>(
    const float* Input,
    int8_t* Output,
    size_t N,
    float Scale,
    int8_t ZeroPoint
    );


template
void
MLASCALL
MlasQuantizeLinear<uint8_t>(
    const float* Input,
    uint8_t* Output,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    );


#if defined(MLAS_SSE2_INTRINSICS)

MLAS_FORCEINLINE
MLAS_INT32X4
MlasRequantizeOutputVector(
    MLAS_INT32X4 IntegerVector,
    MLAS_INT32X4 BiasVector,
    MLAS_FLOAT32X4 ScaleVector,
    MLAS_FLOAT32X4 MinimumValueVector,
    MLAS_FLOAT32X4 MaximumValueVector,
    MLAS_INT32X4 ZeroPointVector
    )
{
    IntegerVector = _mm_add_epi32(IntegerVector, BiasVector);
    MLAS_FLOAT32X4 FloatVector = _mm_cvtepi32_ps(IntegerVector);

    //
    // Scale the input vector and clamp the values to the minimum and maximum
    // range (adjusted by the zero point value).
    //

    FloatVector = MlasMultiplyFloat32x4(FloatVector, ScaleVector);

    // N.B. MINPS and MAXPS returns the value from the second vector if the
    // value from the first vector is a NaN.
    FloatVector = _mm_max_ps(FloatVector, MinimumValueVector);
    FloatVector = _mm_min_ps(FloatVector, MaximumValueVector);

    //
    // Convert the float values to integer using "round to nearest even" and
    // then shift the output range using the zero point value.
    //

    // N.B. Assumes MXCSR has been configured with the default rounding mode of
    // "round to nearest even".
    IntegerVector = _mm_cvtps_epi32(FloatVector);
    IntegerVector = _mm_add_epi32(IntegerVector, ZeroPointVector);

    return IntegerVector;
}

void
MLASCALL
MlasRequantizeOutput(
    const int32_t* Input,
    uint8_t* Output,
    const int32_t* Bias,
    size_t M,
    size_t N,
    float Scale,
    uint8_t ZeroPoint
    )
/*++

Routine Description:

    This routine requantizes the intermediate buffer to the output buffer
    optionally adding the supplied bias.

Arguments:

    Input - Supplies the input matrix.

    Output - Supplies the output matrix.

    Bias - Supplies the optional bias vector to be added to the input buffer
        before requantization.

    Buffer - Supplies the output matrix.

    M - Supplies the number of elements of the bias vector and the number of
        rows in the output matrix.

    N - Supplies the number of columns of the output matrix.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
    MLAS_FLOAT32X4 ScaleVector = MlasBroadcastFloat32x4(Scale);
    MLAS_FLOAT32X4 MinimumValueVector = MlasBroadcastFloat32x4(float(0 - ZeroPoint));
    MLAS_FLOAT32X4 MaximumValueVector = MlasBroadcastFloat32x4(float(255 - ZeroPoint));
    MLAS_INT32X4 ZeroPointVector = MlasBroadcastInt32x4(ZeroPoint);
    MLAS_INT32X4 BiasVector = _mm_setzero_si128();

    //
    // Step through each row of the output matrix.
    //

    while (M-- > 0) {

        if (Bias != nullptr) {
            BiasVector = MlasBroadcastInt32x4(*Bias++);
        }

        size_t n = N;

        while (n >= 4) {

            MLAS_INT32X4 IntegerVector = _mm_loadu_si128((const __m128i *)Input);
            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);
            IntegerVector = _mm_packus_epi16(IntegerVector, IntegerVector);

            *((int32_t*)Output) = _mm_cvtsi128_si32(IntegerVector);

            Input += 4;
            Output += 4;
            n -= 4;
        }

        while (n > 0) {

            MLAS_INT32X4 IntegerVector = _mm_cvtsi32_si128(*Input);
            IntegerVector = MlasRequantizeOutputVector(IntegerVector, BiasVector,
                ScaleVector, MinimumValueVector, MaximumValueVector, ZeroPointVector);

            *Output = (uint8_t)_mm_cvtsi128_si32(IntegerVector);

            Input += 1;
            Output += 1;
            n -= 1;
        }
    }
}

#endif


void
MLASCALL
MlasQLinearAddS8Kernel(
    const int8_t* InputA,
    float ScaleA,
    int8_t ZeroPointA,
    const int8_t* InputB,
    float ScaleB,
    int8_t ZeroPointB,
    float ScaleC,
    int8_t ZeroPointC,
    int8_t* OutputC,
    size_t LengthA,
    size_t LengthB
    )
{
    MlasQLinearAddKernel<int8_t>(
        InputA, ScaleA, ZeroPointA,
        InputB, ScaleB, ZeroPointB,
        ScaleC, ZeroPointC, OutputC,
        LengthA, LengthB
    );
}

void
MLASCALL
MlasQLinearAddU8Kernel(
    const uint8_t* InputA,
    float ScaleA,
    uint8_t ZeroPointA,
    const uint8_t* InputB,
    float ScaleB,
    uint8_t ZeroPointB,
    float ScaleC,
    uint8_t ZeroPointC,
    uint8_t* OutputC,
    size_t LengthA,
    size_t LengthB
    )
{
    MlasQLinearAddKernel<uint8_t>(
        InputA, ScaleA, ZeroPointA,
        InputB, ScaleB, ZeroPointB,
        ScaleC, ZeroPointC, OutputC,
        LengthA, LengthB
    );
}


template <typename T>
class MlasQLinearAddPlatformKernel
{
public:
    typedef void (MLASCALL KernelFunction)(
        const T* InputA,
        float ScaleA,
        T ZeroPointA,
        const T* InputB,
        float ScaleB,
        T ZeroPointB,
        float ScaleC,
        T ZeroPointC,
        T* OutputC,
        size_t LengthA,
        size_t LengthB
        );

    typedef KernelFunction* PKernelFunction;

    static PKernelFunction GetKernel();
};

template <>
MlasQLinearAddPlatformKernel<uint8_t>::PKernelFunction
MlasQLinearAddPlatformKernel<uint8_t>::GetKernel()
{
    return MlasPlatform.QLinearAddU8Kernel;
}

template <>
MlasQLinearAddPlatformKernel<int8_t>::PKernelFunction
MlasQLinearAddPlatformKernel<int8_t>::GetKernel()
{
    return MlasPlatform.QLinearAddS8Kernel;
}

template <typename T>
void
MLASCALL
MlasQLinearAdd(
    const T* InputA,
    float ScaleA,
    T ZeroPointA,
    const T* InputB,
    float ScaleB,
    T ZeroPointB,
    float ScaleC,
    T ZeroPointC,
    T* OutputC,
    size_t LengthA,
    size_t LengthB
    )
{
#if defined(MLAS_TARGET_AMD64)
        auto kernel = MlasQLinearAddPlatformKernel<T>::GetKernel();
        kernel(
#else
        MlasQLinearAddKernel<T>(
#endif
            InputA, ScaleA, ZeroPointA, InputB, ScaleB, ZeroPointB, ScaleC, ZeroPointC, OutputC, LengthA, LengthB);
}


template
void
MLASCALL
MlasQLinearAdd<int8_t>(
    const int8_t* InputA,
    float ScaleA,
    int8_t ZeroPointA,
    const int8_t* InputB,
    float ScaleB,
    int8_t ZeroPointB,
    float ScaleC,
    int8_t ZeroPointC,
    int8_t* OutputC,
    size_t LengthA,
    size_t LengthB
    );

template
void
MLASCALL
MlasQLinearAdd<uint8_t>(
    const uint8_t* InputA,
    float ScaleA,
    uint8_t ZeroPointA,
    const uint8_t* InputB,
    float ScaleB,
    uint8_t ZeroPointB,
    float ScaleC,
    uint8_t ZeroPointC,
    uint8_t* OutputC,
    size_t LengthA,
    size_t LengthB
    );

MLAS_INTERNAL_DATA  MLAS_DECLSPEC_ALIGN(const uint8_t MLasPackBytesMM256VpshufbControl[32], 32) = {
    0,4,8,12,        255,255,255,255, 255,255,255,255, 255,255,255,255,
    255,255,255,255, 0,4,8,12,        255,255,255,255, 255,255,255,255
};

MLAS_INTERNAL_DATA  MLAS_DECLSPEC_ALIGN(const int32_t MLasPackBytesMM256VpermpsControl[8], 32) = {
    0, 5, 2, 3, 4, 1, 6, 7
};
