;++
;
; Copyright (c) Microsoft Corporation. All rights reserved.
;
; Licensed under the MIT License.
;
; Module Name:
;
;    QLinearBinaryOpKernelAvx2.asm
;
; Abstract:
;
;    This module implements the kernels for the quantized linear add
;    for element type int8_t and uint8_t.
;
;    This implementation uses AVX2 instructions.
;
;--

        .xlist
INCLUDE mlasi.inc
        .list

        EXTERN  MLasPackBytesMM256VpshufbControl:NEAR
        EXTERN  MLasPackBytesMM256VpermpsControl:NEAR

;
; Stack frame layout for the QLinearBinaryOp kernels.
;

QLinearBinaryOpFrame STRUCT
        SavedXmm6 OWORD ?
        SavedXmm7 OWORD ?
        SavedXmm8 OWORD ?
        SavedXmm9 OWORD ?
        SavedXmm10 OWORD ?
        SavedXmm11 OWORD ?
        Padding0 QWORD ?
        Padding1 QWORD ?
        Padding2 QWORD ?
        ReturnAddress QWORD ?
        PreviousP1Home QWORD ?
        PreviousP2Home QWORD ?
        PreviousP3Home QWORD ?
        PreviousP4Home QWORD ?
        ScaleB  QWORD ?
        ZeroPointB QWORD ?
        ScaleC QWORD ?
        ZeroPointC QWORD ?
        OutputC QWORD ?
        LengthA QWORD ?
        LengthB QWord ?
QLinearBinaryOpFrame ENDS

;
; Macro Description:
;
;   This macro generates code to extend signed/unsigned int8 to
;   signed/unsigned int64/32/16 respectively, according to DataType
;   and target.
;
; Arguments:
;
;   Target - target register, could be 16/32/64 bits one.
;
;   Source - Supplies address of 8 x 8bits integers.
;
;   DataType - S8 or U8
;

Extend8BitsInt MACRO Target, Source, DataType
IFIDN <DataType>, <S8>
        movsx Target,BYTE PTR Source
ELSE
        movzx Target,BYTE PTR Source
ENDIF
        ENDM

;
; Macro Description:
;
;   This macro generates code to broadcast one (s/u)int8 to 8 x int32,
;   according to signed/unsigned.
;
; Arguments:
;   TargetYmm - target ymm register.
;
;   TargetXmm -- intermedia xmm register used
;
;   Source - Supplies address of 8 x 8bits integers.
;
;   DataType - S8 or U8
;

BroadcastByteDWords MACRO TargetYmm, TargetXmm, Source, DataType
IFIDN <DataType>, <S8> 
        movsx TargetXmm,BYTE PTR Source
ELSE
        movzx TargetXmm,BYTE PTR Source
ENDIF
        vpbroadcastd TargetYmm,TargetXmm
        ENDM


;
; Macro Description:
;
;   This macro generates code to unpack 8 x (s/u)int8 to 8 x int32,
;   according to signed/unsigned.
;
; Arguments:
;
;   Source - Supplies address of 8 x 8bits integers.
;
;   Target - target ymm register.
;
;   DataType - S8 or U8
;

UnpackBytesDWords MACRO Target, Source, DataType
IFIDN <DataType>, <S8> 
        vpmovsxbd Target,QWORD PTR Source
ELSE
        vpmovzxbd Target,QWORD PTR Source
ENDIF
        ENDM

;
; Macro Description:
;
;   This macro generates code to set Target 64bits register with the
;   max value of signed/unsigned int8 specified by DataType.
;
; Arguments:
;
;   Target - target 64bits register.
;
;   DataType - S8 or U8.
;

SetMax8BitsValue MACRO Target, DataType
IFIDN <DataType>, <S8>
        mov Target,QWORD PTR 127
ELSE
        mov Target,QWORD PTR 255
ENDIF
        ENDM

;
; Macro Description:
;
;   This macro generates code to set Target 64bits register with the
;   min value of signed/unsigned int8 specified by DataType.
;
; Arguments:
;
;   Target - target 64bits register.
;
;   DataType - S8 or U8.
;

SetMin8BitsValue MACRO Target, DataType
IFIDN <DataType>, <S8>
        mov Target,QWORD PTR -128
ELSE
        mov Target,QWORD PTR 0
ENDIF
        ENDM

;
; Macro Description:
;
;   This macro generates code for function QLinearOpName()
;   on the specified signed/unsigned int8 DataType.
;
; Arguments:
;
;   DataType - S8 or U8.
;
;   OpName - Name of the QLinearOp, like Add, Mul, etc.
;
;   OpInstruction - the assembly code prefix which op() two ymm vector of floats,
;                   like vaddps, vmulps, etc
;

QLinearBinaryOpAvx2 MACRO DataType, OpName, OpInstruction

;
; Routine Description:
;
;    This routine implements the kernels for the Quantize Linear OpName
;    for element type DataType, vector on vector.
;
; Arguments:
;
;    InputA (rcx) - Supplies the address of InputA.
;
;    ScaleA (xmm1) - Supplies A's Scale value in float.
;
;    ZeroPointA (r8) - Supplies A's zero point value.
;
;    InputB (r9) - Supplies the address of InputB.
;
;    ScaleB - Supplies B's Scale value in float.
;
;    ZeroPointB - Supplies B's zero point value.
;
;    ScaleC - Supplies C's Scale value in float.
;
;    ZeroPointC - Supplies C's zero point value.
;
;    OutputC - Supplies the address of OutputC.
;
;    LengthA - Supplies the number of elements of InputA.
;
;    LengthB - Supplies the number of elements of InputB.
;              LengthB == LengthA or LengthA == 1 or LengthB == 1.
;
; Return Value:
;
;    None.
;

        NESTED_ENTRY MlasQLinear&OpName&&DataType&KernelAvx2, _TEXT

        alloc_stack (QLinearBinaryOpFrame.ReturnAddress)

        save_xmm128_avx xmm6,QLinearBinaryOpFrame.SavedXmm6
        save_xmm128_avx xmm7,QLinearBinaryOpFrame.SavedXmm7
        save_xmm128_avx xmm8,QLinearBinaryOpFrame.SavedXmm8
        save_xmm128_avx xmm9,QLinearBinaryOpFrame.SavedXmm9
        save_xmm128_avx xmm10,QLinearBinaryOpFrame.SavedXmm10
        save_xmm128_avx xmm11,QLinearBinaryOpFrame.SavedXmm11

        END_PROLOGUE

        vbroadcastss ymm0,xmm1                  ; Vector of ScaleA
        vbroadcastss ymm1,DWORD PTR QLinearBinaryOpFrame.ScaleB[rsp]
        vbroadcastss ymm2,DWORD PTR QLinearBinaryOpFrame.ScaleC[rsp]
        Extend8BitsInt r8,r8b,DataType          ; Zero Point A,B,C
        Extend8BitsInt rdx,BYTE PTR QLinearBinaryOpFrame.ZeroPointB[rsp],DataType
        Extend8BitsInt rax,BYTE PTR QLinearBinaryOpFrame.ZeroPointC[rsp],DataType
        movq    xmm3,r8
        movq    xmm4,rdx
        movq    xmm5,rax
        vbroadcastss ymm3,xmm3                  ; Vector of ZeroPointA
        vbroadcastss ymm4,xmm4                  ; Vector of ZeroPointB
        vbroadcastss ymm5,xmm5                  ; Vector of ZeroPointC

        lea     rdx,MLasPackBytesMM256VpshufbControl
        lea     r8,MLasPackBytesMM256VpermpsControl
        vmovaps ymm10,[rdx]
        vmovaps ymm11,[r8]

        SetMax8BitsValue rdx,DataType
        movq    xmm6,rdx
        vbroadcastss ymm6,xmm6
        vpsubd  ymm6,ymm6,ymm5

        SetMin8BitsValue r8,DataType
        movq    xmm7,r8
        vbroadcastss ymm7,xmm7
        vpsubd  ymm7,ymm7,ymm5

        mov     rax,QWORD PTR 8
        mov     r8,QLinearBinaryOpFrame.OutputC[rsp]
        mov     rdx,QLinearBinaryOpFrame.LengthA[rsp]
        cmp     rdx,QWORD PTR 1
        jae     QLinear&OpName&&DataType&Avx2Process8EntranceScalarOnVector

QLinear&OpName&&DataType&Avx2Process8EntranceScalarOnVector:
        mov     rdx,QLinearBinaryOpFrame.LengthB[rsp]
        test    rdx,rdx
        jz      QLinear&OpName&&DataType&Avx2Exit
        BroadcastByteDWords ymm8,xmm8,[rcx],DataType
        vpsubd  ymm8,ymm8,ymm3                  ; - ZeroPointA
        vcvtdq2ps ymm8,ymm8                     ; FloatVectorA
        vmulps  ymm8,ymm8,ymm0                  ; * ScaleA
        
QLinear&OpName&&DataType&Avx2Process8LoopScalarOnVector:
        test    rdx,rdx
        jz      QLinear&OpName&&DataType&Avx2Exit

        UnpackBytesDWords ymm9,[r9],DataType    ; IntegerVectorB
        vpsubd  ymm9,ymm9,ymm4                  ; - ZeroPointB
        vcvtdq2ps ymm9,ymm9                     ; FloatVectorB
        vmulps  ymm9,ymm9,ymm1                  ; * ScaleB

        OpInstruction  ymm9,ymm8,ymm9           ; OpName two float values

        vdivps  ymm9,ymm9,ymm2                  ; Quantize 8 values, / ScaleC
        add     r9,rax                          ; out-of-order instruction(s)
        vcvtps2dq ymm9,ymm9                     ; nearbyintf()
        vpmaxsd ymm9,ymm9,ymm7
        vpminsd ymm9,ymm9,ymm6
        vpaddd  ymm9,ymm9,ymm5                  ; + ZeroPointC
        vpshufb ymm9,ymm9,ymm10                 ; pack 32bits integers into 8bit integers
        vpermps ymm9,ymm11,ymm9

        sub     rdx,rax
        jb      QLinear&OpName&&DataType&Avx2StoreLessThan8ScalarOnVector

        movsd   QWORD PTR [r8],xmm9
        add     r8,rax
        jmp     QLinear&OpName&&DataType&Avx2Process8LoopScalarOnVector

QLinear&OpName&&DataType&Avx2StoreLessThan8ScalarOnVector:
        add     rdx,rax
        pextrq  rax,xmm9,0

QLinear&OpName&&DataType&Avx2StoreLoopScalarOnVector:
        mov     BYTE PTR [r8],al
        shr     rax,8
        inc     r8
        dec     rdx
        jnz     QLinear&OpName&&DataType&Avx2StoreLoopScalarOnVector
        jmp      QLinear&OpName&&DataType&Avx2Exit


QLinear&OpName&&DataType&Avx2Process8EntranceVectorOnScalar:
        mov     rdx,QLinearBinaryOpFrame.LengthB[rsp]
        test    rdx,rdx
        jz      QLinear&OpName&&DataType&Avx2Exit
        BroadcastByteDWords ymm9,xmm9,[r9],DataType ; IntegerVectorB
        vpsubd  ymm9,ymm9,ymm4                  ; - ZeroPointB
        vcvtdq2ps ymm9,ymm9                     ; FloatVectorB
        vmulps  ymm9,ymm9,ymm1                  ; * ScaleB

QLinear&OpName&&DataType&Avx2Process8LoopVectorOnScalar:
        test    rdx,rdx
        jz      QLinear&OpName&&DataType&Avx2Exit

        UnpackBytesDWords ymm8,[rcx],DataType   ; IntegerVectorA
        vpsubd  ymm8,ymm8,ymm3                  ; - ZeroPointA
        vcvtdq2ps ymm8,ymm8                     ; FloatVectorA
        vmulps  ymm8,ymm8,ymm0                  ; * ScaleA

        OpInstruction  ymm8,ymm8,ymm9           ; OpName two float values

        vdivps  ymm8,ymm8,ymm2                  ; Quantize 8 values, / ScaleC
        add     rcx,rax                         ; out-of-order instruction(s)
        vcvtps2dq ymm8,ymm8                     ; nearbyintf()()
        vpmaxsd ymm8,ymm8,ymm7
        vpminsd ymm8,ymm8,ymm6
        vpaddd  ymm8,ymm8,ymm5                  ; + ZeroPointC
        vpshufb ymm8,ymm8,ymm10                 ; pack 32bits integers into 8bit integers
        vpermps ymm8,ymm11,ymm8

        sub     rdx,rax
        jb      QLinear&OpName&&DataType&Avx2StoreLessThan8VectorOnScalar

        movsd   QWORD PTR [r8],xmm8
        add     r8,rax
        jmp     QLinear&OpName&&DataType&Avx2Process8LoopVectorOnScalar

QLinear&OpName&&DataType&Avx2StoreLessThan8VectorOnScalar:
        add     rdx,rax
        pextrq  rax,xmm8,0

QLinear&OpName&&DataType&Avx2StoreLoopVectorOnScalar:
        mov     BYTE PTR [r8],al
        shr     rax,8
        inc     r8
        dec     rdx
        jnz     QLinear&OpName&&DataType&Avx2StoreLoopVectorOnScalar

        jmp      QLinear&OpName&&DataType&Avx2Exit


QLinear&OpName&&DataType&Avx2Process8LoopVectorOnVector:
        test    rdx,rdx
        jz      QLinear&OpName&&DataType&Avx2Exit

        UnpackBytesDWords ymm8,[rcx],DataType   ; IntegerVectorA
        UnpackBytesDWords ymm9,[r9],DataType    ; IntegerVectorB
        vpsubd  ymm8,ymm8,ymm3                  ; - ZeroPointA
        vpsubd  ymm9,ymm9,ymm4                  ; - ZeroPointB
        vcvtdq2ps ymm8,ymm8                     ; FloatVectorA
        vcvtdq2ps ymm9,ymm9                     ; FloatVectorB
        vmulps  ymm8,ymm8,ymm0                  ; * ScaleA
        vmulps  ymm9,ymm9,ymm1                  ; * ScaleB

        OpInstruction  ymm8,ymm8,ymm9           ; OpName two float values

        vdivps  ymm8,ymm8,ymm2                  ; Quantize 8 values, / ScaleC
        add     rcx,rax                         ; out-of-order instruction(s)
        add     r9,rax                          ; out-of-order instruction(s)
        vcvtps2dq ymm8,ymm8                     ; nearbyintf()
        vpmaxsd ymm8,ymm8,ymm7
        vpminsd ymm8,ymm8,ymm6
        vpaddd  ymm8,ymm8,ymm5                  ; + ZeroPointC
        vpshufb ymm8,ymm8,ymm10                 ; pack 32bits integers into 8bit integers
        vpermps ymm8,ymm11,ymm8

        sub     rdx,rax
        jb      QLinear&OpName&&DataType&Avx2StoreLessThan8VectorOnVector

        movsd   QWORD PTR [r8],xmm8
        add     r8,rax
        jmp     QLinear&OpName&&DataType&Avx2Process8LoopVectorOnVector

QLinear&OpName&&DataType&Avx2StoreLessThan8VectorOnVector:
        add     rdx,rax
        pextrq  rax,xmm8,0

QLinear&OpName&&DataType&Avx2StoreLoopVectorOnVector:
        mov     BYTE PTR [r8],al
        shr     rax,8
        inc     r8
        dec     rdx
        jnz     QLinear&OpName&&DataType&Avx2StoreLoopVectorOnVector

QLinear&OpName&&DataType&Avx2Exit:
        vzeroupper
        vmovaps xmm6,QLinearBinaryOpFrame.SavedXmm6[rsp]
        vmovaps xmm7,QLinearBinaryOpFrame.SavedXmm7[rsp]
        vmovaps xmm8,QLinearBinaryOpFrame.SavedXmm8[rsp]
        vmovaps xmm9,QLinearBinaryOpFrame.SavedXmm9[rsp]
        vmovaps xmm10,QLinearBinaryOpFrame.SavedXmm10[rsp]
        vmovaps xmm11,QLinearBinaryOpFrame.SavedXmm11[rsp]
        add     rsp,(QLinearBinaryOpFrame.ReturnAddress)

        BEGIN_EPILOGUE

        ret

        NESTED_END MlasQLinear&OpName&&DataType&KernelAvx2, _TEXT

        ENDM

;
; Generate the QLinearAdd Avx2 S8 kernel.
;

QLinearBinaryOpAvx2 S8,Add,vaddps


;
; Generate the QLinearAdd Avx2 U8 kernel.
;

QLinearBinaryOpAvx2 U8,Add,vaddps


        END