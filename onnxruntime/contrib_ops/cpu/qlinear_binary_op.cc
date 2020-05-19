// Copyright (c Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinear_binary_op.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

template <typename TBroadcaster, typename Output, typename Input0Scalar, typename Input1Scalar, typename General>
void QLinearBroadcastLoop(TBroadcaster& bc, Output& output, Input0Scalar input0scalar, Input1Scalar input1scalar, General general,
                          float A_scale, float B_scale, float C_scale, int A_zero_point, int B_zero_point, int C_zero_point) {
  if (bc.IsInput0Scalar()) {
    while (output)
      input0scalar(output.NextSpanOutput(), bc.NextScalar0(), bc.NextSpan1(), A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
  } else if (bc.IsInput1Scalar()) {
    while (output)
      input1scalar(output.NextSpanOutput(), bc.NextSpan0(), bc.NextScalar1(), A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
  } else {
    while (output)
      general(output.NextSpanOutput(), bc.NextSpan0(), bc.NextSpan1(), A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
  }
}

template <typename T, typename Input0Scalar, typename Input1Scalar, typename General>
void QLinearBroadcastOneSpan(ThreadPool* tp, gsl::span<const T> input0_span,  gsl::span<const T> input1_span, gsl::span<T> output_span,
                             Input0Scalar input0scalar, Input1Scalar input1scalar, General general,
                             float A_scale, float B_scale, float C_scale, int A_zero_point, int B_zero_point, int C_zero_point) {
  if (bc.IsInput0Scalar()) {
    ThreadPool::TryParallelFor(
        tp, output_len, unit_cost,
        [output_span, input0_span, input1_span](std::ptrdiff_t first, std::ptrdiff_t last) {
          size_t count = static_cast<size_t>(last - first);
          input0scalar(output_span.subspan(first, count), *input0_span.data(), input1_span.subspan(first, count), 
                        A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
    }
  }
  else if (bc.IsInput1Scalar()) {
    ThreadPool::TryParallelFor(
        tp, output_len, unit_cost,  
        [output_span, input0_span, input1_span](std::ptrdiff_t first, std::ptrdiff_t last) {
            size_t count = static_cast<size_t>(last - first);
            input1scalar(output_span.subspan(first, count), input0_span.subspan(first, count), *input1_span.data(),
                          A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
    }
  }
  else {
    ThreadPool::TryParallelFor(tp, output_len, unit_cost,
        size_t count = static_cast<size_t>(last - first);
        [output_span, input0_span, input1_span](std::ptrdiff_t first, std::ptrdiff_t last) {
            general(output_span.subspan(first, count), input0_span.subspan(first, count), input1_span.subspan(first, count),
                    A_scale, B_scale, C_scale, A_zero_point, B_zero_point, C_zero_point);
    }
  }
}

template <typename T, typename Input0Scalar, typename Input1Scalar, typename General>
Status QLinearBroadcastTwo(OpKernelContext& context, Input0Scalar input0scalar, Input1Scalar input1scalar, General general, double unit_cost) {
  auto tensor_a_scale = context.Input<Tensor>(1);
  auto tensor_a_zero_point = context.Input<Tensor>(2);
  auto tensor_b_scale = context.Input<Tensor>(4);
  auto tensor_b_zero_point = context.Input<Tensor>(5);
  auto tensor_c_scale = context.Input<Tensor>(6);
  auto tensor_c_zero_point = context.Input<Tensor>(7);

  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_a_scale),
              "MatmulInteger : input1 A_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_a_zero_point == nullptr || IsScalarOr1ElementVector(tensor_a_zero_point),
              "MatmulInteger : input1 A_zero_point must be a scalar or 1D tensor of size 1 if given");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_b_scale),
              "MatmulInteger : input1 B_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_b_zero_point == nullptr || IsScalarOr1ElementVector(tensor_b_zero_point),
              "MatmulInteger : input1 B_zero_point must be a scalar or 1D tensor of size 1 if given");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_c_scale),
              "MatmulInteger : input1 C_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_c_zero_point == nullptr || IsScalarOr1ElementVector(tensor_c_zero_point),
              "MatmulInteger : input1 C_zero_point must be a scalar or 1D tensor of size 1 if given");

  const float A_scale = *(tensor_a_scale->Data<float>());
  const T A_zero_point = (nullptr == tensor_a_zero_point) ? static_cast<T>(0) : *(tensor_a_zero_point->template Data<T>());
  const float B_scale = *(tensor_b_scale->Data<float>());
  const T B_zero_point = (nullptr == tensor_b_zero_point) ? static_cast<T>(0) : *(tensor_b_zero_point->template Data<T>());
  const float C_scale = *(tensor_c_scale->Data<float>());
  const T C_zero_point = (nullptr == tensor_c_zero_point) ? static_cast<T>(0) : *(tensor_c_zero_point->template Data<T>());

  TBroadcaster<T, T> bc(*context.Input<Tensor>(0), *context.Input<Tensor>(3));
  Tensor& output_tensor = *context.Output(0, bc.GetOutputShape());
  auto span_size = bc.GetSpanSize();
  TBroadcastOutput<T> output(span_size, output_tensor);
  int64_t output_len = output_tensor.Shape().Size();

  if (output_len == static_cast<int64_t>(span_size)) {
    // Only one big span for all data, parallel inside it
    QLinearBroadcastOneSpan(tp, bc.NextSpan0(), bc.NextSpan1(), output.NextSpanOutput(), input0scalar, input1scalar, general);
  }
  else {
    ThreadPool::TryParallelFor(
        tp, output_len / span_size, unit_cost * span_size,
        [const &bc, tp](std::ptrdiff_t first_span, std::ptrdiff_t last_span) {
            TBroadcaster<T, T> span_bc(bc);
            TBroadcastOutput<T> output(span_size, output_tensor, first_span * span_size, last_span * span_size);
            span_bc.AdvanceBy(first_span * span_size);
            QLinearBroadcastLoop(
                span_bc, output, input0scalar, input1scalar, general, A_scale, B_scale, C_scale,
                static_cast<int>(A_zero_point), static_cast<int>(B_zero_point), static_cast<int>(C_zero_point));
        });
  }
  return Status::OK();
}

template <typename T>
Status QLinearAdd<T>::Compute(OpKernelContext* context) const {
  auto thread_pool = context->GetOperatorThreadPool();
  return QLinearBroadcastTwo<T>(
      *context,
      [](gsl::span<T> output, T input0, gsl::span<const T> input1, 
         float A_scale, float B_scale, float C_scale, int A_zero_point, int B_zero_point, int C_zero_point) {
        constexpr int qmax = (int)std::numeric_limits<T>::max();
        constexpr int qmin = (int)std::numeric_limits<T>::min();
        float a_value = A_scale * (static_cast<int>(input0) - A_zero_point);
        Q
        output = (((((input1.array().template cast<float>() - static_cast<float>(B_zero_point)) * B_scale) + a_value) / C_scale).round().template cast<int>() + C_zero_point)
                     .max(qmin)
                     .min(qmax)
                     .template cast<T>();
      },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1, float A_scale, float B_scale, float C_scale,
         int A_zero_point, int B_zero_point, int C_zero_point) {
        constexpr int qmax = (int)std::numeric_limits<T>::max();
        constexpr int qmin = (int)std::numeric_limits<T>::min();
        float b_value = B_scale * (static_cast<int>(input1) - B_zero_point);
        output = (((((input0.array().template cast<float>() - static_cast<float>(A_zero_point)) * A_scale) + b_value) / C_scale).round().template cast<int>() + C_zero_point)
                     .max(qmin)
                     .min(qmax)
                     .template cast<T>();
      },
      [thread_pool](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1, float A_scale, float B_scale, float C_scale,
                    int A_zero_point, int B_zero_point, int C_zero_point) {
        MlasQLinearAdd(input0.data(), A_scale, (T)A_zero_point,
                       input1.data(), B_scale, (T)B_zero_point,
                       C_scale, (T)C_zero_point, output.data(), output.outerStride(), thread_pool);
      });
}

template <typename T>
Status QLinearMul<T>::Compute(OpKernelContext* context) const {
  return QLinearBroadcastTwo<T>(
      *context,
      [](EigenVectorMap<T> output, T input0, ConstEigenVectorMap<T> input1, float A_scale, float B_scale, float C_scale,
         int A_zero_point, int B_zero_point, int C_zero_point) {
        constexpr int qmax = (int)std::numeric_limits<T>::max();
        constexpr int qmin = (int)std::numeric_limits<T>::min();
        float a_value_scaled_b_c = A_scale * (static_cast<int>(input0) - A_zero_point) * B_scale / C_scale;
        output = (((input1.array().template cast<int>() - B_zero_point).template cast<float>() * a_value_scaled_b_c).round().template cast<int>() + C_zero_point)
                     .max(qmin)
                     .min(qmax)
                     .template cast<T>();
      },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, T input1, float A_scale, float B_scale, float C_scale,
         int A_zero_point, int B_zero_point, int C_zero_point) {
        constexpr int qmax = (int)std::numeric_limits<T>::max();
        constexpr int qmin = (int)std::numeric_limits<T>::min();
        float b_value_scaled_a_c = B_scale * (static_cast<int>(input1) - B_zero_point) * A_scale / C_scale;
        output = (((input0.array().template cast<int>() - A_zero_point).template cast<float>() * b_value_scaled_a_c).round().template cast<int>() + C_zero_point)
                     .max(qmin)
                     .min(qmax)
                     .template cast<T>();
      },
      [](EigenVectorMap<T> output, ConstEigenVectorMap<T> input0, ConstEigenVectorMap<T> input1, float A_scale, float B_scale, float C_scale,
         int A_zero_point, int B_zero_point, int C_zero_point) {
        constexpr int qmax = (int)std::numeric_limits<T>::max();
        constexpr int qmin = (int)std::numeric_limits<T>::min();
        output = (((((input0.array().template cast<int>() - A_zero_point).template cast<float>() * A_scale) *
                    ((input1.array().template cast<int>() - B_zero_point).template cast<float>() * B_scale)) /
                   C_scale)
                      .round()
                      .template cast<int>() +
                  C_zero_point)
                     .max(qmin)
                     .min(qmax)
                     .template cast<T>();
      });
}

#define REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(op_name, version, data_type, KERNEL_CLASS) \
  ONNX_CPU_OPERATOR_TYPED_MS_KERNEL(                                                    \
      op_name, version, data_type,                                                      \
      KernelDefBuilder()                                                                \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()),               \
      KERNEL_CLASS<data_type>);

REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearAdd, 1, int8_t, QLinearAdd);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearAdd, 1, uint8_t, QLinearAdd);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearMul, 1, int8_t, QLinearMul);
REG_QLINEAR_ELEMENTWISE_TYPED_KERNEL(QLinearMul, 1, uint8_t, QLinearMul);

}  // namespace contrib
}  // namespace onnxruntime
