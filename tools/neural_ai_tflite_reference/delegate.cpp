// SPDX-License-Identifier: Apache-2.0
// Independent, scalar reference kernels. This is not a CPU performance backend.
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/delegates/external/external_delegate_interface.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

namespace
{
struct Scale
{
    int32_t multiplier = 0;
    int shift = 0;
};

// Deliberately independent of compiler implementation, with the same contract.
bool Quantize(double real, Scale &scale)
{
    if (!std::isfinite(real) || real <= 0) return false;
    double best = std::numeric_limits<double>::infinity();
    bool found = false;
    for (int shift = 0; shift <= 31; ++shift)
    {
        const double scaled = std::ldexp(real, shift);
        if (scaled < 1 || scaled > INT32_MAX) continue;
        const double rounded = std::round(scaled);
        if (rounded > INT32_MAX) continue;
        const double error = std::abs(std::ldexp(rounded, -shift) - real);
        if (error <= best)
        {
            scale = {int32_t(rounded), shift};
            best = error;
            found = true;
        }
    }
    return found;
}

int8_t Requantize(int64_t accumulator, Scale scale, int zp, int low, int high)
{
    const __int128 product = __int128(accumulator) * scale.multiplier;
    const __int128 magnitude = product < 0 ? -product : product;
    const __int128 rounded = scale.shift ?
        (magnitude + (__int128(1) << (scale.shift - 1))) >> scale.shift : magnitude;
    const __int128 value = (product < 0 ? -rounded : rounded) + zp;
    return int8_t(std::clamp<__int128>(value, low, high));
}

bool StaticTensor(const TfLiteTensor &tensor)
{
    if (!tensor.dims || tensor.allocation_type == kTfLiteDynamic) return false;
    for (int i = 0; i < tensor.dims->size; ++i)
        if (tensor.dims->data[i] <= 0) return false;
    return !tensor.sparsity;
}

int64_t Elements(const TfLiteTensor &tensor)
{
    int64_t count = 1;
    for (int i = 0; i < tensor.dims->size; ++i)
    {
        if (count > INT32_MAX / tensor.dims->data[i]) return -1;
        count *= tensor.dims->data[i];
    }
    return count;
}

const TfLiteAffineQuantization *Affine(const TfLiteTensor &tensor)
{
    if (tensor.quantization.type != kTfLiteAffineQuantization) return nullptr;
    return static_cast<const TfLiteAffineQuantization *>(tensor.quantization.params);
}

bool PerTensor(const TfLiteTensor &tensor)
{
    const auto *q = Affine(tensor);
    return q && q->scale && q->scale->size == 1 && q->zero_point && q->zero_point->size == 1 &&
        std::isfinite(q->scale->data[0]) && q->scale->data[0] > 0 &&
        q->zero_point->data[0] >= -128 && q->zero_point->data[0] <= 127;
}

struct Kernel
{
    int input = -1, weight = -1, bias = -1, output = -1;
    int code = 0, channels = 0, batches = 0;
    int ih = 1, iw = 1, ic = 1, oh = 1, ow = 1, kh = 1, kw = 1;
    int sh = 1, sw = 1, dh = 1, dw = 1, ph = 0, pw = 0, depthMultiplier = 1;
    int inputZp = 0, outputZp = 0, low = -128, high = 127;
    std::vector<Scale> scales;
    std::vector<int> inputShape, outputShape;
};

bool Describe(TfLiteContext *ctx, const TfLiteNode &node, int code, Kernel &k)
{
    if (code != kTfLiteBuiltinConv2d && code != kTfLiteBuiltinDepthwiseConv2d &&
        code != kTfLiteBuiltinFullyConnected) return false;
    if (!node.builtin_data || !node.inputs || node.inputs->size < 2 || node.inputs->size > 3 ||
        !node.outputs || node.outputs->size != 1) return false;
    k.input = node.inputs->data[0]; k.weight = node.inputs->data[1];
    k.bias = node.inputs->size == 3 ? node.inputs->data[2] : -1;
    k.output = node.outputs->data[0]; k.code = code;
    const auto &x = ctx->tensors[k.input];
    const auto &w = ctx->tensors[k.weight];
    const auto &y = ctx->tensors[k.output];
    if (x.type != kTfLiteInt8 || w.type != kTfLiteInt8 || y.type != kTfLiteInt8 ||
        !StaticTensor(x) || !StaticTensor(w) || !StaticTensor(y) ||
        !PerTensor(x) || !PerTensor(y) || Elements(x) <= 0 || Elements(w) <= 0 || Elements(y) <= 0)
        return false;
    if (w.allocation_type != kTfLiteMmapRo) return false; // Constants only.
    k.inputShape.assign(x.dims->data, x.dims->data + x.dims->size);
    k.outputShape.assign(y.dims->data, y.dims->data + y.dims->size);
    k.inputZp = Affine(x)->zero_point->data[0];
    k.outputZp = Affine(y)->zero_point->data[0];
    TfLiteFusedActivation activation;
    TfLitePadding padding = kTfLitePaddingUnknown;
    if (code == kTfLiteBuiltinFullyConnected)
    {
        const auto &p = *static_cast<const TfLiteFullyConnectedParams *>(node.builtin_data);
        if (p.weights_format != kTfLiteFullyConnectedWeightsFormatDefault || p.asymmetric_quantize_inputs ||
            w.dims->size != 2) return false;
        activation = p.activation;
        k.channels = w.dims->data[0]; k.ic = w.dims->data[1];
        if (Elements(x) % k.ic) return false;
        k.batches = int(Elements(x) / k.ic);
        if (Elements(y) != int64_t(k.batches) * k.channels) return false;
    }
    else
    {
        if (x.dims->size != 4 || w.dims->size != 4 || y.dims->size != 4) return false;
        k.batches = x.dims->data[0]; k.ih = x.dims->data[1]; k.iw = x.dims->data[2]; k.ic = x.dims->data[3];
        k.oh = y.dims->data[1]; k.ow = y.dims->data[2]; k.channels = y.dims->data[3];
        k.kh = w.dims->data[1]; k.kw = w.dims->data[2];
        if (k.batches != y.dims->data[0]) return false;
        if (code == kTfLiteBuiltinConv2d)
        {
            const auto &p = *static_cast<const TfLiteConvParams *>(node.builtin_data);
            activation = p.activation; padding = p.padding;
            k.sh = p.stride_height; k.sw = p.stride_width;
            k.dh = p.dilation_height_factor; k.dw = p.dilation_width_factor;
            if (w.dims->data[0] != k.channels || w.dims->data[3] != k.ic) return false;
        }
        else
        {
            const auto &p = *static_cast<const TfLiteDepthwiseConvParams *>(node.builtin_data);
            activation = p.activation; padding = p.padding;
            k.sh = p.stride_height; k.sw = p.stride_width;
            k.dh = p.dilation_height_factor; k.dw = p.dilation_width_factor;
            k.depthMultiplier = p.depth_multiplier;
            if (k.depthMultiplier <= 0 || int64_t(k.ic) * k.depthMultiplier != k.channels ||
                w.dims->data[0] != 1 || w.dims->data[3] != k.channels) return false;
        }
        if (k.sh <= 0 || k.sw <= 0 || k.dh <= 0 || k.dw <= 0) return false;
        const int64_t eh = int64_t(k.kh - 1) * k.dh + 1, ew = int64_t(k.kw - 1) * k.dw + 1;
        if (eh > INT32_MAX || ew > INT32_MAX) return false;
        if (padding == kTfLitePaddingSame)
        {
            if (k.oh != (int64_t(k.ih) + k.sh - 1) / k.sh || k.ow != (int64_t(k.iw) + k.sw - 1) / k.sw)
                return false;
            const int64_t ph = std::max<int64_t>(0, (int64_t(k.oh) - 1) * k.sh + eh - k.ih) / 2;
            const int64_t pw = std::max<int64_t>(0, (int64_t(k.ow) - 1) * k.sw + ew - k.iw) / 2;
            if (ph > INT32_MAX || pw > INT32_MAX) return false;
            k.ph = int(ph); k.pw = int(pw);
        }
        else if (padding != kTfLitePaddingValid || k.ih < eh || k.iw < ew ||
                 k.oh != (k.ih - eh) / k.sh + 1 || k.ow != (k.iw - ew) / k.sw + 1) return false;
    }
    const auto *q = Affine(w);
    const int axis = code == kTfLiteBuiltinDepthwiseConv2d ? 3 : 0;
    if (!q || !q->scale || !q->zero_point ||
        (q->scale->size != 1 && q->scale->size != k.channels) ||
        (q->scale->size > 1 && q->quantized_dimension != axis) ||
        (q->zero_point->size != 1 && q->zero_point->size != q->scale->size)) return false;
    for (int i = 0; i < q->zero_point->size; ++i) if (q->zero_point->data[i] != 0) return false;
    if (k.bias >= 0)
    {
        const auto &b = ctx->tensors[k.bias];
        if (b.type != kTfLiteInt32 || !StaticTensor(b) || Elements(b) != k.channels) return false;
    }
    auto quantActivation = [&](double value) {
        return int(std::clamp(std::round(value / double(Affine(y)->scale->data[0])) + k.outputZp, -128.0, 127.0));
    };
    switch (activation)
    {
        case kTfLiteActNone: break;
        case kTfLiteActRelu: k.low = std::max(-128, k.outputZp); break;
        case kTfLiteActRelu6: k.low = std::max(-128, k.outputZp); k.high = quantActivation(6); break;
        case kTfLiteActReluN1To1: k.low = quantActivation(-1); k.high = quantActivation(1); break;
        default: return false;
    }
    k.scales.resize(k.channels);
    for (int c = 0; c < k.channels; ++c)
        if (!Quantize(double(Affine(x)->scale->data[0]) * q->scale->data[c % q->scale->size] /
                      double(Affine(y)->scale->data[0]), k.scales[c])) return false;
    return true;
}

void *Init(TfLiteContext *ctx, const char *buffer, size_t)
{
    const auto *params = reinterpret_cast<const TfLiteDelegateParams *>(buffer);
    if (params->nodes_to_replace->size != 1) return nullptr;
    TfLiteNode *node = nullptr;
    TfLiteRegistration *registration = nullptr;
    if (ctx->GetNodeAndRegistration(ctx, params->nodes_to_replace->data[0], &node, &registration) != kTfLiteOk)
        return nullptr;
    auto kernel = std::make_unique<Kernel>();
    if (!Describe(ctx, *node, registration->builtin_code, *kernel)) return nullptr;
    return kernel.release();
}

void Free(TfLiteContext *, void *data) { delete static_cast<Kernel *>(data); }

TfLiteStatus PrepareKernel(TfLiteContext *ctx, TfLiteNode *node)
{
    if (!node->user_data)
    {
        ctx->ReportError(ctx, "Neural-AI: invalid static kernel configuration");
        return kTfLiteError;
    }
    const auto &k = *static_cast<const Kernel *>(node->user_data);
    const auto *x = ctx->tensors[k.input].dims;
    const auto *y = ctx->tensors[k.output].dims;
    if (!x || !y || std::vector<int>(x->data, x->data + x->size) != k.inputShape ||
        std::vector<int>(y->data, y->data + y->size) != k.outputShape)
    {
        ctx->ReportError(ctx, "Neural-AI: shape changed; create a new reference interpreter");
        return kTfLiteError;
    }
    return kTfLiteOk;
}

TfLiteStatus Invoke(TfLiteContext *ctx, TfLiteNode *node)
{
    const auto &k = *static_cast<const Kernel *>(node->user_data);
    const int8_t *input = ctx->tensors[k.input].data.int8;
    const int8_t *weight = ctx->tensors[k.weight].data.int8;
    const int32_t *bias = k.bias >= 0 ? ctx->tensors[k.bias].data.i32 : nullptr;
    int8_t *output = ctx->tensors[k.output].data.int8;
    if (!input || !weight || !output || (k.bias >= 0 && !bias)) return kTfLiteError;
    for (int n = 0; n < k.batches; ++n)
        for (int oy = 0; oy < k.oh; ++oy)
            for (int ox = 0; ox < k.ow; ++ox)
                for (int oc = 0; oc < k.channels; ++oc)
                {
                    int64_t accumulator = bias ? bias[oc] : 0;
                    for (int ky = 0; ky < k.kh; ++ky)
                    {
                        const int64_t iy = int64_t(oy) * k.sh + int64_t(ky) * k.dh - k.ph;
                        if (iy < 0 || iy >= k.ih) continue;
                        for (int kx = 0; kx < k.kw; ++kx)
                        {
                            const int64_t ix = int64_t(ox) * k.sw + int64_t(kx) * k.dw - k.pw;
                            if (ix < 0 || ix >= k.iw) continue;
                            const int64_t base = ((int64_t(n) * k.ih + iy) * k.iw + ix) * k.ic;
                            if (k.code == kTfLiteBuiltinDepthwiseConv2d)
                            {
                                accumulator += (int(input[base + oc / k.depthMultiplier]) - k.inputZp) *
                                    int(weight[(ky * k.kw + kx) * k.channels + oc]);
                            }
                            else
                            {
                                const int8_t *w = weight + ((oc * k.kh + ky) * k.kw + kx) * k.ic;
                                const int8_t *x = input + base;
                                for (int ic = 0; ic < k.ic; ++ic)
                                    accumulator += (int(x[ic]) - k.inputZp) * int(w[ic]);
                            }
                        }
                    }
                    // Reject overflow rather than silently claiming an RTL-compatible result.
                    if (accumulator < -(int64_t(1) << 33) || accumulator >= (int64_t(1) << 33))
                    {
                        ctx->ReportError(ctx, "Neural-AI: biased accumulator exceeds signed 34-bit range");
                        return kTfLiteError;
                    }
                    *output++ = Requantize(accumulator, k.scales[oc], k.outputZp, k.low, k.high);
                }
    return kTfLiteOk;
}

TfLiteStatus PrepareDelegate(TfLiteContext *ctx, TfLiteDelegate *delegate)
{
    TfLiteIntArray *plan = nullptr;
    if (ctx->GetExecutionPlan(ctx, &plan) != kTfLiteOk) return kTfLiteError;
    std::vector<int> selected;
    for (int i = 0; i < plan->size; ++i)
    {
        TfLiteNode *node = nullptr;
        TfLiteRegistration *registration = nullptr;
        if (ctx->GetNodeAndRegistration(ctx, plan->data[i], &node, &registration) != kTfLiteOk) return kTfLiteError;
        Kernel kernel;
        if (Describe(ctx, *node, registration->builtin_code, kernel)) selected.push_back(plan->data[i]);
    }
    TfLiteRegistration registration{};
    registration.init = Init; registration.free = Free;
    registration.prepare = PrepareKernel; registration.invoke = Invoke;
    registration.builtin_code = kTfLiteBuiltinDelegate;
    registration.custom_name = "NeuralAiReference";
    registration.version = 1;
    // One partition per op: all intermediate tensors remain ordinary CPU tensors,
    // including adjacent Conv/FC nodes. This also enables per-tensor diagnosis.
    for (int index : selected)
    {
        alignas(TfLiteIntArray) int storage[2] = {1, index};
        auto *nodes = reinterpret_cast<TfLiteIntArray *>(storage);
        if (ctx->ReplaceNodeSubsetsWithDelegateKernels(ctx, registration, nodes, delegate) != kTfLiteOk)
            return kTfLiteError;
    }
    return kTfLiteOk;
}
} // namespace

extern "C" TFL_EXTERNAL_DELEGATE_EXPORT TfLiteDelegate *tflite_plugin_create_delegate(
    const char *const *, const char *const *, size_t count, void (*reportError)(const char *))
{
    if (count)
    {
        if (reportError) reportError("Neural-AI reference accepts no delegate options");
        return nullptr;
    }
    auto *delegate = new TfLiteDelegate{};
    delegate->Prepare = PrepareDelegate;
    return delegate;
}

extern "C" TFL_EXTERNAL_DELEGATE_EXPORT void tflite_plugin_destroy_delegate(TfLiteDelegate *delegate)
{
    delete delegate;
}

extern "C" TFL_EXTERNAL_DELEGATE_EXPORT int neural_ai_reference_requant(
    int64_t accumulator, double scale, int zp, int low, int high)
{
    Scale quantized;
    if (!Quantize(scale, quantized) || low < -128 || high > 127 || low > high) return INT32_MIN;
    return Requantize(accumulator, quantized, zp, low, high);
}
