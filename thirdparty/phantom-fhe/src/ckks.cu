#include "ckks.h"
#include "fft.h"

#include <mutex>
#include <unordered_map>

using namespace std;
using namespace phantom;
using namespace phantom::util;
using namespace phantom::arith;

__global__ void bit_reverse_and_zero_padding(cuDoubleComplex *dst, cuDoubleComplex *src, uint64_t in_size,
                                             uint32_t slots, uint32_t logn) {
    for (uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < slots;
         tid += blockDim.x * gridDim.x) {
        if (tid < uint32_t(in_size)) {
            dst[reverse_bits_uint32(tid, logn)] = src[tid];
        } else {
            dst[reverse_bits_uint32(tid, logn)] = (cuDoubleComplex) {0.0, 0.0};
        }
    }
}

__global__ void bit_reverse(cuDoubleComplex *dst, cuDoubleComplex *src, uint32_t slots, uint32_t logn) {
    for (uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
         tid < slots;
         tid += blockDim.x * gridDim.x) {
        dst[reverse_bits_uint32(tid, logn)] = src[tid];
    }
}

PhantomCKKSEncoder::PhantomCKKSEncoder(const PhantomContext &context) {
    const auto &s = global_variables::default_stream->get_stream();

    auto &context_data = context.get_context_data(first_chain_index_);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    std::size_t coeff_count = parms.poly_modulus_degree();

    if (parms.scheme() != scheme_type::ckks) {
        throw std::invalid_argument("unsupported scheme");
    }
    slots_ = coeff_count >> 1; // n/2

    // Newly added: set sparse_slots immediately if specified
    auto specified_sparse_slots = context_data.parms().sparse_slots();
    if (specified_sparse_slots) {
        // cout << "Setting decoding sparse slots to: " << specified_sparse_slots << endl;
        decoding_sparse_slots_ = specified_sparse_slots;
    }

    uint32_t m = coeff_count << 1;
    uint32_t slots_half = slots_ >> 1;
    gpu_ckks_msg_vec_ = std::make_unique<DCKKSEncoderInfo>(coeff_count, s);

    // We need m powers of the primitive 2n-th root, m = 2n
    root_powers_.reserve(m);
    rotation_group_.reserve(slots_half);

    uint32_t gen = 5;
    uint32_t pos = 1; // Position in normal bit order
    for (size_t i = 0; i < slots_half; i++) {
        // Set the bit-reversed locations
        rotation_group_[i] = pos;

        // Next primitive root
        pos *= gen; // 5^i mod m
        pos &= (m - 1);
    }

    // Powers of the primitive 2n-th root have 4-fold symmetry
    if (m >= 8) {
        complex_roots_ = std::make_unique<util::ComplexRoots>(util::ComplexRoots(static_cast<size_t>(m)));
        for (size_t i = 0; i < m; i++) {
            root_powers_[i] = complex_roots_->get_root(i);
        }
    } else if (m == 4) {
        root_powers_[0] = {1, 0};
        root_powers_[1] = {0, 1};
        root_powers_[2] = {-1, 0};
        root_powers_[3] = {0, -1};
    }

    cudaMemcpyAsync(gpu_ckks_msg_vec_->twiddle(), root_powers_.data(), m * sizeof(cuDoubleComplex),
                    cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(gpu_ckks_msg_vec_->mul_group(), rotation_group_.data(), slots_half * sizeof(uint32_t),
                    cudaMemcpyHostToDevice, s);
}

void PhantomCKKSEncoder::encode_internal(const PhantomContext &context, const cuDoubleComplex *values,
                                         size_t values_size, size_t chain_index, double scale,
                                         PhantomPlaintext &destination, const cudaStream_t &stream) {
    auto &context_data = context.get_context_data(chain_index);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto &rns_tool = context_data.gpu_rns_tool();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    std::size_t coeff_count = parms.poly_modulus_degree();

    if (!values && values_size > 0) {
        throw std::invalid_argument("values cannot be null");
    }
    if (values_size > slots_) {
        throw std::invalid_argument("values_size is too large");
    }

    // Check that scale is positive and not too large
    if (scale <= 0 || (static_cast<int>(log2(scale)) + 1 >= context_data.total_coeff_modulus_bit_count())) {
        throw std::invalid_argument("scale out of bounds");
    }

    if (sparse_slots_ == 0) {
        uint32_t log_sparse_slots = ceil(log2(values_size));
        sparse_slots_ = 1 << log_sparse_slots;
    } else {
        // Newly commented, not sure if we need this:
        // if (values_size > sparse_slots_) {
        //     throw std::invalid_argument("values_size exceeds previous message length: " + std::to_string(values_size) + " > " + std::to_string(sparse_slots_));
        // }
    }
    // size_t log_sparse_slots = ceil(log2(slots_));
    // sparse_slots_ = slots_;
    if (sparse_slots_ < 2) {
        throw std::invalid_argument("single value encoding is not available");
    }

    gpu_ckks_msg_vec_->set_sparse_slots(sparse_slots_);
    PHANTOM_CHECK_CUDA(cudaMemsetAsync(gpu_ckks_msg_vec_->in(), 0, slots_ * sizeof(cuDoubleComplex), stream));
    auto temp = make_cuda_auto_ptr<cuDoubleComplex>(values_size, stream);
    PHANTOM_CHECK_CUDA(cudaMemsetAsync(temp.get(), 0, values_size * sizeof(cuDoubleComplex), stream));
    PHANTOM_CHECK_CUDA(
            cudaMemcpyAsync(temp.get(), values, sizeof(cuDoubleComplex) * values_size, cudaMemcpyHostToDevice, stream));

    uint32_t log_sparse_n = log2(sparse_slots_);
    uint64_t gridDimGlb = ceil(sparse_slots_ / blockDimGlb.x);
    bit_reverse_and_zero_padding<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            gpu_ckks_msg_vec_->in(), temp.get(), values_size, sparse_slots_, log_sparse_n);

    double fix = scale / static_cast<double>(sparse_slots_);

    // same as SEAL's fft_handler_.transform_from_rev
    special_fft_backward(*gpu_ckks_msg_vec_, fix, stream);

    // TODO: boundary check on GPU
    vector<cuDoubleComplex> temp2(sparse_slots_);
    PHANTOM_CHECK_CUDA(cudaMemcpyAsync(temp2.data(), gpu_ckks_msg_vec_->in(), sparse_slots_ * sizeof(cuDoubleComplex),
                                       cudaMemcpyDeviceToHost, stream));
    // explicit stream synchronize to avoid error
    cudaStreamSynchronize(stream);

    double max_coeff = 0;
    for (std::size_t i = 0; i < sparse_slots_; i++) {
        max_coeff = std::max(max_coeff, std::fabs(temp2[i].x));
    }
    for (std::size_t i = 0; i < sparse_slots_; i++) {
        max_coeff = std::max(max_coeff, std::fabs(temp2[i].y));
    }
    // Verify that the values are not too large to fit in coeff_modulus
    // Note that we have an extra + 1 for the sign bit
    // Don't compute logarithmis of numbers less than 1
    int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max(max_coeff, 1.0)))) + 1;

    if (max_coeff_bit_count >= context_data.total_coeff_modulus_bit_count()) {
        throw std::invalid_argument("encoded values are too large");
    }

    // we can in fact find all coeff_modulus in DNTTTable structure....
    rns_tool.base_Ql().decompose_array(destination.data(), gpu_ckks_msg_vec_->in(), sparse_slots_ << 1,
                                       (uint32_t) slots_ / sparse_slots_, max_coeff_bit_count, stream);

    nwt_2d_radix8_forward_inplace(destination.data(), context.gpu_rns_tables(), coeff_modulus_size, 0, stream);

    destination.chain_index_ = chain_index;
    destination.scale_ = scale;
}

void PhantomCKKSEncoder::decode_internal(const PhantomContext &context, const PhantomPlaintext &plain,
                                         cuDoubleComplex *destination, const cudaStream_t &stream) {
    if (!destination) {
        throw std::invalid_argument("destination cannot be null");
    }

    auto &context_data = context.get_context_data(plain.chain_index_);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto &rns_tool = context_data.gpu_rns_tool();
    const size_t coeff_modulus_size = coeff_modulus.size();
    const size_t coeff_count = parms.poly_modulus_degree();
    const size_t rns_poly_uint64_count = coeff_count * coeff_modulus_size;

    if (plain.scale() <= 0 ||
        (static_cast<int>(log2(plain.scale())) >= context_data.total_coeff_modulus_bit_count())) {
        throw std::invalid_argument("scale out of bounds");
    }

    auto upper_half_threshold = context_data.upper_half_threshold();
    int logn = arith::get_power_of_two(coeff_count);
    auto gpu_upper_half_threshold = make_cuda_auto_ptr<uint64_t>(upper_half_threshold.size(), stream);
    cudaMemcpyAsync(gpu_upper_half_threshold.get(), upper_half_threshold.data(),
                    upper_half_threshold.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);

    gpu_ckks_msg_vec_->set_sparse_slots(sparse_slots_);
    cudaMemsetAsync(gpu_ckks_msg_vec_->in(), 0, slots_ * sizeof(cuDoubleComplex), stream);

    // Quick sanity check
    if ((logn < 0) || (coeff_count < POLY_MOD_DEGREE_MIN) || (coeff_count > POLY_MOD_DEGREE_MAX)) {
        throw std::logic_error("invalid parameters");
    }

    double inv_scale = double(1.0) / plain.scale();
    // Create mutable copy of input
    auto plain_copy = make_cuda_auto_ptr<uint64_t>(rns_poly_uint64_count, stream);
    cudaMemcpyAsync(plain_copy.get(), plain.data(), rns_poly_uint64_count * sizeof(uint64_t), cudaMemcpyDeviceToDevice,
                    stream);

    nwt_2d_radix8_backward_inplace(plain_copy.get(), context.gpu_rns_tables(), coeff_modulus_size, 0, stream);
    // cout << "[DEBUG] decoding_sparse_slots: " << decoding_sparse_slots_ << std::endl;
    // cout << "[DEBUG] sparse_slots: " << sparse_slots_ << std::endl;
    // cout << "[DEBUG] coeff_count: " << coeff_count << std::endl;
    // CRT-compose the polynomial
    if (decoding_sparse_slots_) {
        rns_tool.base_Ql().compose_array(gpu_ckks_msg_vec().in(), plain_copy.get(), gpu_upper_half_threshold.get(),
                                         inv_scale, coeff_count, sparse_slots_ << 1, slots_ / sparse_slots_,
                                         slots_ / decoding_sparse_slots_, stream);
    } else {
        rns_tool.base_Ql().compose_array(gpu_ckks_msg_vec().in(), plain_copy.get(), gpu_upper_half_threshold.get(),
                                         inv_scale, coeff_count, sparse_slots_ << 1, slots_ / sparse_slots_, stream);
    }

    special_fft_forward(*gpu_ckks_msg_vec_, stream);

    // finally, bit-reverse and output
    auto out = make_cuda_auto_ptr<cuDoubleComplex>(sparse_slots_, stream);
    uint32_t log_sparse_n = log2(sparse_slots_);
    size_t gridDimGlb = ceil(sparse_slots_ / blockDimGlb.x);
    bit_reverse<<<gridDimGlb, blockDimGlb, 0, stream>>>(
            out.get(), gpu_ckks_msg_vec_->in(), sparse_slots_, log_sparse_n);

    if (decoding_sparse_slots_) {
        cudaMemcpyAsync(destination, out.get(), decoding_sparse_slots_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream);
    } else {
        cudaMemcpyAsync(destination, out.get(), sparse_slots_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost, stream);
    }

    // explicit synchronization in case user wants to use the result immediately
    cudaStreamSynchronize(stream);
}

namespace {

struct CkksUniformSlotBasisCache {
    phantom::util::cuda_auto_ptr<cuDoubleComplex> basis{};
    double max_abs = 0.0;
};

std::mutex g_ckks_uniform_basis_mu;
std::unordered_map<uint32_t, CkksUniformSlotBasisCache> g_ckks_uniform_basis_by_slots;

__global__ void ckks_fill_unit_real_kernel(cuDoubleComplex *dst, uint32_t n) {
    for (uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) {
        dst[tid] = make_cuDoubleComplex(1.0, 0.0);
    }
}

__global__ void ckks_scale_complex_by_real_kernel(const cuDoubleComplex *src, cuDoubleComplex *dst, uint32_t n,
                                                  double s) {
    for (uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x) {
        cuDoubleComplex b = src[tid];
        dst[tid] = make_cuDoubleComplex(b.x * s, b.y * s);
    }
}

const CkksUniformSlotBasisCache &get_ckks_uniform_basis(uint32_t slots, DCKKSEncoderInfo &gp, const cudaStream_t &stream) {
    std::lock_guard<std::mutex> lock(g_ckks_uniform_basis_mu);
    auto [it, inserted] = g_ckks_uniform_basis_by_slots.try_emplace(slots);
    CkksUniformSlotBasisCache &cache = it->second;
    if (inserted) {
        cache.basis = make_cuda_auto_ptr<cuDoubleComplex>(slots, stream);
        gp.set_sparse_slots(slots);
        PHANTOM_CHECK_CUDA(cudaMemsetAsync(gp.in(), 0, slots * sizeof(cuDoubleComplex), stream));
        const uint32_t grid = (slots + blockDimGlb.x - 1) / blockDimGlb.x;
        ckks_fill_unit_real_kernel<<<grid, blockDimGlb, 0, stream>>>(gp.in(), slots);
        special_fft_backward(gp, 1.0, stream);
        PHANTOM_CHECK_CUDA(cudaMemcpyAsync(cache.basis.get(), gp.in(), slots * sizeof(cuDoubleComplex),
                                           cudaMemcpyDeviceToDevice, stream));
        PHANTOM_CHECK_CUDA(cudaStreamSynchronize(stream));
        vector<cuDoubleComplex> host(slots);
        PHANTOM_CHECK_CUDA(
                cudaMemcpy(host.data(), cache.basis.get(), slots * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < slots; ++i) {
            cache.max_abs = std::max(cache.max_abs, std::fabs(host[i].x));
            cache.max_abs = std::max(cache.max_abs, std::fabs(host[i].y));
        }
    }
    return cache;
}

} // namespace

void PhantomCKKSEncoder::encode_internal_uniform_real(const PhantomContext &context, double value, size_t chain_index,
                                                      double scale, PhantomPlaintext &destination,
                                                      const cudaStream_t &stream) {
    auto &context_data = context.get_context_data(chain_index);
    auto &parms = context_data.parms();
    auto &coeff_modulus = parms.coeff_modulus();
    auto &rns_tool = context_data.gpu_rns_tool();
    std::size_t coeff_modulus_size = coeff_modulus.size();
    std::size_t coeff_count = parms.poly_modulus_degree();

    if (parms.scheme() != scheme_type::ckks) {
        throw std::invalid_argument("unsupported scheme");
    }
    if (scale <= 0 || (static_cast<int>(log2(scale)) + 1 >= context_data.total_coeff_modulus_bit_count())) {
        throw std::invalid_argument("scale out of bounds");
    }
    if (slots_ < 2) {
        throw std::invalid_argument("uniform real encoding unavailable for slots < 2");
    }

    sparse_slots_ = slots_;
    gpu_ckks_msg_vec_->set_sparse_slots(slots_);

    const CkksUniformSlotBasisCache &basis_cache = get_ckks_uniform_basis(slots_, *gpu_ckks_msg_vec_, stream);

    const double fix = scale / static_cast<double>(slots_);
    const double scaled_value = value * fix;
    const uint32_t grid_dim = (slots_ + blockDimGlb.x - 1) / blockDimGlb.x;
    ckks_scale_complex_by_real_kernel<<<grid_dim, blockDimGlb, 0, stream>>>(
            basis_cache.basis.get(), gpu_ckks_msg_vec_->in(), slots_, scaled_value);

    const double max_coeff = std::fabs(scaled_value) * basis_cache.max_abs;
    const int max_coeff_bit_count = static_cast<int>(std::ceil(std::log2(std::max(max_coeff, 1.0)))) + 1;

    if (max_coeff_bit_count >= context_data.total_coeff_modulus_bit_count()) {
        throw std::invalid_argument("encoded values are too large");
    }

    rns_tool.base_Ql().decompose_array(destination.data(), gpu_ckks_msg_vec_->in(), slots_ << 1, 1u,
                                         max_coeff_bit_count, stream);

    nwt_2d_radix8_forward_inplace(destination.data(), context.gpu_rns_tables(), coeff_modulus_size, 0, stream);

    destination.chain_index_ = chain_index;
    destination.scale_ = scale;
}
