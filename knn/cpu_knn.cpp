#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

static int s_AllocationCount = 0;
static int total_size = 0;

void* operator new(size_t size){
    // std::cout<<"Allocated "<<size<<" bytes\n";
    s_AllocationCount++;
    total_size += size;
    return malloc(size);
}
class Timer {
public:
    explicit Timer(const std::string& name)
        : name_(name), start_(std::chrono::steady_clock::now()) {}
    ~Timer() {
        const auto end = std::chrono::steady_clock::now();
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        std::cout << name_ << " took " << us << " us (" << (us / 1000.0) << " ms)\n";
    }
private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

static inline size_t idx4(size_t b, size_t t, size_t n, size_t d,
                          size_t B, size_t T, size_t N, size_t D) {
    return (((b * T + t) * N + n) * D + d);
}

struct DistIdx {
    float dist;
    int   idx;
};

// Squared L2 distance between point i and j within (b,t)
static inline float l2_sq(const std::vector<float>& x,
                          size_t b, size_t t, size_t i, size_t j,
                          size_t B, size_t T, size_t N, size_t D) {
    float acc = 0.0f;
    const size_t base_i = idx4(b, t, i, 0, B, T, N, D);
    const size_t base_j = idx4(b, t, j, 0, B, T, N, D);
    for (size_t d = 0; d < D; ++d) {
        const float diff = x[base_i + d] - x[base_j + d];
        acc += diff * diff;
    }
    return acc;
}

// (B,T,N,K) flattened as [(((b*T+t)*N+i)*K + k)]
void knn_naive_cpu_inplace(
    const std::vector<float>& x,
    size_t B, size_t T, size_t N, size_t D, size_t K,
    std::vector<int>& out,
    std::vector<DistIdx>& buf)
{
    assert(K < N);

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            for (size_t i = 0; i < N; ++i) {

                for (size_t j = 0; j < N; ++j) {
                    float dist = (j == i)
                        ? std::numeric_limits<float>::infinity()
                        : l2_sq(x, b, t, i, j, B, T, N, D);
                    buf[j] = DistIdx{dist, static_cast<int>(j)};
                }

                auto kth = buf.begin() + static_cast<ptrdiff_t>(K);
                std::nth_element(buf.begin(), kth, buf.end(),
                    [](const DistIdx& a, const DistIdx& b) { return a.dist < b.dist; });

                std::sort(buf.begin(), kth,
                    [](const DistIdx& a, const DistIdx& b) { return a.dist < b.dist; });

                const size_t out_base = (((b * T + t) * N + i) * K);
                for (size_t k = 0; k < K; ++k)
                    out[out_base + k] = buf[k].idx;
            }
        }
    }
}

std::vector<int> knn_symmetric_cpu_bt(
    const std::vector<float>& x,
    size_t B, size_t T, size_t N, size_t D,
    size_t K)
{

    std::vector<int> out(B * T * N * K);

    for (size_t b = 0; b < B; ++b) {
        for (size_t t = 0; t < T; ++t) {
            std::vector<std::vector<DistIdx>> knn(N);
            for (size_t i = 0; i < N; ++i)
                knn[i].reserve(K);

            // Only compute upper triangle as (i,j) == (j,i)
            for (size_t i = 0; i < N; ++i) {
                for (size_t j = i + 1; j < N; ++j) {

                    float dist = l2_sq(x, b, t, i, j, B, T, N, D);

                    // ---- update i ----
                    if (knn[i].size() < K) {
                        knn[i].push_back({dist, static_cast<int>(j)});
                    } else {
                        size_t worst = 0;
                        for (size_t k = 1; k < K; ++k)
                            if (knn[i][k].dist > knn[i][worst].dist)
                                worst = k;

                        if (dist < knn[i][worst].dist)
                            knn[i][worst] = {dist, static_cast<int>(j)};
                    }

                    //update j
                    if (knn[j].size() < K) {
                        knn[j].push_back({dist, static_cast<int>(i)});
                    } else {
                        size_t worst = 0;
                        for (size_t k = 1; k < K; ++k)
                            if (knn[j][k].dist > knn[j][worst].dist)
                                worst = k;

                        if (dist < knn[j][worst].dist)
                            knn[j][worst] = {dist, static_cast<int>(i)};
                    }
                }
            }

            // Sort and write output
            for (size_t i = 0; i < N; ++i) {
                std::sort(knn[i].begin(), knn[i].end(),
                          [](const DistIdx& a, const DistIdx& b) {
                              return a.dist < b.dist;
                          });

                size_t base = (((b * T + t) * N + i) * K);
                for (size_t k = 0; k < K; ++k)
                    out[base + k] = knn[i][k].idx;
            }
        }
    }

    return out;
}

std::vector<float> make_random_points(size_t B, size_t T, size_t N, size_t D, uint32_t seed = 123) {
    std::vector<float> x(B * T * N * D);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) v = dist(rng);
    return x;
}

int main() {
    const size_t B = 16;
    const size_t T = 1;
    const size_t N = 4096;

    const std::vector<size_t> Ds = {3};
    const std::vector<size_t> Ks = {10};

    // Warmup note: first run may be slower due to CPU frequency scaling and cache effects.
    for (size_t D : Ds) {
        std::cout << "\n=== D = " << D << " ===\n";
        auto x = make_random_points(B, T, N, D, 123);
        std::cout<<"--------- WARM UP ---------\n";
        for(int i=0; i<5;i++)
        {
            std::vector<int> out;
            std::vector<DistIdx> buf;
            out.reserve(B * T * N * 20);
            buf.reserve(N);
            knn_naive_cpu_inplace(x, B, T, N, D, 10, out, buf);
            volatile int checksum1 = 0;
            for (size_t i = 0; i < std::min<size_t>(out.size(), 1024); ++i) checksum1 += out[i];
        }
        std::cout<<"WARMUP: "<<s_AllocationCount<<" allocations\n";
        std::cout<<"WARMUP: "<<total_size<<" bytes allocated\n";

            std::cout<<"--------- WARM UP ---------\n";
        for (size_t K : Ks) {
            {
                Timer timer1("CPU naive KNN (B=16,T=1,N=4096,D=" + std::to_string(D) +
                            ",K=" + std::to_string(K) + ")");
                std::vector<int> out;
                std::vector<DistIdx> buf;
                out.reserve(B * T * N * 20); // reserve for max K if you want
                buf.reserve(N);              // reserve once

                knn_naive_cpu_inplace(x, B, T, N, D, K, out, buf);
                volatile int checksum1 = 0;
                for (size_t i = 0; i < std::min<size_t>(out.size(), 1024); ++i) checksum1 += out[i];
                std::cout<<"NAIVE: "<<s_AllocationCount<<" allocations\n";
                std::cout<<"NAIVE: "<<total_size<<" bytes allocated\n";
            }
        }

        for (size_t K : Ks) {
            {
                
                Timer timer2("CPU symmetric KNN (B=16,T=1,N=4096,D=" + std::to_string(D) +
                            ",K=" + std::to_string(K) + ")");
                auto out2 = knn_symmetric_cpu_bt(x, B, T, N, D, K);

                volatile int checksum2 = 0;
                for (size_t i = 0; i < std::min<size_t>(out2.size(), 1024); ++i) checksum2 += out2[i];
                std::cout<<"SYMM: "<<s_AllocationCount<<" allocations\n";
                std::cout<<"SYMM: "<<total_size<<" bytes allocated\n";
            }
        }
    }
    return 0;
    }