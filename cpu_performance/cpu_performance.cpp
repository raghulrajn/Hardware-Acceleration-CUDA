#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <immintrin.h>
#include <cstring>
#include <cstdlib>
#include <cstdint>

class Timer {
public:
    Timer(const std::string& name)
        : name_(name),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        std::cout << name_ << " took "
                  << duration.count() << " us\n";
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

float dot_scalar(const float* a, 
                 const float* b, 
                 size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i)
        sum += a[i] * b[i];
    return sum;
}


float dot_avx(const float* a, const float* b, size_t n) {
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 prod = _mm256_mul_ps(va, vb);
        sum_vec = _mm256_add_ps(sum_vec, prod);
    }

    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);

    float result=0;
    for(auto t:temp) result+=t;

    for (; i < n; ++i)
        result += a[i] * b[i];

    return result;
}

float dot_avx_restrict(const float* __restrict__ a,
                       const float* __restrict__ b, size_t n) {
    __m256 sum_vec = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 7 < n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 prod = _mm256_mul_ps(va, vb);
        sum_vec = _mm256_add_ps(sum_vec, prod);
    }

    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);

    float result=0;
    for(auto t:temp) result+=t;

    for (; i < n; ++i)
        result += a[i] * b[i];

    return result;
}

int main() {
    const size_t N = 10000;
    const int iterations = 10000;

    std::vector<float> a(N), b(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (size_t i = 0; i < N; ++i) {
        a[i] = dist(rng);
        b[i] = dist(rng);
    }

    

    {
        float result = 0.0f;
        Timer t("Scalar");
        for (int i = 0; i < iterations; ++i)
            result += dot_scalar(a.data(), b.data(), N);
        std::cout<<result<<std::endl;
    }

    {
        float result = 0.0f;
        Timer t("AVX Intrinsics");
        for (int i = 0; i < iterations; ++i)
            result += dot_avx(a.data(), b.data(), N);
        std::cout<<result<<std::endl;
    }

    {
        float result = 0.0f;
        Timer t("AVX Intrinsics with restrict");
        for (int i = 0; i < iterations; ++i)
            result += dot_avx_restrict(a.data(), b.data(), N);
        std::cout<<result<<std::endl;
    }

    return 0;

}
