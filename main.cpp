#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <iostream>
#include "avx_mathfun.h"
#include <chrono>
#include <unordered_map>

using namespace std::chrono;

std::unordered_map<std::string, high_resolution_clock::time_point> t_stamps;
high_resolution_clock::time_point t_now = high_resolution_clock::now();

constexpr size_t VECTOR_LEN = 1UL<<16;
constexpr size_t ALIGN = 32;
constexpr size_t ITER_COUNT = 1000;
constexpr float EPSILON = 1e-5;

void
stamp(const std::string& name = "") {
  t_stamps[name] = high_resolution_clock::now();
}

void
show_duration_since_stamp(const std::string& routine_name, const std::string& unit, const std::string& stamp_name = "") {
  t_now = high_resolution_clock::now();
  if (routine_name == "" && stamp_name == "") {
    std::cout << "--- Time for default clock is ";
  } else if (routine_name == ""){
    std::cout << "--- Time for " << stamp_name << " is ";
  } else {
    std::cout << "--- Time for " << routine_name << " is ";
  }
  if (unit == "second" || unit == "s") {
    auto dur = duration_cast<duration<float, std::ratio<1, 1> > >(t_now - t_stamps[stamp_name]);
    std::cout << dur.count()/ITER_COUNT << "s" << std::endl;
  } else if (unit == "millisecond" || unit == "ms") {
    auto dur = duration_cast<milliseconds>(t_now - t_stamps[stamp_name]);
    std::cout << dur.count()/ITER_COUNT << "ms" << std::endl;
  } else if (unit == "microsecond" || unit == "us") {
    auto dur = duration_cast<microseconds>(t_now - t_stamps[stamp_name]);
    std::cout << dur.count()/ITER_COUNT << "µs" << std::endl;
  } else if (unit == "nanosecond" || unit == "ns") {
    auto dur = duration_cast<nanoseconds>(t_now - t_stamps[stamp_name]);
    std::cout << dur.count()/ITER_COUNT << "ns" << std::endl;
  }
}

class AVXMathfunTest : public ::testing::Test {
private:
    template <typename T>
    inline void allocMemory(T** data) {
        posix_memalign(reinterpret_cast<void**>(data), ALIGN, VECTOR_LEN*sizeof(float));
    }

#define allocMem(func)\
    do {\
        allocMemory(& func##NaiveResult_);\
        allocMemory(& func##SimdResult_);\
    } while(0)

#define freeMem(func)\
    do {\
        free(func##NaiveResult_);\
        free(func##SimdResult_);\
    } while(0)


protected:
    virtual void TearDown() {
        free(inputData_);
        freeMem(exp);
        freeMem(log);
        freeMem(sin);
        freeMem(cos);
    }

    virtual void SetUp() {
        allocMemory(&inputData_);
        allocMem(exp);
        allocMem(log);
        allocMem(sin);
        allocMem(cos);


        std::random_device dev;
        std::mt19937_64 eng;
        eng.seed(dev());
        std::uniform_real_distribution<float> distribution(0, 1);

        for (size_t i = 0; i < VECTOR_LEN; ++i) {
            float tmp = distribution(eng);
            inputData_[i] = tmp;
        }
        stamp();
        for (int n = 0; n < ITER_COUNT; ++n) {
          for (size_t i = 0; i < VECTOR_LEN; ++i) {
            expNaiveResult_[i] = std::exp(inputData_[i]);
          }
        }
        show_duration_since_stamp("exp", "us");
        stamp();
        for (int n = 0; n < ITER_COUNT; ++n) {
          for (size_t i = 0; i < VECTOR_LEN; ++i) {
            logNaiveResult_[i] = std::log(inputData_[i]);
          }
        }
        show_duration_since_stamp("log", "us");
        stamp();
        for (int n = 0; n < ITER_COUNT; ++n) {
          for (size_t i = 0; i < VECTOR_LEN; ++i) {
            sinNaiveResult_[i] = std::sin(inputData_[i]);
          }
        }
        show_duration_since_stamp("sin", "us");
        stamp();
        for (int n = 0; n < ITER_COUNT; ++n) {
          for (size_t i = 0; i < VECTOR_LEN; ++i) {
            cosNaiveResult_[i] = std::cos(inputData_[i]);
          }
        }
        show_duration_since_stamp("cos", "us");

        // std::cout << "Working on " << VECTOR_LEN << " numbers" << std::endl;
    }

    float* inputData_;
    float* expNaiveResult_;
    float* expSimdResult_;
    float* logNaiveResult_;
    float* logSimdResult_;
    float* sinNaiveResult_;
    float* sinSimdResult_;
    float* cosNaiveResult_;
    float* cosSimdResult_;
};

#define TEST_AVX(func)\
TEST_F(AVXMathfunTest, func) {\
    stamp(#func);                                      \
    for (size_t i = 0; i < ITER_COUNT; ++i) {\
        __m256 tmp;\
        __m256 ipt;\
        for (size_t j = 0; j < VECTOR_LEN; j+= 8) {\
            ipt = _mm256_load_ps(inputData_ + j);\
            tmp = func##256_ps(ipt);\
            _mm256_store_ps(func##SimdResult_ + j, tmp);\
        }\
    }\
    show_duration_since_stamp(#func, "us", #func);\
\
    for (size_t i = 0; i < VECTOR_LEN; ++i) {\
        ASSERT_NEAR(func##NaiveResult_[i], func##SimdResult_[i], EPSILON);\
    }\
}

TEST_AVX(exp)
TEST_AVX(log)
TEST_AVX(sin)
TEST_AVX(cos)
