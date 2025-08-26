// cpu_gemm.cpp — 2D-only, naive 3-loop GEMM: C = A * B

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>

struct Matrix2D {
  int rows = 0, cols = 0, ld = 0;   // ld = rows (column-major stride)
  float* data = nullptr;
  bool owner = false;

  Matrix2D() = default;
  Matrix2D(int r, int c): rows(r), cols(c), ld(r), owner(true) {
    void* p = nullptr;
#if defined(_MSC_VER)
    p = _aligned_malloc((size_t)ld * c * sizeof(float), 64);
    if (!p) throw std::bad_alloc();
#else
    if (posix_memalign(&p, 64, (size_t)ld * c * sizeof(float)) != 0) throw std::bad_alloc();
#endif
    data = static_cast<float*>(p);
  }

  ~Matrix2D() {
    if (owner && data) {
#if defined(_MSC_VER)
      _aligned_free(data);
#else
      free(data);
#endif
    }
  }

  inline float& at(int r, int c) { return data[r + c * ld]; }          // column-major
  inline const float& at(int r, int c) const { return data[r + c * ld]; }

  void fill_zero() { std::memset(data, 0, (size_t)ld * cols * sizeof(float)); }
};

static void InitializeMatrix(Matrix2D& M, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(-8, 7);
  for (int j = 0; j < M.cols; ++j)
    for (int i = 0; i < M.rows; ++i)
      M.at(i, j) = static_cast<float>(dist(rng));
}

// Naive 3-loop GEMM: C = A * B (i, j, k)
static int Gemm3Loops(const Matrix2D& A, const Matrix2D& B, Matrix2D& C) {
  if (A.cols != B.rows || A.rows != C.rows || B.cols != C.cols) {
    std::cerr << "Dimension mismatch.\n"; return 1;
  }
  const int M = C.rows, N = C.cols, K = A.cols;

  C.fill_zero();

  auto t0 = std::chrono::steady_clock::now();

  for (int i = 0; i < M; ++i) {         // rows of C / A
    for (int j = 0; j < N; ++j) {       // cols of C / B
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {     // inner product
        acc += A.at(i, k) * B.at(k, j);
      }
      C.at(i, j) = acc;
    }
  }

  auto t1 = std::chrono::steady_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  double gflops = (2.0 * (double)M * (double)N * (double)K) / sec / 1e9;

  std::cout << "[Perf] 3-loop GEMM time = " << std::fixed << std::setprecision(6)
            << sec << " s  (" << std::setprecision(2) << gflops << " GFLOP/s)\n";
  return 0;
}

int main(int argc, const char* argv[]) {
  // Define rows/cols first
  int M = 0, N = 0, K = 0;

  if (argc >= 4) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
  } else {
    std::cout << "Enter M N K: ";
    std::cin >> M >> N >> K;
  }

  if (M <= 0 || N <= 0 || K <= 0) {
    std::cerr << "Invalid sizes.\n";
    return 1;
  }

  // Memory estimate (A + B + C)
  double gib = ((double)M * K + (double)K * N + (double)M * N) * sizeof(float)
               / (1024.0 * 1024.0 * 1024.0);
  std::cout << "[Info] Estimated memory ~ " << std::fixed << std::setprecision(2)
            << gib << " GiB\n";

  try {
    Matrix2D A(M, K), B(K, N), C(M, N);
    InitializeMatrix(A, 1);
    InitializeMatrix(B, 2);

    if (Gemm3Loops(A, B, C) != 0) return 1;

    std::cout << "Done.\n";
  } catch (const std::bad_alloc&) {
    std::cerr << "Allocation failed. Reduce sizes.\n";
    return 1;
  }

  return 0;
}
