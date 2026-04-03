#include "sokolov_k_matrix_double_fox/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "sokolov_k_matrix_double_fox/common/include/common.hpp"
#include "util/include/util.hpp"

namespace sokolov_k_matrix_double_fox {

namespace {

void DecomposeToBlocks(const std::vector<double> &flat, std::vector<double> &blocks, int n, int bs, int q) {
  for (int bi = 0; bi < q; bi++) {
    for (int bj = 0; bj < q; bj++) {
      int block_off = ((bi * q) + bj) * (bs * bs);
      for (int i = 0; i < bs; i++) {
        for (int j = 0; j < bs; j++) {
          blocks[block_off + (i * bs) + j] = flat[(((bi * bs) + i) * n) + ((bj * bs) + j)];
        }
      }
    }
  }
}

void AssembleFromBlocks(const std::vector<double> &blocks, std::vector<double> &flat, int n, int bs, int q) {
  for (int bi = 0; bi < q; bi++) {
    for (int bj = 0; bj < q; bj++) {
      int block_off = ((bi * q) + bj) * (bs * bs);
      for (int i = 0; i < bs; i++) {
        for (int j = 0; j < bs; j++) {
          flat[(((bi * bs) + i) * n) + ((bj * bs) + j)] = blocks[block_off + (i * bs) + j];
        }
      }
    }
  }
}

void MultiplyBlocks(const std::vector<double> &a, int a_off, const std::vector<double> &b, int b_off,
                    std::vector<double> &c, int c_off, int bs) {
  for (int i = 0; i < bs; i++) {
    for (int k = 0; k < bs; k++) {
      double val = a[a_off + (i * bs) + k];
      for (int j = 0; j < bs; j++) {
        c[c_off + (i * bs) + j] += val * b[b_off + (k * bs) + j];
      }
    }
  }
}

void FoxStepParallel(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c, int bs, int q,
                     int step, int row_start, int row_end) {
  int bsq = bs * bs;
  for (int i = row_start; i < row_end; i++) {
    int k = (i + step) % q;
    for (int j = 0; j < q; j++) {
      MultiplyBlocks(a, ((i * q) + k) * bsq, b, ((k * q) + j) * bsq, c, ((i * q) + j) * bsq, bs);
    }
  }
}

int ChooseBlockSize(int n) {
  for (int div = static_cast<int>(std::sqrt(static_cast<double>(n))); div >= 1; div--) {
    if (n % div == 0) {
      return div;
    }
  }
  return 1;
}

}  // namespace

SokolovKMatrixDoubleFoxSTL::SokolovKMatrixDoubleFoxSTL(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool SokolovKMatrixDoubleFoxSTL::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool SokolovKMatrixDoubleFoxSTL::PreProcessingImpl() {
  GetOutput() = 0;
  n_ = GetInput();
  block_size_ = ChooseBlockSize(n_);
  q_ = n_ / block_size_;
  auto sz = static_cast<std::size_t>(n_) * n_;
  std::vector<double> a(sz, 1.5);
  std::vector<double> b(sz, 2.0);
  blocks_a_.resize(sz);
  blocks_b_.resize(sz);
  blocks_c_.assign(sz, 0.0);
  DecomposeToBlocks(a, blocks_a_, n_, block_size_, q_);
  DecomposeToBlocks(b, blocks_b_, n_, block_size_, q_);
  return true;
}

bool SokolovKMatrixDoubleFoxSTL::RunImpl() {
  std::ranges::fill(blocks_c_, 0.0);

  int num_threads = std::min(ppc::util::GetNumThreads(), q_);
  if (num_threads <= 1) {
    for (int step = 0; step < q_; step++) {
      FoxStepParallel(blocks_a_, blocks_b_, blocks_c_, block_size_, q_, step, 0, q_);
    }
  } else {
    std::vector<std::thread> threads(num_threads);
    for (int step = 0; step < q_; step++) {
      int rows_per_thread = q_ / num_threads;
      int extra = q_ % num_threads;
      int current_row = 0;
      for (int t = 0; t < num_threads; t++) {
        int row_start = current_row;
        int row_end = row_start + rows_per_thread + (t < extra ? 1 : 0);
        current_row = row_end;
        threads[t] = std::thread(FoxStepParallel, std::cref(blocks_a_), std::cref(blocks_b_), std::ref(blocks_c_),
                                 block_size_, q_, step, row_start, row_end);
      }
      for (int t = 0; t < num_threads; t++) {
        threads[t].join();
      }
    }
  }

  return true;
}

bool SokolovKMatrixDoubleFoxSTL::PostProcessingImpl() {
  std::vector<double> result(static_cast<std::size_t>(n_) * n_);
  AssembleFromBlocks(blocks_c_, result, n_, block_size_, q_);
  double expected = 3.0 * n_;
  bool ok = std::ranges::all_of(result, [expected](double v) { return std::abs(v - expected) <= 1e-9; });
  GetOutput() = ok ? GetInput() : -1;
  std::vector<double>().swap(blocks_a_);
  std::vector<double>().swap(blocks_b_);
  std::vector<double>().swap(blocks_c_);
  return true;
}

}  // namespace sokolov_k_matrix_double_fox
