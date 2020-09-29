/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include "../MmulHelper.h"
#include <array/NDArrayFactory.h>
#include <helpers/BlasHelper.h>
#include <helpers/ShapeUtils.h>
#include <exceptions/datatype_exception.h>
#include <execution/Threads.h>
#include <helpers/LoopsCoordsHelper.h>

namespace sd {

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN              -> actual sequence of axes doesn't matter
template <typename T1, typename T2, typename T3>
static  void usualGemm(const NDArray* vA, const NDArray* vB, NDArray* vC,
                                 const int aMaxis, const int aKaxis, const int bKaxis, const int bNaxis, const int cMaxis, const int cNaxis,
                                 const double alpha, const double beta) {

    const T1* A = vA->bufferAsT<T1>();
    const T2* B = vB->bufferAsT<T2>();
          T3* C = vC->bufferAsT<T3>();

    const T3 alphaZ = alpha;
    const T3 betaZ  = beta;

    const bool betaPersent = beta;

    const Nd4jLong* aShapeInfo = vA->shapeInfo();
    const Nd4jLong* bShapeInfo = vB->shapeInfo();
    const Nd4jLong* cShapeInfo = vC->shapeInfo();

    const int aRank = vA->rankOf();
    const int bRank = vB->rankOf();
    const int cRank = vC->rankOf();

    const Nd4jLong cLen = vC->lengthOf();

    const int K = vA->sizeAt(aKaxis);

    auto func = PRAGMA_THREADS_FOR {

        std::vector<Nd4jLong> aCoords(2), bCoords(2), cCoords(2);

        for (auto i = start; i < stop; ++i) {

            // evaluate C coordinates
            shape::index2coordsCPU(start, i, cShapeInfo, cCoords.data());

            // evaluate A coordinates
            aCoords[aMaxis] = cCoords[cMaxis];
            aCoords[aKaxis] = 0;

            // evaluate B coordinates
            bCoords[bKaxis] = 0;
            bCoords[bNaxis] = cCoords[cNaxis];

            auto aOffset = shape::getOffset(aShapeInfo, aCoords.data());
            auto bOffset = shape::getOffset(bShapeInfo, bCoords.data());

            T3 val = A[aOffset] * B[bOffset];                       // first iteration

            for (int j = 1; j < K; ++j) {                          // rest iterations
                aOffset += shape::stride(aShapeInfo)[aKaxis];
                bOffset += shape::stride(bShapeInfo)[bKaxis];
                val = val + A[aOffset] * B[bOffset];
            }

            auto cOffset = shape::getOffset(cShapeInfo, cCoords.data());

            if(betaPersent)
                C[cOffset] = alphaZ * val + betaZ * C[cOffset];
            else
                C[cOffset] = alphaZ * val;
        }
    };

    samediff::Threads::parallel_tad(func, 0, cLen);
}


//////////////////////////////////////////////////////////////////////////////
// MXN x N = M  -> actual sequence of {M,N} axes doesn't matter
template <typename T1, typename T2, typename T3>
static  void usualGemv(const NDArray* vA, const NDArray* vX, NDArray* vY, const int incx, const int incy, const int aMaxis, const double alpha, const double beta) {

    const T1* A = vA->bufferAsT<T1>();
    const T2* X = vX->bufferAsT<T2>();
          T3* Y = vY->bufferAsT<T3>();

    const T3 alphaZ = alpha;
    const T3 betaZ  = beta;

    const bool betaPersent = beta;

    const Nd4jLong* aShapeInfo = vA->shapeInfo();
    const Nd4jLong* xShapeInfo = vX->shapeInfo();
    const Nd4jLong* yShapeInfo = vY->shapeInfo();

    const int N = vX->lengthOf();
    const int M = vY->lengthOf();

    const auto aMstride = vA->strideAt(aMaxis);
    const auto aNstride = vA->strideAt(aMaxis == 0 ? 1 : 0);

    auto func = PRAGMA_THREADS_FOR {

        for (auto i = start; i < stop; ++i) {

            // evaluate offsets
            auto aOffset = i * aMstride;
            auto xOffset = 0;

            T3 val = A[aOffset] * X[xOffset];                       // first iteration

            for (int j = 1; j < N; ++j) {                          // rest iterations
                aOffset += aNstride;
                xOffset += incx;
                val = val + A[aOffset] * X[xOffset];
            }

            auto yOffset = i * incy;

            if(betaPersent)
                Y[yOffset] = alphaZ * val + betaZ * Y[yOffset];
            else
                Y[yOffset] = alphaZ * val;
        }
    };

    samediff::Threads::parallel_tad(func, 0, M);
}

//////////////////////////////////////////////////////////////////////////////
// (X*Y) = Z[0]
template <typename T1, typename T2, typename T3>
static void usualDot(const Nd4jLong length, const double alpha, const void* vX, const Nd4jLong incx, const void* vY, const Nd4jLong incy, const double beta, void* vZ) {

    T1* X = reinterpret_cast<T1*>(const_cast<void*>(vX));
    T2* Y = reinterpret_cast<T2*>(const_cast<void*>(vY));
    T3* Z = reinterpret_cast<T3*>(vZ);
    T3 alphaZ(alpha), betaZ(beta);

    const bool betaPersent = beta;

    T3 sum = 0;
    PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(length > Environment::getInstance()->elementwiseThreshold()) schedule(guided) reduction(OMP_SUMT:sum))
    for(Nd4jLong i = 0; i < length; ++i)
            sum += X[i * incx] * Y[i * incy];

    if(betaPersent)
        *Z = alphaZ * sum + betaZ * *Z;
    else
        *Z = alphaZ * sum;
}

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
NDArray* MmulHelper::mmulMxM(const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder) {
    if (A->dataType() != B->dataType())
        throw datatype_exception::build("mmulMxM expects all data types to be the same", A->dataType(), B->dataType());

    if (C != nullptr && A->dataType() != C->dataType())
        throw datatype_exception::build("mmulMxM expects all data types to be the same", A->dataType(), C->dataType());

    if(A->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of A array is not equal 2 !");
    if(B->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of B array is not equal 2 !");

    const auto M = A->sizeAt(0);
    const auto K = A->sizeAt(1);
    const auto N = B->sizeAt(1);

    if(C != nullptr && C->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxM: rank of C array is not equal 2 !");
    if(B->sizeAt(0) != K)
        throw std::runtime_error("MmulHelper::mmulMxM: B array has wrong number of rows !");
    if(C != nullptr && C->sizeAt(0) != M)
        throw std::runtime_error("MmulHelper::mmulMxM: C array has wrong number of rows !");
    if(C != nullptr && C->sizeAt(1) != N)
        throw std::runtime_error("MmulHelper::mmulMxM: C array has wrong number of columns !");

    if(C == nullptr)
        C = new NDArray(outOrder, {M,N}, DataTypeUtils::pickPairwiseResultType(A->dataType(), B->dataType()), A->getContext());

    if (C->isEmpty())
        return C;

    const auto aType = A->dataType();
    const auto bType = B->dataType();
    const auto cType = C->dataType();

    const bool AB(aType == bType), AC(aType == cType), ABC(AB && AC);
    const bool hasGemm = BlasHelper::getInstance()->hasGEMM(aType);

    const bool typeDouble = hasGemm && ABC &&  aType == DataType::DOUBLE;
    const bool typeFloat  = hasGemm && ABC &&  aType == DataType::FLOAT32;

    if(!typeFloat && !typeDouble) {
        BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemm, (A, B, C, 0, 1, 0, 1, 0, 1, alpha, beta), NUMERIC_TYPES);
        // BUILD_TRIPLE_SELECTOR(aType, bType, cType, usualGemm, (A, B, C, 0, 1, 0, 1, 0, 1, alpha, beta), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    }
    else {

        std::vector<NDArray*> toDelete;

        NDArray *pA(const_cast<NDArray*>(A)), *pB(const_cast<NDArray*>(B)), *pC(const_cast<NDArray*>(C));

        bool aMcont = M == 1 || A->strideAt(0) == 1;
        bool aKcont = K == 1 || A->strideAt(1) == 1;
        bool bKcont = K == 1 || B->strideAt(0) == 1;
        bool bNcont = N == 1 || B->strideAt(1) == 1;
        bool cMcont = M == 1 || C->strideAt(0) == 1;
        bool cNcont = N == 1 || C->strideAt(1) == 1;

        if(!aMcont && !aKcont) {
            pA = new NDArray(A->dup('f'));
            toDelete.push_back(pA);
            aMcont = true;
        }
        if(!bKcont && !bNcont) {
            pB = new NDArray(B->dup('f'));
            toDelete.push_back(pB);
            bKcont = true;
        }
        if(!cMcont && !cNcont) {
            pC = new NDArray(C->dup('f'));
            toDelete.push_back(pC);
            cMcont = true;
        }

        const CBLAS_ORDER blasOrder = cMcont ? CblasColMajor : CblasRowMajor;

        const bool transA = (!aMcont && cMcont) || (aMcont && !cMcont);
        const bool transB = (!bKcont && cMcont) || (bKcont && !cMcont);

        const CBLAS_TRANSPOSE transAblas = transA ? CblasTrans : CblasNoTrans;
        const CBLAS_TRANSPOSE transBblas = transB ? CblasTrans : CblasNoTrans;

        const int lda = (aMcont && aKcont) ? M : !aMcont ? pA->strideAt(0) : pA->strideAt(1);
        const int ldb = (bKcont && bNcont) ? K : !bKcont ? pB->strideAt(0) : pB->strideAt(1);
        const int ldc = (cMcont && cNcont) ? M : !cMcont ? pC->strideAt(0) : pC->strideAt(1);

        if(typeFloat) {
            BlasHelper::getInstance()->sgemm()(blasOrder, transAblas, transBblas, M, N, K, (float) alpha, pA->bufferAsT<float>(), lda, pB->bufferAsT<float>(), ldb, (float) beta, pC->bufferAsT<float>(), ldc);
        }
        else if(typeDouble) {
            BlasHelper::getInstance()->dgemm()(blasOrder, transAblas, transBblas, M, N, K, (double) alpha, pA->bufferAsT<double>(), lda, pB->bufferAsT<double>(), ldb, (double) beta, pC->bufferAsT<double>(), ldc);
        }

        if(pC != C) {
            C->assign(pC);
            delete pC;
        }
        if(pA != A)
            delete pA;
        if(pB != B)
            delete pB;
    }

    return C;
}

////////////////////////////////////////////////////////////////////////////
// MXN x N = M
NDArray* MmulHelper::mmulMxV(const NDArray* A, const NDArray* X, sd::NDArray* Y, const double alpha, const double beta, const char outOrder) {

    if (X->dataType() != A->dataType())
        throw datatype_exception::build("mmulMxV expects all data types to be the same", A->dataType(), X->dataType());

    if (Y != nullptr && X->dataType() != Y->dataType())
        throw datatype_exception::build("mmulMxV expects all data types to be the same", A->dataType(), Y->dataType());

    int xLenDim, yLenDim(0);

    if(A->rankOf() != 2)
        throw std::runtime_error("MmulHelper::mmulMxV: rank of A array is not equal 2 !");
    if(!shape::isCommonVector(X->shapeInfo(), xLenDim))
        throw std::runtime_error("MmulHelper::mmulMxV: X array must be vector !");

    const auto M = A->sizeAt(0);
    const auto N = A->sizeAt(1);

    if(Y != nullptr && !shape::isCommonVector(Y->shapeInfo(), yLenDim))
        throw std::runtime_error("MmulHelper::mmulMxV: Y array must be vector !");
    if(X->lengthOf() != N)
        throw std::runtime_error("MmulHelper::mmulMxV: X vector has wrong length !");
    if(Y != nullptr && Y->lengthOf() != M)
        throw std::runtime_error("MmulHelper::mmulMxV: Y array has wrong length !");

    if(Y == nullptr)
        Y = new NDArray(outOrder, {M}, DataTypeUtils::pickPairwiseResultType(A->dataType(), X->dataType()), A->getContext());

    if (Y->isEmpty())
        return Y;

    const int incx = X->stridesOf()[xLenDim];
    const int incy = Y->stridesOf()[yLenDim];

    const auto aType = A->dataType();
    const auto xType = X->dataType();
    const auto yType = Y->dataType();

    const bool AX(aType == xType), AY(aType == yType), AXY(AX && AY);
    const bool hasGemv = BlasHelper::getInstance()->hasGEMV(aType);

    const bool typeDouble = hasGemv && AXY && aType == DataType::DOUBLE;
    const bool typeFloat  = hasGemv && AXY && aType == DataType::FLOAT32;

    if(!typeDouble && !typeFloat) {
        BUILD_SINGLE_SELECTOR_THRICE(aType, usualGemv, (A, X, Y, incx, incy, 0, alpha, beta), NUMERIC_TYPES);
        // BUILD_TRIPLE_SELECTOR(aType, xType, yType, usualGemv, (A, X, Y, incx, incy, 0, alpha, beta), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
    }
    else {

        NDArray *pA(const_cast<NDArray*>(A));

        bool aMcont = M == 1 || A->strideAt(0) == 1;
        bool aNcont = N == 1 || A->strideAt(1) == 1;

        if(!aMcont && !aNcont) {
            pA = new NDArray(A->dup('f'));
            aMcont = true;
        }
        const CBLAS_ORDER blasOrder = aMcont ? CblasColMajor : CblasRowMajor;

        const int lda = (aMcont && aNcont) ? M : !aMcont ? pA->strideAt(0) : pA->strideAt(1);

        // choose appropriate cuda gemm api depending on data types
        if(typeDouble) {
            BlasHelper::getInstance()->dgemv()(blasOrder, CblasNoTrans, M, N, alpha, (double*)pA->buffer(), lda, (double*)X->buffer(), incx, beta, (double*)Y->buffer(), incy);
        }
        else if(typeFloat) {
            BlasHelper::getInstance()->sgemv()(blasOrder, CblasNoTrans, M, N, (float)alpha, (float*)pA->buffer(), lda, (float*)X->buffer(), incx, (float)beta, (float*)Y->buffer(), incy);
        }

        if(pA != A)
            delete pA;
    }

    return Y;
}

////////////////////////////////////////////////////////////////////////////
// (X * Y) = Z[0]
NDArray* MmulHelper::dot(const NDArray* X, const NDArray* Y, sd::NDArray* Z, const double alpha, const double beta) {
    if (X->dataType() != Y->dataType())
        throw datatype_exception::build("Dot expects all data types to be the same", X->dataType(), Y->dataType());

    if (Z != nullptr && X->dataType() != Z->dataType())
        throw datatype_exception::build("Dot expects all data types to be the same", X->dataType(), Z->dataType());

    int xLenDim(0), yLenDim(0);

    if(!shape::isCommonVector(X->shapeInfo(), xLenDim))
        throw std::runtime_error("MmulHelper::dot cuda: X array must be vector !");
    if(!shape::isCommonVector(Y->shapeInfo(), yLenDim))
        throw std::runtime_error("MmulHelper::dot cuda: Y array must be vector !");
    if (Z != nullptr && Z->lengthOf() != 1)
        throw std::runtime_error("MmulHelper::dot cuda: Z array must have length equal to unity !");

    const auto length = X->lengthOf();

    if(Y->lengthOf() != length)
        throw std::runtime_error("MmulHelper::dot: lengths of input vectors are different !");

    if(Z == nullptr)
        Z = new NDArray(DataTypeUtils::pickPairwiseResultType(X->dataType(), Y->dataType()), X->getContext());

    const Nd4jLong incx = X->stridesOf()[xLenDim];
    const Nd4jLong incy = Y->stridesOf()[yLenDim];

    const auto xType = X->dataType();
    const auto yType = Y->dataType();
    const auto zType = Z->dataType();

    BUILD_SINGLE_SELECTOR_THRICE(xType, usualDot, (length, alpha, X->buffer(), incx, Y->buffer(), incy, beta, Z->buffer()), NUMERIC_TYPES);
        //BUILD_TRIPLE_SELECTOR(xType, yType, zType, usualDot, (length, alpha, X->buffer(), incx, Y->buffer(), incy, beta, Z->buffer()), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);

    return Z;
}

//////////////////////////////////////////////////////////////////////////////
// [bS,M,K] x [bS,K,N] = [bS,M,N]
// [bS,M,K] x    [K,N] = [bS,M,N]
//    [M,K] x [bS,K,N] = [bS,M,N]
// bS could stand for several axes

template <typename T1, typename T2, typename T3>
static void innerGemmWoChecks(const Nd4jLong M, const Nd4jLong N, const Nd4jLong K, const T1 alpha, const T1* __restrict A, const Nd4jLong aStride_M, const Nd4jLong aStride_K, const T2* __restrict B, Nd4jLong bStride_K, Nd4jLong bStride_N, T3* __restrict C, const Nd4jLong cStride_M) {
#if 0
    nd4j_printf(" M %ld   , N %ld  , K %ld , alpha %lf, aStride_M %ld  , aStride_K %ld,    bStride_K %ld  bStride_N %ld,   cStride_M %ld   \n",
        M, N, K, alpha, aStride_M, aStride_K, bStride_K, bStride_N, cStride_M);
#endif

    // Nd4jLong aStride_Mx2 = aStride_M *2;

    if (aStride_K == 1 && bStride_N == 1) {
        //cStride_N ==1
        Nd4jLong M_L = M & -8;;
        Nd4jLong K_L = K & -8;// 8;

        for (Nd4jLong k = 0; k < K_L; k += 8) {

            const T2* __restrict BB0 = &(B[k * bStride_K]);
            const T2* __restrict BB1 = &(B[k * bStride_K + bStride_K]);
            const T2* __restrict BB2 = &(B[k * bStride_K + 2 * bStride_K]);
            const T2* __restrict BB3 = &(B[k * bStride_K + 3 * bStride_K]);
            const T2* __restrict BB4 = &(B[k * bStride_K + 4 * bStride_K]);
            const T2* __restrict BB5 = &(B[k * bStride_K + 5 * bStride_K]);
            const T2* __restrict BB6 = &(B[k * bStride_K + 6 * bStride_K]);
            const T2* __restrict BB7 = &(B[k * bStride_K + 7 * bStride_K]);
            const T1* __restrict AA = &(A[k * 1]);
            for (Nd4jLong m = 0; m < M_L; m += 8) {
                Nd4jLong OFF_A = m * aStride_M;
                T1 AA0 = AA[OFF_A];
                T1 AA1 = AA[OFF_A + aStride_M];
                T1 AA2 = AA[OFF_A + 2 * aStride_M];
                T1 AA3 = AA[OFF_A + 3 * aStride_M];
                T1 AA4 = AA[OFF_A + 4 * aStride_M];
                T1 AA5 = AA[OFF_A + 5 * aStride_M];
                T1 AA6 = AA[OFF_A + 6 * aStride_M];
                T1 AA7 = AA[OFF_A + 7 * aStride_M];

                T1 AA0_1 = AA[OFF_A + 1];
                T1 AA1_1 = AA[OFF_A + aStride_M + 1];
                T1 AA2_1 = AA[OFF_A + 2 * aStride_M + 1];
                T1 AA3_1 = AA[OFF_A + 3 * aStride_M + 1];
                T1 AA4_1 = AA[OFF_A + 4 * aStride_M + 1];
                T1 AA5_1 = AA[OFF_A + 5 * aStride_M + 1];
                T1 AA6_1 = AA[OFF_A + 6 * aStride_M + 1];
                T1 AA7_1 = AA[OFF_A + 7 * aStride_M + 1];

                T1 AA0_2 = AA[OFF_A + 2];
                T1 AA1_2 = AA[OFF_A + aStride_M + 2];
                T1 AA2_2 = AA[OFF_A + 2 * aStride_M + 2];
                T1 AA3_2 = AA[OFF_A + 3 * aStride_M + 2];
                T1 AA4_2 = AA[OFF_A + 4 * aStride_M + 2];
                T1 AA5_2 = AA[OFF_A + 5 * aStride_M + 2];
                T1 AA6_2 = AA[OFF_A + 6 * aStride_M + 2];
                T1 AA7_2 = AA[OFF_A + 7 * aStride_M + 2];

                T1 AA0_3 = AA[OFF_A + 3];
                T1 AA1_3 = AA[OFF_A + aStride_M + 3];
                T1 AA2_3 = AA[OFF_A + 2 * aStride_M + 3];
                T1 AA3_3 = AA[OFF_A + 3 * aStride_M + 3];
                T1 AA4_3 = AA[OFF_A + 4 * aStride_M + 3];
                T1 AA5_3 = AA[OFF_A + 5 * aStride_M + 3];
                T1 AA6_3 = AA[OFF_A + 6 * aStride_M + 3];
                T1 AA7_3 = AA[OFF_A + 7 * aStride_M + 3];

                T1 AA0_4 = AA[OFF_A + 4];
                T1 AA1_4 = AA[OFF_A + aStride_M + 4];
                T1 AA2_4 = AA[OFF_A + 2 * aStride_M + 4];
                T1 AA3_4 = AA[OFF_A + 3 * aStride_M + 4];
                T1 AA4_4 = AA[OFF_A + 4 * aStride_M + 4];
                T1 AA5_4 = AA[OFF_A + 5 * aStride_M + 4];
                T1 AA6_4 = AA[OFF_A + 6 * aStride_M + 4];
                T1 AA7_4 = AA[OFF_A + 7 * aStride_M + 4];


                T1 AA0_5 = AA[OFF_A + 5];
                T1 AA1_5 = AA[OFF_A + aStride_M + 5];
                T1 AA2_5 = AA[OFF_A + 2 * aStride_M + 5];
                T1 AA3_5 = AA[OFF_A + 3 * aStride_M + 5];
                T1 AA4_5 = AA[OFF_A + 4 * aStride_M + 5];
                T1 AA5_5 = AA[OFF_A + 5 * aStride_M + 5];
                T1 AA6_5 = AA[OFF_A + 6 * aStride_M + 5];
                T1 AA7_5 = AA[OFF_A + 7 * aStride_M + 5];

                T1 AA0_6 = AA[OFF_A + 6];
                T1 AA1_6 = AA[OFF_A + aStride_M + 6];
                T1 AA2_6 = AA[OFF_A + 2 * aStride_M + 6];
                T1 AA3_6 = AA[OFF_A + 3 * aStride_M + 6];
                T1 AA4_6 = AA[OFF_A + 4 * aStride_M + 6];
                T1 AA5_6 = AA[OFF_A + 5 * aStride_M + 6];
                T1 AA6_6 = AA[OFF_A + 6 * aStride_M + 6];
                T1 AA7_6 = AA[OFF_A + 7 * aStride_M + 6];


                T1 AA0_7 = AA[OFF_A + 7];
                T1 AA1_7 = AA[OFF_A + aStride_M + 7];
                T1 AA2_7 = AA[OFF_A + 2 * aStride_M + 7];
                T1 AA3_7 = AA[OFF_A + 3 * aStride_M + 7];
                T1 AA4_7 = AA[OFF_A + 4 * aStride_M + 7];
                T1 AA5_7 = AA[OFF_A + 5 * aStride_M + 7];
                T1 AA6_7 = AA[OFF_A + 6 * aStride_M + 7];
                T1 AA7_7 = AA[OFF_A + 7 * aStride_M + 7];

                T3* __restrict CC0 = &(C[m * cStride_M]);
                T3* __restrict CC1 = &(C[m * cStride_M + cStride_M]);
                T3* __restrict CC2 = &(C[m * cStride_M + 2 * cStride_M]);
                T3* __restrict CC3 = &(C[m * cStride_M + 3 * cStride_M]);

                T3* __restrict CC4 = &(C[m * cStride_M + 4 * cStride_M]);
                T3* __restrict CC5 = &(C[m * cStride_M + 5 * cStride_M]);
                T3* __restrict CC6 = &(C[m * cStride_M + 6 * cStride_M]);
                T3* __restrict CC7 = &(C[m * cStride_M + 7 * cStride_M]);

                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //   nd4j_printf("%p %lf %lf*%lf + %lf*%lf +%lf*%lf +%lf*%lf +%lf*%lf +%lf*%lf +%lf*%lf +  \n", &(CC0[n]),CC0[n],  AA0 , BB0[n], AA0_1, BB1[n], AA0_2 , BB2[n] ,AA0_3, BB3[n] , AA0_4 , BB4[n] , AA0_5, BB5[n] , AA0_6, BB6[n] , AA0_7, BB7[n]);
                        CC0[n] += alpha * (AA0 * BB0[n] + AA0_1 * BB1[n] + AA0_2 * BB2[n] + AA0_3 * BB3[n] + AA0_4 * BB4[n] + AA0_5 * BB5[n] + AA0_6 * BB6[n] + AA0_7 * BB7[n]);
                        CC1[n] += alpha * (AA1 * BB0[n] + AA1_1 * BB1[n] + AA1_2 * BB2[n] + AA1_3 * BB3[n] + AA1_4 * BB4[n] + AA1_5 * BB5[n] + AA1_6 * BB6[n] + AA1_7 * BB7[n]);
                        CC2[n] += alpha * (AA2 * BB0[n] + AA2_1 * BB1[n] + AA2_2 * BB2[n] + AA2_3 * BB3[n] + AA2_4 * BB4[n] + AA2_5 * BB5[n] + AA2_6 * BB6[n] + AA2_7 * BB7[n]);
                        CC3[n] += alpha * (AA3 * BB0[n] + AA3_1 * BB1[n] + AA3_2 * BB2[n] + AA3_3 * BB3[n] + AA3_4 * BB4[n] + AA3_5 * BB5[n] + AA3_6 * BB6[n] + AA3_7 * BB7[n]);

                        CC4[n] += alpha * (AA4 * BB0[n] + AA4_1 * BB1[n] + AA4_2 * BB2[n] + AA4_3 * BB3[n] + AA4_4 * BB4[n] + AA4_5 * BB5[n] + AA4_6 * BB6[n] + AA4_7 * BB7[n]);
                        CC5[n] += alpha * (AA5 * BB0[n] + AA5_1 * BB1[n] + AA5_2 * BB2[n] + AA5_3 * BB3[n] + AA5_4 * BB4[n] + AA5_5 * BB5[n] + AA5_6 * BB6[n] + AA5_7 * BB7[n]);
                        CC6[n] += alpha * (AA6 * BB0[n] + AA6_1 * BB1[n] + AA6_2 * BB2[n] + AA6_3 * BB3[n] + AA6_4 * BB4[n] + AA6_5 * BB5[n] + AA6_6 * BB6[n] + AA6_7 * BB7[n]);
                        CC7[n] += alpha * (AA7 * BB0[n] + AA7_1 * BB1[n] + AA7_2 * BB2[n] + AA7_3 * BB3[n] + AA7_4 * BB4[n] + AA7_5 * BB5[n] + AA7_6 * BB6[n] + AA7_7 * BB7[n]);
                    }//N
            }//M


            for (Nd4jLong m = M_L; m < M; m++) {

                Nd4jLong OFF_A = m * aStride_M;
                T1 AA0 = AA[OFF_A];
                T1 AA0_1 = AA[OFF_A + 1];
                T1 AA0_2 = AA[OFF_A + 2];
                T1 AA0_3 = AA[OFF_A + 3];
                T1 AA0_4 = AA[OFF_A + 4];
                T1 AA0_5 = AA[OFF_A + 5];
                T1 AA0_6 = AA[OFF_A + 6];
                T1 AA0_7 = AA[OFF_A + 7];
                T3* __restrict CC0 = &(C[m * cStride_M]);


                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        CC0[n] += alpha * (AA0 * BB0[n] + AA0_1 * BB1[n] + AA0_2 * BB2[n] + AA0_3 * BB3[n] + AA0_4 * BB4[n] + AA0_5 * BB5[n] + AA0_6 * BB6[n] + AA0_7 * BB7[n]);
                    }//N
            }//M
        }//K
        for (Nd4jLong k = K_L; k < K; k++) {
            const T2* __restrict BB0 = &(B[k * bStride_K]);
            const T1* __restrict AA = &(A[k * 1]);

            for (Nd4jLong m = 0; m < M_L; m += 8) {
                Nd4jLong OFF_A = m * aStride_M;
                T1 AA0 = alpha * AA[OFF_A];
                T1 AA1 = alpha * AA[OFF_A + aStride_M];
                T1 AA2 = alpha * AA[OFF_A + 2 * aStride_M];
                T1 AA3 = alpha * AA[OFF_A + 3 * aStride_M];
                T1 AA4 = alpha * AA[OFF_A + 4 * aStride_M];
                T1 AA5 = alpha * AA[OFF_A + 5 * aStride_M];
                T1 AA6 = alpha * AA[OFF_A + 6 * aStride_M];
                T1 AA7 = alpha * AA[OFF_A + 7 * aStride_M];

                T3* __restrict CC0 = &(C[m * cStride_M]);
                T3* __restrict CC1 = &(C[m * cStride_M + cStride_M]);
                T3* __restrict CC2 = &(C[m * cStride_M + 2 * cStride_M]);
                T3* __restrict CC3 = &(C[m * cStride_M + 3 * cStride_M]);

                T3* __restrict CC4 = &(C[m * cStride_M + 4 * cStride_M]);
                T3* __restrict CC5 = &(C[m * cStride_M + 5 * cStride_M]);
                T3* __restrict CC6 = &(C[m * cStride_M + 6 * cStride_M]);
                T3* __restrict CC7 = &(C[m * cStride_M + 7 * cStride_M]);

                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //  nd4j_printf("%p %lf %lf %lf \n", &(CC[n]),CC[n], AA0, BB[n]);
                        CC0[n] += AA0 * BB0[n];
                        CC1[n] += AA1 * BB0[n];
                        CC2[n] += AA2 * BB0[n];
                        CC3[n] += AA3 * BB0[n];
                        CC4[n] += AA4 * BB0[n];
                        CC5[n] += AA5 * BB0[n];
                        CC6[n] += AA6 * BB0[n];
                        CC7[n] += AA7 * BB0[n];
                    }//N
            }//M

            for (Nd4jLong m = M_L; m < M; m++) {

                Nd4jLong OFF_A = m * aStride_M;
                T1 AA0 = alpha * AA[OFF_A];
                T3* __restrict CC = &(C[m * cStride_M]);


                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        CC[n] += AA0 * BB0[n];
                    }//N
            }//M
        }//K
    }
    else {
   // printf("strided\n");
    Nd4jLong M_L = M & -4;;
    Nd4jLong K_L =  K & -4;

        for (Nd4jLong k = 0; k < K_L; k += 4) {

            const T2* __restrict BB0 = &(B[k * bStride_K]);
            const T2* __restrict BB1 = &(B[k * bStride_K + bStride_K]);
            const T2* __restrict BB2 = &(B[k * bStride_K + 2 * bStride_K]);
            const T2* __restrict BB3 = &(B[k * bStride_K + 3 * bStride_K]);

            const T1* __restrict AA = &(A[k * aStride_K]);
            for (Nd4jLong m = 0; m < M_L; m += 4) {

                T1 AA0 = AA[m * aStride_M];
                T1 AA0_1 = AA[m * aStride_M + aStride_K];
                T1 AA0_2 = AA[m * aStride_M + 2 * aStride_K];
                T1 AA0_3 = AA[m * aStride_M + 3 * aStride_K];

                T1 AA1 = AA[m * aStride_M + aStride_M];
                T1 AA1_1 = AA[m * aStride_M + aStride_M + aStride_K];
                T1 AA1_2 = AA[m * aStride_M + aStride_M + 2 * aStride_K];
                T1 AA1_3 = AA[m * aStride_M + aStride_M + 3 * aStride_K];

                T1 AA2 = AA[m * aStride_M + 2 * aStride_M];
                T1 AA2_1 = AA[m * aStride_M + 2 * aStride_M + aStride_K];
                T1 AA2_2 = AA[m * aStride_M + 2 * aStride_M + 2 * aStride_K];
                T1 AA2_3 = AA[m * aStride_M + 2 * aStride_M + 3 * aStride_K];

                T1 AA3 = AA[m * aStride_M + 3 * aStride_M];
                T1 AA3_1 = AA[m * aStride_M + 3 * aStride_M + aStride_K];
                T1 AA3_2 = AA[m * aStride_M + 3 * aStride_M + 2 * aStride_K];
                T1 AA3_3 = AA[m * aStride_M + 3 * aStride_M + 3 * aStride_K];

                T3* __restrict CC0 = &(C[m * cStride_M]);
                T3* __restrict CC1 = &(C[m * cStride_M + cStride_M]);
                T3* __restrict CC2 = &(C[m * cStride_M + 2 * cStride_M]);
                T3* __restrict CC3 = &(C[m * cStride_M + 3 * cStride_M]);

                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //  nd4j_printf("%p %lf %lf %lf \n", &(CC[n]),CC[n], AA0, BB[n * bStride_N]);
                        CC0[n] += alpha * (AA0 * BB0[n * bStride_N] + AA0_1 * BB1[n * bStride_N] + AA0_2 * BB2[n * bStride_N] + AA0_3 * BB3[n * bStride_N]);
                        CC1[n] += alpha * (AA1 * BB0[n * bStride_N] + AA1_1 * BB1[n * bStride_N] + AA1_2 * BB2[n * bStride_N] + AA1_3 * BB3[n * bStride_N]);
                        CC2[n] += alpha * (AA2 * BB0[n * bStride_N] + AA2_1 * BB1[n * bStride_N] + AA2_2 * BB2[n * bStride_N] + AA2_3 * BB3[n * bStride_N]);
                        CC3[n] += alpha * (AA3 * BB0[n * bStride_N] + AA3_1 * BB1[n * bStride_N] + AA3_2 * BB2[n * bStride_N] + AA3_3 * BB3[n * bStride_N]);
                    }//N
            }//M

            for (Nd4jLong m = M_L; m < M; m++) {

                T1 AA0 = AA[m * aStride_M];
                T1 AA0_1 = AA[m * aStride_M + aStride_K];
                T1 AA0_2 = AA[m * aStride_M + 2 * aStride_K];
                T1 AA0_3 = AA[m * aStride_M + 3 * aStride_K];
                T3* __restrict CC = &(C[m * cStride_M]);


                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //  nd4j_printf("%p %lf %lf %lf \n", &(CC[n]),CC[n], AA0, BB[n * bStride_N]);
                        CC[n] += alpha * (AA0 * BB0[n * bStride_N] + AA0_1 * BB1[n * bStride_N] + AA0_2 * BB2[n * bStride_N] + AA0_3 * BB3[n * bStride_N]);
                    }//N
            }//M
        }//K
        for (Nd4jLong k = K_L; k < K; k++) {
            const T2* __restrict BB = &(B[k * bStride_K]);
            const T1* __restrict AA = &(A[k * aStride_K]);

            for (Nd4jLong m = 0; m < M_L; m += 4) {

                T1 AA0 = alpha * AA[m * aStride_M];
                T1 AA1 = alpha * AA[m * aStride_M + aStride_M];
                T1 AA2 = alpha * AA[m * aStride_M + 2 * aStride_M];
                T1 AA3 = alpha * AA[m * aStride_M + 3 * aStride_M];
                T3* __restrict CC0 = &(C[m * cStride_M]);
                T3* __restrict CC1 = &(C[m * cStride_M + cStride_M]);
                T3* __restrict CC2 = &(C[m * cStride_M + 2 * cStride_M]);
                T3* __restrict CC3 = &(C[m * cStride_M + 3 * cStride_M]);

                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //  nd4j_printf("%p %lf %lf %lf \n", &(CC[n]),CC[n], AA0, BB[n * bStride_N]);
                        CC0[n] += AA0 * BB[n * bStride_N];
                        CC1[n] += AA1 * BB[n * bStride_N];
                        CC2[n] += AA2 * BB[n * bStride_N];
                        CC3[n] += AA3 * BB[n * bStride_N];
                    }//N
            }//M

            for (Nd4jLong m = M_L; m < M; m++) {
                T1 AA0 = alpha * AA[m * aStride_M];
                T3* __restrict CC = &(C[m * cStride_M]);


                PRAGMA_OMP_SIMD
                    for (Nd4jLong n = 0; n < N; n++) {
                        //  nd4j_printf("%p %lf %lf %lf \n", &(CC[n]),CC[n], AA0, BB[n * bStride_N]);
                        CC[n] += AA0 * BB[n * bStride_N];
                    }//N
            }//M
        }//K

    }
}

template<typename T3>
static FORCEINLINE void zero_buffer(T3* C_PTR, int M, int N) {
    int M_8 = M & (-8);

    for (Nd4jLong m = 0; m < M_8; m += 8) {
        T3* C_PTR1 = &(C_PTR[N]);
        T3* C_PTR2 = &(C_PTR[2 * N]);
        T3* C_PTR3 = &(C_PTR[3 * N]);
        T3* C_PTR4 = &(C_PTR[4 * N]);
        T3* C_PTR5 = &(C_PTR[5 * N]);
        T3* C_PTR6 = &(C_PTR[6 * N]);
        T3* C_PTR7 = &(C_PTR[7 * N]);

        for (Nd4jLong n = 0; n < N; n++) {
            C_PTR[n] = 0;
            C_PTR1[n] = 0;
            C_PTR2[n] = 0;
            C_PTR3[n] = 0;
            C_PTR4[n] = 0;
            C_PTR5[n] = 0;
            C_PTR6[n] = 0;
            C_PTR7[n] = 0;
        }//M

        C_PTR += 8 * N;
    }//N

    for (Nd4jLong m = M_8; m < M; m++) {

        for (Nd4jLong n = 0; n < N; n++) {
            C_PTR[n] = 0;
        }//M 
        C_PTR += N;
    }//N

}

template<typename T3>
static FORCEINLINE void scal_buffer(const T3 beta, T3* C_PTR, int M, int N, Nd4jLong stride_m) {
    int M_8 = M & (-8);

    for (Nd4jLong m = 0; m < M_8; m += 8) {
        T3* C_PTR1 = &(C_PTR[stride_m]);
        T3* C_PTR2 = &(C_PTR[2 * stride_m]);
        T3* C_PTR3 = &(C_PTR[3 * stride_m]);
        T3* C_PTR4 = &(C_PTR[4 * stride_m]);
        T3* C_PTR5 = &(C_PTR[5 * stride_m]);
        T3* C_PTR6 = &(C_PTR[6 * stride_m]);
        T3* C_PTR7 = &(C_PTR[7 * stride_m]);

        for (Nd4jLong n = 0; n < N; n++) {
            C_PTR[n] = beta * C_PTR[n];
            C_PTR1[n] = beta * C_PTR1[n];
            C_PTR2[n] = beta * C_PTR2[n];
            C_PTR3[n] = beta * C_PTR3[n];
            C_PTR4[n] = beta * C_PTR4[n];
            C_PTR5[n] = beta * C_PTR5[n];
            C_PTR6[n] = beta * C_PTR6[n];
            C_PTR7[n] = beta * C_PTR7[n];
        }//M

        C_PTR += 8 * stride_m;
    }//N

    for (Nd4jLong m = M_8; m < M; m++) {

        for (Nd4jLong n = 0; n < N; n++) {
            C_PTR[n] = beta * C_PTR[n];
        }//M 
        C_PTR += stride_m;
    }//N

}

template<typename T3>
void copy_buffer(T3* dest, T3* source, T3 betaZ, int M, int N, Nd4jLong dest_stride_m, Nd4jLong dest_stride_n) {
    if (dest_stride_n == 1 && dest_stride_m == N) {
        int M_8 = M & (-8);

        if ((bool)betaZ) {

            for (Nd4jLong m = 0; m < M_8; m += 8) {

                T3* dest1 = &(dest[1 * N]);
                T3* dest2 = &(dest[2 * N]);
                T3* dest3 = &(dest[3 * N]);
                T3* dest4 = &(dest[4 * N]);
                T3* dest5 = &(dest[5 * N]);
                T3* dest6 = &(dest[6 * N]);
                T3* dest7 = &(dest[7 * N]);

                T3* source1 = &(source[1 * N]);
                T3* source2 = &(source[2 * N]);
                T3* source3 = &(source[3 * N]);
                T3* source4 = &(source[4 * N]);
                T3* source5 = &(source[5 * N]);
                T3* source6 = &(source[6 * N]);
                T3* source7 = &(source[7 * N]);

                for (Nd4jLong n = 0; n < N; n++) {
                    dest[n * 1] = betaZ * dest[n * 1] + source[n];
                    dest1[n * 1] = betaZ * dest1[n * 1] + source1[n];
                    dest2[n * 1] = betaZ * dest2[n * 1] + source2[n];
                    dest3[n * 1] = betaZ * dest3[n * 1] + source3[n];
                    dest4[n * 1] = betaZ * dest4[n * 1] + source4[n];
                    dest5[n * 1] = betaZ * dest5[n * 1] + source5[n];
                    dest6[n * 1] = betaZ * dest6[n * 1] + source6[n];
                    dest7[n * 1] = betaZ * dest7[n * 1] + source7[n];
                }//M


                source += 8 * N;
                dest += 8 * N;
            }//N

            for (Nd4jLong m = M_8; m < M; m++) {

                for (Nd4jLong n = 0; n < N; n++) {
                    dest[n * 1] = betaZ * dest[n * 1] + source[n];
                    source[n] = 0;
                }//M 
                source += N;
                dest += N;
            }//N
        }
        else {

            for (Nd4jLong m = 0; m < M_8; m += 8) {

                T3* dest1 = &(dest[N]);
                T3* dest2 = &(dest[2 * N]);
                T3* dest3 = &(dest[3 * N]);
                T3* dest4 = &(dest[4 * N]);
                T3* dest5 = &(dest[5 * N]);
                T3* dest6 = &(dest[6 * N]);
                T3* dest7 = &(dest[7 * N]);

                T3* source1 = &(source[N]);
                T3* source2 = &(source[2 * N]);
                T3* source3 = &(source[3 * N]);
                T3* source4 = &(source[4 * N]);
                T3* source5 = &(source[5 * N]);
                T3* source6 = &(source[6 * N]);
                T3* source7 = &(source[7 * N]);

                for (Nd4jLong n = 0; n < N; n++) {
                    dest[n * 1] = source[n];
                    dest1[n * 1] = source1[n];
                    dest2[n * 1] = source2[n];
                    dest3[n * 1] = source3[n];
                    dest4[n * 1] = source4[n];
                    dest5[n * 1] = source5[n];
                    dest6[n * 1] = source6[n];
                    dest7[n * 1] = source7[n];
                }//M


                source += 8 * N;
                dest += 8 * N;
            }//N

            for (Nd4jLong m = M_8; m < M; m++) {

                for (Nd4jLong n = 0; n < N; n++) {
                    dest[n * 1] = source[n];
                    source[n] = 0;
                }//M 
                source += N;
                dest += N;
            }//N

        }//betaZ

    }
    else if (dest_stride_m < dest_stride_n) {

        if ((bool)betaZ) {
            for (Nd4jLong n = 0; n < N; n++) {
                T3* __restrict source_0 = &(source[n]);
                for (Nd4jLong m = 0; m < M; m++) {
                    dest[m * dest_stride_m] = betaZ * dest[m * dest_stride_m] + (*source_0);
                    source_0 += N;
                }//N
                dest += dest_stride_n;
            }//M

        }
        else {
            for (Nd4jLong n = 0; n < N; n++) {
                T3* __restrict source_0 = &(source[n]);
                for (Nd4jLong m = 0; m < M; m++) {
                    dest[m * dest_stride_m] = (*source_0);
                    source_0 += N;
                }//N
                dest += dest_stride_n;
            }//M  

        }

    }
    else {
        for (Nd4jLong m = 0; m < M; m++) {
            for (Nd4jLong n = 0; n < N; n++) {
                dest[n * dest_stride_n] = betaZ * dest[n * dest_stride_n] + source[n];
            }//M

            source += N;
            dest += dest_stride_m;
        }//N
    }
}


template <typename T1, typename T2, typename T3>
static void parallel_batchedGemm3(const NDArray* vA, const NDArray* vB, NDArray* vC,
    const double alpha, const double beta, Nd4jLong start, Nd4jLong stop) {

    const T1* A = vA->bufferAsT<T1>();
    const T2* B = vB->bufferAsT<T2>();
    T3* C = vC->bufferAsT<T3>();
    Nd4jLong zero_strides[MAX_RANK] = {}; //zero strides
    const T1 alphaA = (T1)alpha;
    const T3 betaZ  = (T3)beta;

    const Nd4jLong* cShapeInfo = vC->shapeInfo();
    const Nd4jLong* bases = &(cShapeInfo[1]);
    Nd4jLong* aStrides = vA->stridesOf();
    Nd4jLong* bStrides = vB->stridesOf();
    const Nd4jLong* cStrides = vC->stridesOf();

    const char output_order = vC->ordering();

    const int aRank = vA->rankOf();
    const int bRank = vB->rankOf();
    const int cRank = vC->rankOf();

    int max_rank = cRank;

    const int M = vA->sizeAt(aRank - 2);
    const int K = vA->sizeAt(aRank - 1);
    const int N = vC->sizeAt(cRank - 1);
    Nd4jLong aStride_M = aStrides[aRank - 2];
    Nd4jLong aStride_K = aStrides[aRank - 1];
    Nd4jLong bStride_K = bStrides[bRank - 2];
    Nd4jLong bStride_N = bStrides[bRank - 1];
    Nd4jLong cStride_M = cStrides[cRank - 2];
    Nd4jLong cStride_N = cStrides[cRank - 1];

    if (aRank == 2) {
        aStrides = (Nd4jLong*)&zero_strides;
    }
    if (bRank == 2) {
        bStrides = (Nd4jLong*)&zero_strides;
    }

    Nd4jLong coords[MAX_RANK] = {};
    Nd4jLong* ptr_coords = (Nd4jLong*)&coords;
    sd::index2coords_C(start, max_rank - 2, bases, ptr_coords);
    //offset
    sd::triple_size_t offset = sd::offset_from_coords(aStrides, bStrides, cStrides, ptr_coords, max_rank - 2);
#if 0
    nd4j_printf("start:%d stop:%d \na: %d b:%d v:%d    \n", start, stop, offset.first, offset.second, offset.third);
#endif
    Nd4jLong loop = stop - start;
    bool out_order_f = cStride_M < cStride_N;
    T3* __restrict C_PTR_ORIG = C;
    bool allocate_buffer = out_order_f || cStride_N != 1;
#if 0
    if (cStride_N == 1)  allocate_buffer = true;
#endif
    if (allocate_buffer) {
        C_PTR_ORIG = new T3[M * N];
    }

    if (allocate_buffer) {
        // printf("---\n");
        for (Nd4jLong i = 0; i < loop; i++) {
            zero_buffer(C_PTR_ORIG, M, N);
            innerGemmWoChecks(M, N, K, alphaA, &(A[offset.first]), aStride_M, aStride_K, &(B[offset.second]), bStride_K, bStride_N, C_PTR_ORIG, N);
            T3* __restrict CX = &(C[offset.third]);
            copy_buffer(CX, C_PTR_ORIG, betaZ, M, N, cStride_M, cStride_N);
            offset = sd::inc_coords(bases, aStrides, bStrides, cStrides, ptr_coords, offset, max_rank, 2);
        }
    }
    else {
       // printf("+++cStride_M %d cStride_N %d M %d N %d\n", cStride_M, cStride_N, M , N);
        for (Nd4jLong i = 0; i < loop; i++) {
            T3* __restrict CX = &(C[offset.third]);
            scal_buffer(betaZ, CX, M, N, cStride_M);
            innerGemmWoChecks(M, N, K, alphaA, &(A[offset.first]), aStride_M, aStride_K, &(B[offset.second]), bStride_K, bStride_N, CX, cStride_M);

            offset = sd::inc_coords(bases, aStrides, bStrides, cStrides, ptr_coords, offset, max_rank, 2);
        }

    }

    if (allocate_buffer) {
        delete[] C_PTR_ORIG;
    }
}


template <typename T1, typename T2, typename T3>
static void batchedGemmUnPackC(const NDArray* vA, const NDArray* vB, NDArray* vC,
    const double alpha, const double beta, char out_order) {

    const Nd4jLong* cShapeInfo = vC->getShapeInfo();
    const Nd4jLong* bases = &(cShapeInfo[1]);
    const int max_rank = vC->rankOf();
    Nd4jLong batch_len = 1;
    for (int i = 0; i < max_rank - 2; i++) {
        batch_len *= bases[i];
    }


    auto func = [vA, vB, vC, alpha, beta](uint64_t thread_id, int64_t start, int64_t stop, int64_t increment) -> void {
#if 0
        auto timeStart = std::chrono::system_clock::now();
#endif
        parallel_batchedGemm3<T1, T2, T3>(vA, vB, vC, alpha, beta, start, stop);
#if 0
        auto timeEnd = std::chrono::system_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
        nd4j_printf("inner-time:: %lli\n", elapsed_time);
#endif
    };

    // samediff::Threads::parallel_tad(func, 0, batch_len);
    samediff::Threads::parallel_aligned_increment(func, 0, batch_len, 1, false);


}

//////////////////////////////////////////////////////////////////////////
// [bS,M,K] x [bS,K,N] = [bS,M,N]
// [bS,M,K] x    [K,N] = [bS,M,N]
//    [M,K] x [bS,K,N] = [bS,M,N]
// bS could stand for several axes
NDArray* MmulHelper::mmulNxN(const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder) {

    const int aRank = A->rankOf();
    const int bRank = B->rankOf();

    // input ranks validation
    if(aRank > bRank && bRank != 2)
        throw std::runtime_error("MmulHelper::mmulNxN: rank of B array should be equal 2 !");
    else if(bRank > aRank && aRank != 2)
        throw std::runtime_error("MmulHelper::mmulNxN: rank of A array should be equal 2 !");
    else if (aRank == bRank ) {
        for(int i = 0; i < aRank - 2; ++i)
            if(A->sizeAt(i) != B->sizeAt(i))
                throw std::runtime_error("MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication !");
    }

    if(A->sizeAt(-1) != B->sizeAt(-2))
        throw std::runtime_error("MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication !");

    // validation of C array
    std::vector<Nd4jLong> cExpectedShape = aRank > bRank ? A->getShapeAsVector() : B->getShapeAsVector();
    cExpectedShape[cExpectedShape.size() - 2] = A->sizeAt(-2);
    cExpectedShape[cExpectedShape.size() - 1] = B->sizeAt(-1);

    if(C != nullptr ) {
        if(!C->isSameShape(cExpectedShape))
            throw std::runtime_error("MmulHelper::mmulNxN: shape of C array is not suitable for AxB matrix multiplication !");
    }
    else {
        C = new NDArray(outOrder, cExpectedShape, B->dataType());
    }

    if (C->isEmpty())
        return C;


    BUILD_SINGLE_SELECTOR_THRICE(A->dataType(), batchedGemmUnPackC, (A, B, C, alpha, beta, outOrder), NUMERIC_TYPES);

    return C;
}

/*
//////////////////////////////////////////////////////////////////////////
NDArray* MmulHelper::mmulNxN(const NDArray* A, const NDArray* B, NDArray* C, const double alpha, const double beta, const char outOrder) {

    const int aRank = A->rankOf();
    const int bRank = B->rankOf();

    // input ranks validation
    if(aRank > bRank && bRank != 2)
        throw std::runtime_error("MmulHelper::mmulNxN: rank of B array should be equal 2 !");
    else if(bRank > aRank && aRank != 2)
        throw std::runtime_error("MmulHelper::mmulNxN: rank of A array should be equal 2 !");
    else if (aRank == bRank ) {
        for(int i = 0; i < aRank - 2; ++i)
            if(A->sizeAt(i) != B->sizeAt(i))
                throw std::runtime_error("MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication !");
    }

    if(A->sizeAt(-1) != B->sizeAt(-2))
        throw std::runtime_error("MmulHelper::mmulNxN: shapes of A and B arrays are not suitable for matrix multiplication !");

    // validation of C array
    std::vector<Nd4jLong> cExpectedShape = aRank > bRank ? A->getShapeAsVector() : B->getShapeAsVector();
    cExpectedShape[cExpectedShape.size() - 2] = A->sizeAt(-2);
    cExpectedShape[cExpectedShape.size() - 1] = B->sizeAt(-1);

    if(C != nullptr ) {
        if(!C->isSameShape(cExpectedShape))
            throw std::runtime_error("MmulHelper::mmulNxN: shape of C array is not suitable for AxB matrix multiplication !");
    }
    else {
        C = new NDArray(outOrder, cExpectedShape, B->dataType());
    }


    // multiplication
    const std::vector<int> dimsToExclude = ShapeUtils::evalDimsToExclude(C->rankOf(), {-2, -1});
    const Nd4jLong numOfSubArrs = ShapeUtils::getNumOfSubArrs(C->shapeInfo(), dimsToExclude);
    std::vector<Nd4jLong> idxRanges(2 * C->rankOf());

// #pragma omp parallel for schedule(guided) firstprivate(idxRanges)
        for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {

            ShapeUtils::evalIdxRangesForSubArr(i, C->shapeInfo(), dimsToExclude, idxRanges.data());
            NDArray cSubArr = (*C)(idxRanges);

            if(aRank > bRank) {
                NDArray aSubArr = (*A)(idxRanges);
                mmulMxM(&aSubArr, B, &cSubArr, 1., 0., outOrder);
            }
            else if(bRank > aRank) {
                NDArray bSubArr = (*B)(idxRanges);
                mmulMxM(A, &bSubArr, &cSubArr, 1., 0, outOrder);
            }
            else {
                NDArray aSubArr = (*A)(idxRanges);
                NDArray bSubArr = (*B)(idxRanges);
                mmulMxM(&aSubArr, &bSubArr, &cSubArr, 1., 0., outOrder);
            }
        }

    return C;
}

//////////////////////////////////////////////////////////////////////////////
// MXK x KxN = MxN
template <typename T1, typename T2, typename T3>
static void usualGemm(const char cOrder, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* vA, const int lda, const void* vB, const int ldb, const double beta, void* vC, const int ldc) {

    T1* A = reinterpret_cast<T1*>(const_cast<void*>(vA));
    T2* B = reinterpret_cast<T2*>(const_cast<void*>(vB));
    T3* C = reinterpret_cast<T3*>(vC);
    T3 alphaZ(alpha), betaZ(beta);

    const bool flagC = cOrder == 'f';
    const bool flagA = (flagC && transA) || (!flagC && !transA);
    const bool flagB = (flagC && transB) || (!flagC && !transB);

    // PRAGMA_OMP_PARALLEL_FOR_ARGS(OMP_IF(M*N > Environment::getInstance()->elementwiseThreshold()) schedule(guided))
    // for(uint row = 0; row < M; ++row) {

    //     T3* c = flagC ? (C + row) : (C + row * ldc);

    //     for(uint col = 0; col < N; ++col)
    //         c[flagC ? col * ldc : col] = 0;

    //     for(uint i = 0; i < K; ++i) {

    //         T3* b = flagB ? (B + i * ldb) : (B + i);
    //         T3* a = flagA ? (A + row * lda + i) : (A + row + i * lda);

    //         if(flagC) {
    //             for(uint col = 0; col < N; ++col) {
    //                 if(betaZ)
    //                     c[col * ldc] += a * b[flagB ? col : col * ldb] + betaZ * c[col * ldc];
    //                 else
    //                     c[col * ldc] += a * b[flagB ? col : col * ldb];
    //             }
    //         }
    //         else {
    //             for(uint col = 0; col < N; ++col) {
    //                 if(betaZ)
    //                     c[col] += a * b[flagB ? col : col * ldb] + betaZ * c[col];
    //                 else
    //                     c[col] += a * b[flagB ? col : col * ldb];
    //             }
    //         }
    //     }
    // }

    auto func = PRAGMA_THREADS_FOR_2D { ;
        for (auto row = start_x; row < stop_x; row += inc_x) {
            for (auto col = start_y; col < stop_y; col += inc_y) {
                T3 *c = flagC ? (C + row + col * ldc) : (C + row * ldc + col);
                T3 val = 0;

                for (uint i = 0; i < K; ++i) {
                    T3 a = flagA ? *(A + row * lda + i) : *(A + row + i * lda);
                    T3 b = flagB ? *(B + col + i * ldb) : *(B + col * ldb + i);
                    val += alphaZ * a * b;
                }

                if (betaZ)
                    *c = val + betaZ * *c;
                else
                    *c = val;
            }
        }
    };

    samediff::Threads::parallel_tad(func, 0, M, 1, 0, N, 1);
}

//////////////////////////////////////////////////////////////////////////////
// MXN x N = M
template <typename T1, typename T2, typename T3>
static void usualGemv(const char aOrder, const int M, const int N, const double alpha, const void* vA, const int lda, const void* vX, const int incx, const double beta, void* vY, const int incy) {

    T1* A = reinterpret_cast<T1*>(const_cast<void*>(vA));
    T2* X = reinterpret_cast<T2*>(const_cast<void*>(vX));
    T3* Y = reinterpret_cast<T3*>(vY);
    T3 alphaZ(alpha), betaZ(beta);

    const bool flagA = aOrder == 'f';

    auto func = PRAGMA_THREADS_FOR {
        for (auto row = start; row < stop; row += increment) {

            T3 *y = Y + row * incy;
            T3 val = 0;

            for (int i = 0; i < N; ++i) {
                T3 a = flagA ? *(A + row + i * lda) : *(A + row * lda + i);
                T3 x = *(X + i * incx);
                val += alphaZ * a * x;
            }

            if (betaZ)
                *y = val + betaZ * *y;
            else
                *y = val;
        }
    };

        samediff::Threads::parallel_tad(func, 0, M);
}
*/

//BUILD_TRIPLE_TEMPLATE(template void usualGemm, (const char cOrder, const bool transA, const bool transB, const int M, const int N, const int K, const double alpha, const void* A, const int lda, const void* B, const int ldb, const double beta, void* C, const int ldc), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
//BUILD_TRIPLE_TEMPLATE(template void usualGemv, (const char aOrder, const int M, const int N, const double alpha, const void* A, const int lda, const void* B, const int incx, const double beta, void* C, const int incy), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);
//BUILD_TRIPLE_TEMPLATE(template void usualDot,  (const Nd4jLong length, const double alpha, const void* vX, const Nd4jLong incx, const void* vY, const Nd4jLong incy, const double beta, void* vZ), LIBND4J_TYPES, FLOAT_TYPES, FLOAT_TYPES);

}
