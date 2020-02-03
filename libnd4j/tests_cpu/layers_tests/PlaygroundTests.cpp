/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 * Copyright (c) 2019 Konduit K.K.
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
 // Created by raver119 on 20.11.17.
 //

#include "testlayers.h"
#include <Graph.h>
#include <chrono>
#include <Node.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/profiling/GraphProfilingHelper.h>
#include <type_conversions.h>
#include <helpers/threshold.h>
#include <helpers/MmulHelper.h>
#include <ops/ops.h>
#include <OmpLaunchHelper.h>
#include <GradCheck.h>
#include <ops/declarable/helpers/im2col.h>
#include <Loops.h>
#include <RandomLauncher.h>
#include <ops/declarable/helpers/convolutions.h>

#include <ops/declarable/helpers/addBias.h>
#include <helpers/BenchmarkHelper.h>
#include <ops/declarable/helpers/scatter.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <array>
#include <performance/benchmarking/FullBenchmarkSuit.h>
#include <performance/benchmarking/LightBenchmarkSuit.h>
#include <iostream>
#include <ops/declarable/helpers/legacy_helpers.h>
#include <algorithm>
#include <numeric> 
#include <random>
using namespace nd4j;
using namespace nd4j::graph;

class PlaygroundTests : public testing::Test {
public:
	int numIterations = 100;
	int poolSize = 10;

	PlaygroundTests() {
		printf("\n");
		fflush(stdout);
	}
};
#define BENCH_RELEASE 1
#if defined(BENCH_RELEASE)

int bnch_cases[][4] = {
	{40, 2, 4, 2056},
	{2056, 2, 4, 19 },
	{16, 8, 16, 2056},
	{2056, 8, 16, 15 },
    {32, 7, 7, 96},
    {32, 32, 32, 16},
    {32, 600, 600, 3},
    {32, 112, 112, 32},
    {32, 7, 7, 1024},
	 {2,3,1,32}
};

constexpr int coords_k =   6;
#else
int bnch_cases[][4] = {
	{7,3,5,9},
	 {2,3,1,32},
     {32,32,2,2}
};

constexpr int coords_k = 0;

#endif

#define BENCH_TEST_F_ORDER 1
#define BENCH_TEST_F_ORDER_NC 1
#define BENCH_TEST_C_ORDER 1
#define BENCH_TEST_C_ORDER_NC 1

//#define BENCH_RELEASE 1
#if defined(BENCH_RELEASE)
int outter_loops =    10;
int inner_loops =   10;// 100;
#else
int outter_loops = 1;//  10;
int inner_loops = 1;// 10;// 100;
#endif
#if defined (_MSC_VER)
#define use(x)
#define escape(x)
#define clobber()
#else
template<typename T>
void use(T&& t) {
	__asm__ __volatile__("" :: "g" (t));
}

void escape(void* p) {
	__asm__ __volatile__("" : "+r" (p) :: "memory");
}

void clobber() {
	__asm__ __volatile__("" : : : "memory");
}
#endif
//#define bench_coords 1
template<typename Op,typename... Args>
void time_it(Op op,Nd4jLong totalFlops, Args&&... args) {
	std::vector<double> values;

	size_t n_1 = outter_loops > 1 ? outter_loops - 1 : 1;
	for (int e = 0; e < outter_loops; e++) {
		auto timeStart = std::chrono::system_clock::now();

		for (int i = 0; i < inner_loops; i++)
			op(std::forward<Args>(args)...);

		auto timeEnd = std::chrono::system_clock::now();
		auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
		values.emplace_back((double)elapsed_time / double(inner_loops));
	}

	std::sort(values.begin(), values.end());
	auto sum = std::accumulate(std::begin(values), std::end(values), 0.0);
	auto avg = (double)sum / outter_loops;
	auto sum_sq = std::accumulate(std::begin(values), std::end(values), 0.0, [&avg](int sumsq, int x) { return sumsq + (x - avg) * (x - avg); });

	if (totalFlops > 0) {
		nd4j_printf("Median: %f us\tAvg: %f (sd: %f)\tFlops: %f Mflops\n",
			values[values.size() / 2], avg, sqrt(sum_sq / n_1),
			(double)(totalFlops) / avg);
	} 
	else {
		nd4j_printf("Median: %f us\tAvg: %f (sd: %f)\n",
			values[values.size() / 2], avg, sqrt(sum_sq / n_1));
	}
}

template<typename T>
void fill_random(nd4j::NDArray& arr) {
	Nd4jLong coords[MAX_RANK] = {};
	Nd4jLong add_coords[MAX_RANK] = {}; //zero
	std::random_device rd;
	std::mt19937 gen(rd());
	//for floats
	std::uniform_real_distribution<T> dis((T)0.0,(T) 2.0);
	T* x = arr.bufferAsT<T>(); 
	Nd4jLong* shapeInfo = arr.getShapeInfo();  
	Nd4jLong* strides = arr.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]); 

	add_coords[rank - 1] = 1; 
	size_t t = 1;
	for (size_t i = 0; i < rank; i++) {
		t *= bases[i];
	}
	size_t offset = 0;
	for (size_t i = 0; i < t; i++) {
		x[offset] = dis(gen);
		offset = nd4j::ops::helpers::move_by_coords(bases, strides, coords, add_coords, rank);
	}
}

 
 
void check_correctness_with_experimental( Context& ctx, const NDArray& x, const NDArray& y, NDArray& z,bool isNCHW) {

	auto expected = z.ulike();
	nd4j::ops::helpers::addBias_Experimental(ctx, x, y, expected, isNCHW, true);
	ASSERT_TRUE(expected.equalsTo(z));

}

void check_correctness_with_base(Context& ctx, const NDArray& x, const NDArray& y, NDArray& z, bool isNCHW) {

	auto expected = z.ulike();
	nd4j::ops::helpers::addBias(ctx, x, y, expected, isNCHW);
	ASSERT_TRUE(expected.equalsTo(z));

}
#if defined(BENCH_TEST_C_ORDER)
TEST_F(PlaygroundTests, test_bias_base) {
	 

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('c', { bnch_cases[k][3] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias, total_flops, ctx, x, y, z, false);

	};
}

 
TEST_F(PlaygroundTests, test_bias_experimental_continuous) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('c', { bnch_cases[k][3] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, false, false);
		 
	    
		check_correctness_with_base(ctx, x, y, z, false);
	};

}

#endif
#if defined(BENCH_TEST_C_ORDER_NC)
TEST_F(PlaygroundTests, test_bias_NC_base) {


	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('c', { bnch_cases[k][1] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias, total_flops, ctx, x, y, z, true);

	};
}


TEST_F(PlaygroundTests, test_bias_NC_experimental_continuous) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('c', { bnch_cases[k][1] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, true, false);

		//check
		check_correctness_with_base(ctx, x, y, z, true);
	};

}


#endif

#if defined(BENCH_TEST_F_ORDER)
TEST_F(PlaygroundTests, test_bias_Fortran_base) {


	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('f', { bnch_cases[k][3] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias, total_flops, ctx, x, y, z, false);

	};
}

TEST_F(PlaygroundTests, test_bias_Fortran_experimental_continuous) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('f', { bnch_cases[k][3] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, false, false);


		check_correctness_with_base(ctx, x, y, z, false);
	};

}

#endif

#if defined(BENCH_TEST_F_ORDER_NC)
TEST_F(PlaygroundTests, test_bias_Fortran_NC_base) {


	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('f', { bnch_cases[k][1] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias, total_flops, ctx, x, y, z, true);

	};
}

TEST_F(PlaygroundTests, test_bias_Fortran_NC_experimental_continuous) {

	for (int k = 0; k < sizeof(bnch_cases) / sizeof(bnch_cases[0]); k++) {

		Nd4jLong total_flops = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
		auto x = NDArrayFactory::create<float>('f', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
		auto y = NDArrayFactory::create<float>('f', { bnch_cases[k][1] });
		auto z = x.ulike();

		Context ctx(1);

		fill_random<float>(x);
		fill_random<float>(y);
		//nd4j::ops::biasadd op;  
		nd4j_printf("NdArray: {%ld, %ld, %ld, %ld };\n", bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3]);
		time_it(nd4j::ops::helpers::addBias_Experimental, total_flops, ctx, x, y, z, true, false);

		//check
		check_correctness_with_base(ctx, x, y, z, true);
	};

}

#endif


#if defined(bench_coords) 

TEST_F(PlaygroundTests, test_coord1) {
	outter_loops = 4;
	inner_loops = 4;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK];
	Nd4jLong add_coords[MAX_RANK]; 
	Nd4jLong* shapeInfo = x.getShapeInfo();
	shape::index2coords(0, shapeInfo, coords);
	shape::index2coords(inc, shapeInfo, add_coords);
	 
	auto ops = [&]() {
		for (size_t i = 0; i < t; i += inc) {
			shape::index2coords(i, shapeInfo, coords);
			//add to prevent code elimination
			use(coords); 
		}
	};
	time_it(ops, 0 );
 }


TEST_F(PlaygroundTests, test_coord2) {
	outter_loops = 4;
	inner_loops = 4;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK];
	Nd4jLong add_coords[MAX_RANK];
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords);
	shape::index2coords(inc, shapeInfo, add_coords); 
	escape(&rank);
	escape(coords);
	escape(bases); 
	escape(add_coords);
	escape(strides);
	auto ops = [&]() { 
			for (size_t i = 0; i < t; i += inc) {
				nd4j::ops::helpers::move_by_coords(bases, strides, coords, add_coords, rank);
				use(coords); 
				use(i);
		      }
	};
	time_it(ops, 0);
}


TEST_F(PlaygroundTests, test_coord3) {
	outter_loops = 4;
	inner_loops = 4;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK];
	Nd4jLong add_coords[MAX_RANK];
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords);
	shape::index2coords(inc, shapeInfo, add_coords); 
	escape(&rank);
	escape(coords);
	escape(bases); 
	escape(add_coords);
	escape(strides);
	auto ops = [&]() {
		for (size_t i = 0; i < t; i += inc) {
				nd4j::ops::helpers::move_by_coords<4>(bases, strides,  coords, add_coords);
				use(coords); 
				use(i);
			} 
	};
	time_it(ops, 0);
}

TEST_F(PlaygroundTests, test_coord4_offset) {
	outter_loops = 4;
	inner_loops = 4;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK];
	Nd4jLong add_coords[MAX_RANK];
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords);
	shape::index2coords(inc, shapeInfo, add_coords); 
	size_t offset = 0;
	escape(&rank);
	escape(coords);
	escape(bases);
	escape(&offset);
	escape(add_coords);
	escape(strides);
	auto ops = [&]() {
		for (size_t i = 0; i < t; i += inc) {
				offset = nd4j::ops::helpers::move_by_coords(bases, strides, coords, add_coords,rank-1);
				use(coords); 
				use(offset);
				use(i);
			} 
	};
	time_it(ops, 0);
}

TEST_F(PlaygroundTests, test_coord5_offset) {
	outter_loops = 4;
	inner_loops = 4;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK];
	Nd4jLong add_coords[MAX_RANK];
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords);
	shape::index2coords(inc, shapeInfo, add_coords);
	size_t offset = 0;
	escape(&rank);
	escape(coords);
	escape(bases);
	escape(&offset);
	escape(add_coords);
	escape(strides);
	auto ops = [&]() {
		for (size_t i = 0; i < t; i += inc) {
			offset = nd4j::ops::helpers::move_by_coords<3>(bases, strides, coords, add_coords);
			use(coords); 
			use(offset);
			use(i);
		}
	};
	time_it(ops, 0);
}


TEST_F(PlaygroundTests, test_coord6_loop_offset) {
	outter_loops = 4;
	inner_loops = 40;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK];
	Nd4jLong add_coords[MAX_RANK];
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords);
	shape::index2coords(inc, shapeInfo, add_coords); 
	size_t offset = 0;
	escape(&rank);
	escape(coords); 
	auto ops = [&]() {
		size_t zStrideB = strides[0];
		size_t zStrideH = strides[1];
		size_t zStrideW = strides[2];
		escape(&zStrideB);
		escape(&zStrideH);
		escape(&zStrideW);
		for (uint b = 0; b < bnch_cases[k][0]; b+=inc)
			for (uint h = 0; h < bnch_cases[k][1]; h++)
				for (uint w = 0; w < bnch_cases[k][2]; w++) {
						offset = b * zStrideB + h * zStrideH + w * zStrideW  ;
						use(offset);
						use(h);
						use(w);
						use(b);
					}
	};
	time_it(ops, 0);
}

 
TEST_F(PlaygroundTests, test_coord7_offset) {
	outter_loops = 4;
	inner_loops = 40;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK]; 
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords);  
	size_t offset = 0; 
	escape(&rank);
	escape(coords);
	escape(bases);
	escape(&offset); 
	escape(strides);
	auto ops = [&]() {
		for (size_t i = 0; i < t; i+=inc) {
			 
			offset = nd4j::ops::helpers::inc_by_coords<3>(bases, strides, coords, offset); 
			use(coords); 
			use(offset);
			use(i);
		}
	};
	time_it(ops, 0);
}


TEST_F(PlaygroundTests, test_coord8_offset) {
	outter_loops = 4;
	inner_loops = 40;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK]; 
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords);  
	size_t offset = 0; 
	escape(&rank);
	escape(coords);
	escape(bases);
	escape(&offset);
	escape(strides);
	auto ops = [&]() {
		for (size_t i = 0; i < t; i+=inc) {

			offset = nd4j::ops::helpers::inc_by_coords(bases, strides, coords, offset,rank-1);
			use(coords); 
			use(offset);  
			use(i);
		}
	};
	time_it(ops, 0);
}


TEST_F(PlaygroundTests, test_coord9_offset) {
	outter_loops = 4;
	inner_loops = 40;
	constexpr int k = coords_k;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = bnch_cases[k][3];
	fill_random<float>(x);
	Nd4jLong coords[MAX_RANK];
	Nd4jLong adjusts[MAX_RANK];
	Nd4jLong* shapeInfo = x.getShapeInfo();
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	shape::index2coords(0, shapeInfo, coords); 
	nd4j::ops::helpers::get_adjusts_for_inc(strides, bases, adjusts, rank - 1);
	  size_t offset = 0; 
	  escape(&rank);
	  escape(coords);
	  escape(bases);
	  escape(&offset);
	  escape(adjusts);
	  escape(strides);
	auto ops = [&]() {
		for (size_t i = 0; i < t; i += inc) {

			offset = nd4j::ops::helpers::inc_by_coords2(bases, strides, adjusts, coords, offset, rank - 1);
			use(coords);
			use(offset);  
			use(i);
		}
	};
	time_it(ops, 0);
}

TEST_F(PlaygroundTests, test_coord_correctness) {
	constexpr int k = 0;
	Nd4jLong t = bnch_cases[k][0] * bnch_cases[k][1] * bnch_cases[k][2] * bnch_cases[k][3];
	auto x = NDArrayFactory::create<float>('c', { bnch_cases[k][0], bnch_cases[k][1], bnch_cases[k][2], bnch_cases[k][3] });
	size_t inc = 1;// bnch_cases[k][3];


	Nd4jLong coords[MAX_RANK];
	Nd4jLong coords2[MAX_RANK];
	Nd4jLong coords3[MAX_RANK];
	Nd4jLong coords4[MAX_RANK] = {};
	Nd4jLong coords5[MAX_RANK] = {};
	Nd4jLong add_coords[MAX_RANK]; 
	Nd4jLong* shapeInfo = x.getShapeInfo();
	shape::index2coords(0, shapeInfo, coords2);
	shape::index2coords(0, shapeInfo, coords3);
	shape::index2coords(inc, shapeInfo, add_coords);
	Nd4jLong* strides = x.stridesOf();
	Nd4jLong rank = shapeInfo[0];
	Nd4jLong* bases = &(shapeInfo[1]);
	size_t offset = 0;
	size_t offset_uni = 0;
	for (size_t i = 0; i < t; i += inc) {

		shape::index2coords(i, shapeInfo, coords);

		for (int j = 0; j < rank; j++) {
			ASSERT_TRUE(coords2[j] == coords3[j]);
			ASSERT_TRUE(coords[j] == coords2[j]);
		}
		//increment should be last
		nd4j::ops::helpers::move_by_coords(bases,strides, coords2, add_coords, rank);
		nd4j::ops::helpers::move_by_coords<4>(bases,strides, coords3, add_coords);
		ASSERT_EQ(offset, i);
		offset = nd4j::ops::helpers::inc_by_coords<4>(bases, strides, coords4, offset);
		ASSERT_EQ(offset_uni, i);
		offset_uni = nd4j::ops::helpers::inc_by_coords(bases, strides, coords5, offset_uni, rank);
	}



}

#endif
