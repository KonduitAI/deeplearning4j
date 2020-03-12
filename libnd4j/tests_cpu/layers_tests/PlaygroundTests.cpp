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
#include <graph/Graph.h>
#include <chrono>
#include <graph/Node.h>
#include <ops/declarable/CustomOperations.h>
#include <graph/profiling/GraphProfilingHelper.h>
#include <loops/type_conversions.h>
#include <helpers/threshold.h>
#include <helpers/MmulHelper.h>
#include <ops/ops.h>
#include <helpers/OmpLaunchHelper.h>
#include <helpers/GradCheck.h>
#include <ops/declarable/helpers/im2col.h>
#include <helpers/Loops.h>
#include <helpers/RandomLauncher.h>
#include <ops/declarable/helpers/convolutions.h>

#include <helpers/BenchmarkHelper.h>
#include <ops/declarable/helpers/scatter.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/ConstantTadHelper.h>
#include <array>
#include <performance/benchmarking/FullBenchmarkSuit.h>
#include <performance/benchmarking/LightBenchmarkSuit.h>
#include <performance/benchmarking/global_timers.h>

#include <ops/declarable/helpers/legacy_helpers.h>
#include <ops/declarable/helpers/addBias.h>

using namespace sd;
using namespace sd::graph;

class PlaygroundTests : public testing::Test {
public:
    int numIterations = 3;
    int poolSize = 10;

    PlaygroundTests() {
    }
};

TEST_F(PlaygroundTests, test_avx) {
    nd4j_printf("Optimal level: %i; Binary level: %i;\n", ::optimalLevel(), ::binaryLevel());
}

TEST_F(PlaygroundTests, test_split_1) {
    auto axis = NDArrayFactory::create<int>(1);
    auto array = NDArrayFactory::create<double>('c', {1, 512});

    auto outA = NDArrayFactory::create<double>('c', {1, 128});
    auto outB = outA.ulike();
    auto outC = outA.ulike();
    auto outD = outA.ulike();

    sd::ops::split op;

    auto timeStart = std::chrono::system_clock::now();

    auto result = op.execute({&axis, &array},{&outA, &outB, &outC, &outD}, {4});

    auto timeEnd = std::chrono::system_clock::now();
    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
    nd4j_printf("Split time: %lld us;\n", outerTime);

    ASSERT_EQ(Status::OK(), result);
}

TEST_F(PlaygroundTests, test_concat_1) {
    auto t = NDArrayFactory::create<double>('c', {1, 28});
    auto u = NDArrayFactory::create<double>('c', {1, 128});
    auto v = NDArrayFactory::create<int>(1);
    auto z = NDArrayFactory::create<double>('c', {1, 156});

    sd::ops::concat op;
    auto timeStart = std::chrono::system_clock::now();

    auto status = op.execute({&t, &u, &v}, {&z}, {true});

    auto timeEnd = std::chrono::system_clock::now();
    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();

    nd4j_printf("Concat time: %lld\n", outerTime);

    ASSERT_EQ(Status::OK(), status);
}

TEST_F(PlaygroundTests, test_gather_1) {
    // this test will run ONLY if this model exists
    if (sd::graph::getFileSize("resources/Bert_minimal_model/bert_minimal_model.fb") < 0)
        return;

    auto x = NDArrayFactory::create<float>('c', {30522, 768});
    auto y = NDArrayFactory::fromNpyFile("resources/Bert_minimal_model/bert_minimal_input_IteratorGetNext.numpy");
    auto z = NDArrayFactory::create<float>('c', {4, 128, 768});

    x.linspace(1.0f, 0.3f);


    sd::ops::gather op;
    auto timeStart = std::chrono::system_clock::now();

    auto status = op.execute({&x, &y}, {&z});

    auto timeEnd = std::chrono::system_clock::now();
    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();

    nd4j_printf("Gather time: %lld\n", outerTime);
}

TEST_F(PlaygroundTests, test_matmul_1) {
    auto x = NDArrayFactory::create<float>('c', {512, 768});
    auto y = NDArrayFactory::create<float>('c', {768, 768});
    auto z = NDArrayFactory::create<float>('c', {512, 768});

    x.linspace(1.0f, 0.3f);
    y.linspace(1.0f, 0.2f);

    sd::ops::matmul op;
    auto timeStart = std::chrono::system_clock::now();

    auto status = op.execute({&x, &y}, {&z}, {0, 0});

    auto timeEnd = std::chrono::system_clock::now();
    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();

    nd4j_printf("Time: %lld\n", outerTime);
}

TEST_F(PlaygroundTests, test_matmul_2) {
    auto x1 = NDArrayFactory::create<float>('c', {4, 12, 128, 128});
    auto y1 = NDArrayFactory::create<float>('c', {4, 12, 128, 64});
    auto z1 = NDArrayFactory::create<float>('c', {4, 12, 128, 64});

    auto x2 = NDArrayFactory::create<float>('c', {512, 768});
    auto y2 = NDArrayFactory::create<float>('c', {768, 768});
    auto z2 = NDArrayFactory::create<float>('c', {512, 768});

    sd::ops::matmul op;
    auto timeStart1 = std::chrono::system_clock::now();

    op.execute({&x1, &y1}, {&z1}, {0, 0});

    auto timeStart2 = std::chrono::system_clock::now();

    op.execute({&x2, &y2}, {&z2}, {0, 0});

    auto timeEnd = std::chrono::system_clock::now();

    auto t1 = std::chrono::duration_cast<std::chrono::microseconds>(timeStart2 - timeStart1).count();
    auto t2 = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart2).count();

    nd4j_printf("Time 1: %lld; Time 2: %lld;\n", t1, t2);
}


TEST_F(PlaygroundTests, test_reduce_mean_1) {
    auto x = NDArrayFactory::create<float>('c', {512, 768});
    auto y = NDArrayFactory::create<int>(1);
    auto z = NDArrayFactory::create<float>('c', {512});

    x.assign(1.f);

    sd::ops::reduce_mean op;

    auto timeStart = std::chrono::system_clock::now();

    auto status = op.execute({&x, &y}, {&z});

    auto timeEnd = std::chrono::system_clock::now();
    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();

    ASSERT_EQ(Status::OK(), status);

    nd4j_printf("Time: %lld us;\n", outerTime);

    z.printLinearBuffer();
}

TEST_F(PlaygroundTests, test_reduce_mean_2) {
    auto x = NDArrayFactory::create<float>('c', {512, 768});
    auto y = NDArrayFactory::create<int>(1);
    auto z = NDArrayFactory::create<float>('c', {512});

    auto rows = x.sizeAt(0);
    auto cols = x.sizeAt(1);

    auto inBuff = x.bufferAsT<float>();
    auto outBuff = x.bufferAsT<float>();

    auto timeStart = std::chrono::system_clock::now();

    for (int t = 0; t < rows; t++) {
        auto in = inBuff + (t * cols);

        float sum = 0.f;
        for (int e = 0; e < cols; e++) {
            sum = simdOps::Mean<float, float>::update(sum, simdOps::Mean<float, float>::op(in[e], nullptr), nullptr);
        }

        outBuff[t] = sum / cols;
    }

    auto timeEnd = std::chrono::system_clock::now();
    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
    nd4j_printf("Time: %lld us;\n", outerTime);

}

TEST_F(PlaygroundTests, test_softmax_1) {
    auto x = NDArrayFactory::create<float>('c', {4, 12, 128, 128});
    auto y = NDArrayFactory::create<int>(3);
    auto z = x.ulike();
    x.linspace(1.f);

    std::vector<Nd4jLong> values;

    sd::ops::softmax op;

    for (int e = 0; e < 1; e++) {
        auto timeStart = std::chrono::system_clock::now();

        op.execute({&x}, {&z}, {3});

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);
}

TEST_F(PlaygroundTests, test_biasAdd_1) {
    auto x = NDArrayFactory::create<float>('c', {512, 3072});
    auto y = NDArrayFactory::create<float>('c', {3072});

    std::vector<Nd4jLong> values;

    sd::ops::biasadd op;

    for (int e = 0; e < 100; e++) {
        auto timeStart = std::chrono::system_clock::now();

        op.execute({&x, &y}, {&x});

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);
    sd::GlobalTimers::getInstance()->displayTimers();
}


TEST_F(PlaygroundTests, test_bert_full_1) {
    // this test will run ONLY if this model exists
    if (sd::graph::getFileSize("resources/BertFull/model.fb") < 0)
        return;

    auto graph = GraphExecutioner::importFromFlatBuffers("resources/BertFull/model.fb");
    nd4j_printf("Graph successfully loaded\n", "");
    auto t = NDArrayFactory::fromNpyFile("resources/BertFull/in0_IteratorGetNext.npy");
    auto u = NDArrayFactory::fromNpyFile("resources/BertFull/in1_IteratorGetNext_1.npy");
    auto v = NDArrayFactory::fromNpyFile("resources/BertFull/in2_IteratorGetNext_4.npy");
    auto z = NDArrayFactory::fromNpyFile("resources/BertFull/out_loss-Softmax.npy");

    //graph->printOut();

    graph->tagInplaceNodes();

    graph->getVariableSpace()->putVariable(658,0, t);
    graph->getVariableSpace()->putVariable(659,0, u);
    graph->getVariableSpace()->putVariable(660,0, v);


    // validating graph now
    auto status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(1620));

    auto array = graph->getVariableSpace()->getVariable(1620)->getNDArray();
    ASSERT_EQ(z, *array);



    sd::Environment::getInstance()->setProfiling(true);
    auto profile = GraphProfilingHelper::profile(graph, 1);

    profile->printOut();

    sd::Environment::getInstance()->setProfiling(false);
    delete profile;

/*
    std::vector<Nd4jLong> values;

    for (int e = 0; e < 1; e++) {
        auto timeStart = std::chrono::system_clock::now();

        GraphExecutioner::execute(graph);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);
*/
    delete graph;
}


TEST_F(PlaygroundTests, test_bert_1) {
    // this test will run ONLY if this model exists
    if (sd::graph::getFileSize("resources/Bert_minimal_model/bert_minimal_model.fb") < 0)
        return;

    auto graph = GraphExecutioner::importFromFlatBuffers("resources/Bert_minimal_model/bert_minimal_model.fb");

    auto t = NDArrayFactory::fromNpyFile("resources/Bert_minimal_model/bert_minimal_input_IteratorGetNext.numpy");
    auto u = NDArrayFactory::fromNpyFile("resources/Bert_minimal_model/bert_minimal_input_IteratorGetNext_1.numpy");
    auto v = NDArrayFactory::fromNpyFile("resources/Bert_minimal_model/bert_minimal_input_IteratorGetNext_4.numpy");
    auto z = NDArrayFactory::fromNpyFile("resources/Bert_minimal_model/bert_minimal_model_output.numpy");

    //graph->printOut();

    graph->tagInplaceNodes();

    graph->getVariableSpace()->putVariable(85,0, t);
    graph->getVariableSpace()->putVariable(86,0, u);
    graph->getVariableSpace()->putVariable(87,0, v);

/*
    // validating graph now
    auto status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(198));

    auto array = graph->getVariableSpace()->getVariable(198)->getNDArray();
    ASSERT_EQ(z, *array);
*/

    sd::Environment::getInstance()->setProfiling(true);
    auto profile = GraphProfilingHelper::profile(graph, 1);

    profile->printOut();

    sd::Environment::getInstance()->setProfiling(false);
    delete profile;

    /*
    std::vector<Nd4jLong> values;

    for (int e = 0; e < 1; e++) {
        auto timeStart = std::chrono::system_clock::now();

        GraphExecutioner::execute(graph);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
       values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);
    */

    delete graph;
}

TEST_F(PlaygroundTests, test_bert_2) {
    // this test will run ONLY if this model exists
    if (sd::graph::getFileSize("resources/Bert_minimal_model/bert_like_ops.fb") < 0)
        return;

    auto graph = GraphExecutioner::importFromFlatBuffers("resources/Bert_minimal_model/bert_like_ops.fb");

    //graph->printOut();

    graph->tagInplaceNodes();


/*
    // validating graph now
    auto status = GraphExecutioner::execute(graph);
    ASSERT_EQ(Status::OK(), status);
    ASSERT_TRUE(graph->getVariableSpace()->hasVariable(198));

    auto array = graph->getVariableSpace()->getVariable(198)->getNDArray();
    ASSERT_EQ(z, *array);
*/

    sd::Environment::getInstance()->setProfiling(true);
    auto profile = GraphProfilingHelper::profile(graph, 1);

    profile->printOut();

    sd::Environment::getInstance()->setProfiling(false);
    delete profile;

/*
    std::vector<Nd4jLong> values;

    for (int e = 0; e < 1; e++) {
        auto timeStart = std::chrono::system_clock::now();

        GraphExecutioner::execute(graph);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);
*/
    delete graph;
}




TEST_F(PlaygroundTests, test_one_off_ops_1) {
    int pool = 1000;
    std::vector<NDArray*> aX(pool);
    std::vector<NDArray*> aY(pool);
    std::vector<NDArray*> aZ(pool);

    for (int e = 0; e < pool; e++) {
        aX[e] = NDArrayFactory::create_<float>('c', {4, 12, 128, 128});
        aY[e] = NDArrayFactory::create_<float>('c', {4, 1, 128, 128});
        aZ[e] = NDArrayFactory::create_<float>('c', {4, 12, 128, 128});

        aX[e]->assign(119 * (e+1));
        aY[e]->assign(119 * (e+3));
    }

    std::vector<Nd4jLong> values;
    Context ctx(1);

    sd::ops::add op;

    for (int e = 0; e < 1000; e++) {
        auto x = aX[e < pool ? e : e % pool];
        auto y = aY[e < pool ? e : e % pool];
        auto z = aZ[e < pool ? e : e % pool];

        auto timeStart = std::chrono::system_clock::now();

        op.execute({x, y}, {z});

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);

    for (int e = 0; e < pool; e++) {
        delete aX[e];
        delete aY[e];
        delete aZ[e];
    }
}


TEST_F(PlaygroundTests, test_matmul_perf_1) {
    auto x  = NDArrayFactory::create<double>('c', {512, 768});
    auto y  = NDArrayFactory::create<double>('c', {768, 768});
    auto exp = NDArrayFactory::create<double>('f', {512, 768});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {}, {});

    GlobalTimers::getInstance()->displayTimers();
}

TEST_F(PlaygroundTests, test_matmul_perf_2) {
    auto x  = NDArrayFactory::create<double>('c', {4, 12, 128, 64});
    auto y  = NDArrayFactory::create<double>('c', {4, 12, 128, 64});
    auto exp = NDArrayFactory::create<double>('f', {4, 12, 128, 128});

    x.linspace(1.);
    y.linspace(0.5, 0.5);

    sd::ops::matmul op;
    auto results = op.evaluate({&x, &y}, {0, 1});

    GlobalTimers::getInstance()->displayTimers();
}

TEST_F(PlaygroundTests, test_strided_slice_perf) {
    auto matrix = NDArrayFactory::create<double>('c', {5, 2});
    auto b = NDArrayFactory::create<double>('c', {1}, {0.});
    auto e = NDArrayFactory::create<double>('c', {1}, {1});
    auto s = NDArrayFactory::create<double>('c', {1}, {1});

    auto exp = NDArrayFactory::create<double>('c', {2}, {1.0f, 2.0f});

    matrix.linspace(1);

    sd::ops::strided_slice op;
    auto result = op.evaluate({&matrix, &b, &e, &s}, {}, {0, 0, 0, 0, 1});

    sd::GlobalTimers::getInstance()->displayTimers();
}

TEST_F(PlaygroundTests, test_strided_slice_perf_2) {
    auto matrix = NDArrayFactory::create<double>('c', {4, 128, 768});
    auto arg1 = NDArrayFactory::create<int>('c', {3}, {0, 0, 0});
    auto arg2 = NDArrayFactory::create<int>('c', {3}, {0, 1, 0});
    auto arg3 = NDArrayFactory::create<int>('c', {3}, {1, 1, 1});

    matrix.linspace(1);

    sd::ops::strided_slice op;
    auto result = op.evaluate({&matrix, &arg1, &arg2, &arg3},{}, {5, 0, 5, 0, 0});
    GlobalTimers::getInstance()->displayTimers();
}

// printf("strided_slice arguments: %d %d %d %d %d\n", begin_mask, ellipsis_mask, end_mask, new_axis_mask, shrink_axis_mask);
// 5 0 5 0 0


TEST_F(PlaygroundTests, test_permut_perf) {



    NDArray array  = NDArrayFactory::create<double>('c', {4, 12, 128, 64});

    NDArray* x = &array;

    const int rank = x->rankOf();

    std::vector<int> permut(rank);

    for (int i = 0; i < rank-2; ++i)
        permut[i] = i;

    permut[rank-2] = rank - 1;
    permut[rank-1] = rank - 2;

    x->linspace(1.);

    NDArray* yT = new NDArray(x->permute(permut));

    sd::GlobalTimers::getInstance()->displayTimers();
}

TEST_F(PlaygroundTests, test_broadcast_1) {
    int pool = 1000;
    std::vector<NDArray*> aX(pool);
    std::vector<NDArray*> aY(pool);
    std::vector<NDArray*> aZ(pool);

    for (int e = 0; e < pool; e++) {
        aX[e] = NDArrayFactory::create_<float>('c', {512, 3072});
        aY[e] = NDArrayFactory::create_<float>('c', {3072});
        aZ[e] = NDArrayFactory::create_<float>('c', {512, 3072});

        aX[e]->assign(119 * (e+1));
        aY[e]->assign(119 * (e+3));
    }

    std::vector<Nd4jLong> values;
    Context ctx(1);

    sd::ops::biasadd op;

    for (int e = 0; e < 1000; e++) {
        auto x = aX[e < pool ? e : e % pool];
        auto y = aY[e < pool ? e : e % pool];
        auto z = aZ[e < pool ? e : e % pool];

        auto timeStart = std::chrono::system_clock::now();

        //op.execute({x, y}, {z});
        sd::ops::helpers::addBias(ctx, *x, *y, *z, false);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);

    for (int e = 0; e < pool; e++) {
        delete aX[e];
        delete aY[e];
        delete aZ[e];
    }
}


/*
TEST_F(PlaygroundTests, test_broadcast_1) {
    int pool = 500;
    std::vector<NDArray*> aX(pool);
    std::vector<NDArray*> aY(pool);
    std::vector<NDArray*> aZ(pool);

    for (int e = 0; e < pool; e++) {
        aX[e] = NDArrayFactory::create_<float>('c', {512, 3072});
        aY[e] = NDArrayFactory::create_<float>('c', {768});
        aZ[e] = NDArrayFactory::create_<float>('c', {512, 3072});

        aX[e]->assign( (e+1) / 119);
        aY[e]->assign( (e+3) / 119);
    }



    std::vector<Nd4jLong> values;

    for (int e = 0; e < 1000; e++) {
        auto x = aX[e < pool ? e : e % pool];
        auto y = aY[e < pool ? e : e % pool];
        auto z = aZ[e < pool ? e : e % pool];

        auto timeStart = std::chrono::system_clock::now();

        //x->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), *y, *z);
        x->applyTransform(transform::Tanh, *z, nullptr);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);

    for (int e = 0; e < pool; e++) {
        delete aX[e];
        delete aY[e];
        delete aZ[e];
    }
}

*/
/*

TEST_F(PlaygroundTests, test_s_0) {
    std::vector<std::vector<Nd4jLong>> shapes = {{32, 224, 224, 3}, {32, 56, 56, 64}, {32, 7, 7, 512}};
    std::vector<int> threads = {1, 2, 4, 8, 16};

    for (auto shape: shapes) {
        for (auto t: threads) {
            sd::Environment::getInstance()->setMaxMasterThreads(t);

            auto x = NDArrayFactory::create<float>('c', shape);
            auto y = NDArrayFactory::create<float>('c', {shape[3]});
            auto z = x.ulike();

            std::vector<Nd4jLong> values;
            Context ctx(1);
            ctx.setInputArray(0, &x);
            ctx.setInputArray(1, &y);
            ctx.setOutputArray(0, &z);

            sd::ops::biasadd op;


            for (int e = 0; e < 10000; e++) {
                auto timeStart = std::chrono::system_clock::now();

                op.execute(&ctx);
                sd::ops::helpers::addBias(ctx, x, y, z, false);

                auto timeEnd = std::chrono::system_clock::now();
                auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
                values.emplace_back(outerTime);
            }

            std::sort(values.begin(), values.end());

            nd4j_printf("Shape: [%lld, %lld, %lld, %lld]; Threads: [%i]; Time: %lld us;\n", shape[0], shape[1], shape[2], shape[3], t, values[values.size() / 2]);
        }
    }
}

TEST_F(PlaygroundTests, test_s_1) {
    std::vector<std::vector<Nd4jLong>> shapes = {{32, 3, 224, 224}, {32, 64, 56, 56}, {32, 512, 7, 7}};
    std::vector<int> threads = {1, 2, 4, 8, 16};

    for (auto shape: shapes) {
        for (auto t: threads) {
            sd::Environment::getInstance()->setMaxMasterThreads(t);

            auto x = NDArrayFactory::create<float>('c', shape);
            auto y = NDArrayFactory::create<float>('c', {shape[1]});
            auto z = x.ulike();

            std::vector<Nd4jLong> values;
            Context ctx(1);
            ctx.setInputArray(0, &x);
            ctx.setInputArray(1, &y);
            ctx.setOutputArray(0, &z);

            sd::ops::biasadd op;


            for (int e = 0; e < 10000; e++) {
                auto timeStart = std::chrono::system_clock::now();

                //op.execute({&x, &y}, {&z}, {true});
                sd::ops::helpers::addBias(ctx, x, y, z, true);

                auto timeEnd = std::chrono::system_clock::now();
                auto outerTime = std::chrono::duration_cast<std::chrono::microseconds>(timeEnd - timeStart).count();
                values.emplace_back(outerTime);
            }

            std::sort(values.begin(), values.end());

            nd4j_printf("Shape: [%lld, %lld, %lld, %lld]; Threads: [%i]; Time: %lld us;\n", shape[0], shape[1], shape[2], shape[3], t, values[values.size() / 2]);
        }
    }
}
*/

/*
TEST_F(PlaygroundTests, test_s_0) {
    auto x = NDArrayFactory::create<float>('c', {32, 112, 112, 16});
    auto y = NDArrayFactory::create<float>('c', {16});
    auto z = x.ulike();

    std::vector<Nd4jLong> values;
    Context ctx(1);
    ctx.setInputArray(0, &x);
    ctx.setInputArray(1, &y);
    ctx.setOutputArray(0, &z);

    sd::ops::biasadd op;


    for (int e = 0; e < 10000; e++) {
        auto timeStart = std::chrono::system_clock::now();

        op.execute(&ctx);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);
}
*/
/*
TEST_F(PlaygroundTests, test_s_1) {
    auto x0 = NDArrayFactory::create<float>('c', {32, 7, 7, 176});
    auto x1 = x0.ulike();
    auto x2 = x0.ulike();
    auto x3 = x0.ulike();
    auto x4 = x0.ulike();
    auto x5 = x0.ulike();

    auto y = NDArrayFactory::create<int >(3);
    auto z = NDArrayFactory::create<float>('c', {32, 7, 7, 1056});

    Context ctx(1);
    ctx.setInputArray(0, &x0);
    ctx.setInputArray(1, &x1);
    ctx.setInputArray(2, &x2);
    ctx.setInputArray(3, &x3);
    ctx.setInputArray(4, &x4);
    ctx.setInputArray(5, &x5);

    ctx.setInputArray(6, &y);
    ctx.setOutputArray(0, &z);
    ctx.setBArguments({true});

    std::vector<Nd4jLong> values;

    sd::ops::concat op;
    op.execute(&ctx);

    for (int e = 0; e < 1000; e++) {
        auto timeStart = std::chrono::system_clock::now();

        op.execute(&ctx);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }


    std::sort(values.begin(), values.end());

    nd4j_printf("Time: %lld us;\n", values[values.size() / 2]);
}
*/

/*
TEST_F(PlaygroundTests, test_s_1) {
    auto t = ::runLightBenchmarkSuit(true);
    delete[] t;
}

TEST_F(PlaygroundTests, test_s_2) {
    std::atomic<int> s;
    s = 0;
    auto func = PRAGMA_THREADS_FOR {
        s++;
    };

    samediff::Threads::parallel_for(func, 0, 8192, 1, 4);
    std::vector<Nd4jLong> values;

    for (int e = 0; e < 100000; e++) {
        s = 0;

        auto timeStart = std::chrono::system_clock::now();
        //samediff::Threads::parallel_for(func, 0, 8192, 1, 4);
        PRAGMA_OMP_PARALLEL_THREADS(4) {
            s++;
        }

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds> (timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    };
    std::sort(values.begin(), values.end());

    nd4j_printf("Time: %lld;\n", values[values.size() / 2]);
}
 */
/*
TEST_F(PlaygroundTests, test_s_4) {
    std::atomic<float> f;
    std::atomic<int> s;
    std::vector<Nd4jLong> valuesX, valuesY;
    int iterations = 1000;
    s = 0;
    auto func = PRAGMA_THREADS_FOR {
        s++;
    };

    samediff::Threads::parallel_for(func, 0, 8192, 1, 4);

    ////////

    auto x = NDArrayFactory::create<float>('c', {32, 3, 256, 256});
    auto z = NDArrayFactory::create<float>('c', {32, 3, 256, 256});
    x.linspace(1.0);

    auto xs0 = x.sizeAt(0);
    auto xs1 = x.sizeAt(1);
    auto xs2 = x.sizeAt(2);
    auto xs3 = x.sizeAt(3);

    auto buffer = x.bufferAsT<float>();
    auto zbuffer = z.bufferAsT<float>();

    for (int e = 0; e < iterations; e++) {
        auto timeStart = std::chrono::system_clock::now();
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int i = 0; i < xs0; i++) {
            for (int j = 0; j < xs1; j++) {
                auto thread_id = omp_get_thread_num();
                for (int k = 0; k < xs2; k++) {
                    for (int l = 0; l < xs3; l++) {
                        zbuffer[thread_id] += buffer[i * j + (k*l)] * 2.5f;
                    }
                }
            }
        }
        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
        valuesX.emplace_back(outerTime);
    }


    for (int e = 0; e < iterations; e++) {
        auto timeStart = std::chrono::system_clock::now();
        auto f2d = PRAGMA_THREADS_FOR_2D {
            for (auto i = start_x; i < stop_x; i++) {
                for (auto j = start_y; j < stop_y; j++) {

                    for (auto k = 0; k < xs2; k++) {
                        for (auto l = 0; l < xs3; l++) {
                            zbuffer[thread_id] += buffer[i * j + (k * l)] * 2.5f;
                        }
                    }
                }
            }
        };
        samediff::Threads::parallel_for(f2d, 0, xs0, 1, 0, xs1, 1);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
        valuesY.emplace_back(outerTime);
    }

    if (valuesX.size() > 0) {
        std::sort(valuesX.begin(), valuesX.end());
        nd4j_printf("OpenMP time: %lld; Min: %lld; Max: %lld;\n", valuesX[valuesX.size() / 2], valuesX[0], valuesX[valuesX.size() - 1]);
    }

    if (valuesY.size() > 0) {
        std::sort(valuesY.begin(), valuesY.end());
        nd4j_printf("Threads time: %lld; Min: %lld; Max: %lld;\n", valuesY[valuesY.size() / 2], valuesY[0], valuesY[valuesY.size() - 1]);
    }

    nd4j_printf("Sum: %f\n", z.sumNumber().e<float>(0));
}


TEST_F(PlaygroundTests, test_s_5) {
    auto x = NDArrayFactory::create<float>('c', {32, 1, 28, 28});

    std::vector<Nd4jLong> values;
    auto iterations = 100;

    auto startX = 0;
    auto stopX = x.sizeAt(0);
    auto incX = 1;
    auto startY = 0;
    auto stopY = x.sizeAt(1);
    auto incY = 1;
    auto numThreads = 4;

    // number of elements per loop
    auto delta_x = (stopX - startX);
    auto delta_y = (stopY - startY);

    // number of iterations per loop
    auto itersX = delta_x / incX;
    auto itersY = delta_y / incY;

    for (int e = 0; e < iterations; e++) {
        auto timeStart = std::chrono::system_clock::now();

        // picking best fit here
        auto splitLoop = samediff::ThreadsHelper::pickLoop2d(numThreads, itersX, itersY);
        auto span = samediff::Span2::build(splitLoop, 0, numThreads, startX, stopX, incX, startY, stopY, incY);

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    nd4j_printf("Calculations time: [Median: %lld; Min: %lld; Max: %lld;]\n", values[values.size() / 2], values[0], values[values.size()-1]);
}


TEST_F(PlaygroundTests, test_s_6) {
    auto x = NDArrayFactory::create<float>('c', {1024 * 1024 * 64});
    auto buffer = x.bufferAsT<float>();
    auto len = x.lengthOf();
    std::vector<Nd4jLong> values;
    auto iterations = 1000;

    for (int i = 0; i < iterations; i++) {
        auto timeStart = std::chrono::system_clock::now();

        // picking best fit here
        for (int e = 0; e < len; e++) {
            buffer[e] = (buffer[e] + 1.72f) * 3.17f - 0.0012f;
        }

        auto timeEnd = std::chrono::system_clock::now();
        auto outerTime = std::chrono::duration_cast<std::chrono::nanoseconds>(timeEnd - timeStart).count();
        values.emplace_back(outerTime);
    }

    std::sort(values.begin(), values.end());

    nd4j_printf("Calculations time: [Median: %lld; Min: %lld; Max: %lld;]\n", values[values.size() / 2], values[0], values[values.size()-1]);
}


TEST_F(PlaygroundTests, test_s_3) {
    std::atomic<int> s;
    s = 0;
    auto func = PRAGMA_THREADS_FOR {
        s++;
    };

    for (int e = 0; e < 10000; e++) {

        samediff::Threads::parallel_for(func, 0, 8192, 1, 4);
    }
}
 */

/*
TEST_F(PlaygroundTests, test_relubp_1) {
    auto x = NDArrayFactory::create<float>('c', {128, 64, 224, 224});
    auto y = x.ulike();
    auto z = x.ulike();
    RandomGenerator rng(119, 120);
    RandomLauncher::fillUniform(LaunchContext::defaultContext(), rng, &x, -1.0, 1.0);
    RandomLauncher::fillUniform(LaunchContext::defaultContext(), rng, &y, -1.0, 1.0);

    int iterations = 10;

    auto timeStart = std::chrono::system_clock::now();
    for (int e = 0; e < iterations; e++)
        ops::helpers::reluDerivative(LaunchContext::defaultContext(), &x, &y, &z);
    auto timeEnd = std::chrono::system_clock::now();

    auto outerTime = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();
    auto time = (Nd4jLong) outerTime / iterations;
    auto bw = (1000000L * (float) (x.lengthOf() * x.sizeOfT()) / time) / 1024 / 1024 / 1024;

    nd4j_printf("Time: %lld; BW: %f GB/s\n", time, bw);
}

//////////////////////////////////////////////////////////////////////
TEST_F(PlaygroundTests, my) {

    int bS=8, iD=32,iH=32,iW=32,  iC=128,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=2,dH=2,dW=2;
    int       oD,oH,oW;

    sd::ops::ConvolutionUtils::calcOutSizeDeconv3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, 0);

    printf("!!%i, %i, %i\n", oD,oH,oW);

    NDArray col('c', {bS, iC, kD, kH, kW, iD, iH, iW}, sd::DataType::DOUBLE);
    NDArray vol('c', {bS, iC, oD, oH, oW}, sd::DataType::DOUBLE);

    col = 3.77;
    vol = -10.33;

    auto variableSpace = new VariableSpace();
    auto block = new Context(1, variableSpace, false);  // not-in-place

    auto timeStart = std::chrono::system_clock::now();
    sd::ops::ConvolutionUtils::col2vol(*block, col, vol, sD, sH, sW, pD, pH, pW, dD, dH, dW);
    auto timeEnd = std::chrono::system_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    printf("time: %i \n", time);

    delete block;
    delete variableSpace;
}

TEST_F(PlaygroundTests, my) {

    int bS=32, iD=32,iH=64,iW=64,  iC=128,  kD=2,kH=2,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=2,dH=2,dW=2;
    int       oD,oH,oW;

    // sd::ops::ConvolutionUtils::calcOutSizeDeconv3D(oD, oH, oW, kD, kH, kW, sD, sH, sW, pD, pH, pW, dD, dH, dW, iD, iH, iW, 0);
    sd::ops::ConvolutionUtils::calcOutSizeDeconv2D(oH, oW, kH, kW, sH, sW, pH, pW,dH, dW, iH, iW, 0);

    printf("!!%i, %i, %i\n", oD,oH,oW);

    // NDArray col('c', {bS, iC, kD, kH, kW, iD, iH, iW}, sd::DataType::DOUBLE);
    // NDArray vol('c', {bS, iC, oD, oH, oW}, sd::DataType::DOUBLE);
    NDArray col('c', {bS, iC, kH, kW, iH, iW}, sd::DataType::DOUBLE);
    NDArray im('c', {bS, iC, oH, oW}, sd::DataType::DOUBLE);

    col = 3.77;
    // vol = -10.33;
    im = -10.33;

    auto variableSpace = new VariableSpace();
    auto block = new Context(1, variableSpace, false);  // not-in-place

    auto timeStart = std::chrono::system_clock::now();
    // sd::ops::ConvolutionUtils::col2vol(*block, col, vol, sD, sH, sW, pD, pH, pW, dD, dH, dW);
    sd::ops::helpers::col2im(*col.getContext(), col, im, sH, sW, pH, pW, iH, iW, dH, dW);
    auto timeEnd = std::chrono::system_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds> (timeEnd - timeStart).count();

    printf("time: %i \n", time);

    delete block;
    delete variableSpace;
}

TEST_F(PlaygroundTests, my) {

    int N = 100;
    int bS=16, iH=128,iW=128,  iC=32,oC=64,  kH=4,kW=4,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int        oH=128,oW=128;

    int paddingMode = 1;             // 1-SAME, 0-VALID;
    int dataFormat  = 1;             // 1-NHWC, 0-NCHW

    // NDArray input('c', {bS, iC, iH, iW}, sd::DataType::FLOAT32);
    // NDArray output('c', {bS, oC, oH, oW}, sd::DataType::FLOAT32);
    NDArray input('c', {bS, iH, iW, iC}, sd::DataType::FLOAT32);
    NDArray output('c', {bS, oH, oW, oC}, sd::DataType::FLOAT32);
    // NDArray weights('c', {kH, kW, iC, oC}, sd::DataType::FLOAT32);    // permute [kH, kW, iC, oC] -> [oC, iC, kH, kW]
    NDArray weights('c', {oC, iC, kH, kW}, sd::DataType::FLOAT32);
    NDArray bias('c', {oC}, sd::DataType::FLOAT32);

    input = 5.;
    weights = 3.;
    bias = 1.;

    sd::ops::conv2d op;
    auto err = op.execute({&input, &weights, &bias}, {&output}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});

    auto timeStart = std::chrono::system_clock::now();
    for (int i = 0; i < N; ++i)
        err = op.execute({&input, &weights, &bias}, {&output}, {kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto timeEnd = std::chrono::system_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds> ((timeEnd - timeStart) / N).count();

    printf("time: %i \n", time);
}


*/
