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

#include "testlayers.h"
#include <helpers/HessenbergAndSchur.h>
// #include <helpers/biDiagonalUp.h>
// #include <helpers/hhSequence.h>
// #include <helpers/svd.h>
// #include <helpers/hhColPivQR.h>
// #include <array>
// #include <helpers/jacobiSVD.h>
// #include <ops/declarable/helpers/reverse.h>
// #include <ops/declarable/helpers/activations.h>
// #include <ops/declarable/helpers/rnn.h>
// #include <ops/declarable/helpers/sg_cb.h>
// #include <helpers/MmulHelper.h>
// #include <helpers/GradCheck.h>
// #include <ops/declarable/CustomOperations.h>
// #include <ops/declarable/helpers/lstmLayer.h>


using namespace sd;

class HelpersTests2 : public testing::Test {
public:

    HelpersTests2() {

        std::cout<<std::endl<<std::flush;
    }

};

#ifndef __CUDABLAS__

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_1) {


    NDArray x1('c', {1,4}, {14,17,3,1}, sd::DataType::DOUBLE);
    NDArray x2('c', {1,1}, {14}, sd::DataType::DOUBLE);
    NDArray expQ('c', {1,1}, {1}, sd::DataType::DOUBLE);

    ops::helpers::Hessenberg<double> hess1(x1);
    ASSERT_TRUE(hess1._H.isSameShape(&x1));
    ASSERT_TRUE(hess1._H.equalsTo(&x1));
    ASSERT_TRUE(hess1._Q.isSameShape(&expQ));
    ASSERT_TRUE(hess1._Q.equalsTo(&expQ));

    ops::helpers::Hessenberg<double> hess2(x2);
    ASSERT_TRUE(hess2._H.isSameShape(&x2));
    ASSERT_TRUE(hess2._H.equalsTo(&x2));
    ASSERT_TRUE(hess2._Q.isSameShape(&expQ));
    ASSERT_TRUE(hess2._Q.equalsTo(&expQ));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_2) {

    NDArray x('c', {2,2}, {1.5,-2,17,5}, sd::DataType::DOUBLE);
    NDArray expQ('c', {2,2}, {1,0,0,1}, sd::DataType::DOUBLE);

    ops::helpers::Hessenberg<double> hess(x);

    hess._H.printBuffer();

    ASSERT_TRUE(hess._H.isSameShape(&x));
    ASSERT_TRUE(hess._H.equalsTo(&x));

    ASSERT_TRUE(hess._Q.isSameShape(&expQ));
    ASSERT_TRUE(hess._Q.equalsTo(&expQ));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_3) {

    NDArray x('c', {3,3}, {33,24,-48,57,12.5,-3,1.1,10,-5.2}, sd::DataType::DOUBLE);
    NDArray expH('c', {3,3}, {33, -23.06939, -48.45414, -57.01061,  12.62845,  3.344058, 0, -9.655942, -5.328448}, sd::DataType::DOUBLE);
    NDArray expQ('c', {3,3}, {1,0,0,0, -0.99981, -0.019295, 0, -0.019295,0.99981}, sd::DataType::DOUBLE);

    ops::helpers::Hessenberg<double> hess(x);

    ASSERT_TRUE(hess._H.isSameShape(&expH));
    ASSERT_TRUE(hess._H.equalsTo(&expH));

    ASSERT_TRUE(hess._Q.isSameShape(&expQ));
    ASSERT_TRUE(hess._Q.equalsTo(&expQ));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_4) {

    NDArray x('c', {4,4}, {0.33 ,-7.25 ,1.71 ,6.20 ,1.34 ,5.38 ,-2.76 ,-8.51 ,7.59 ,3.44 ,2.24 ,-6.82 ,-1.15 ,4.80 ,-4.67 ,2.14}, sd::DataType::DOUBLE);
    NDArray expH('c', {4,4}, {0.33, 0.4961181,   3.51599,  9.017665, -7.792702,  4.190221,  6.500328,  5.438888, 0,  3.646734, 0.4641911, -7.635502, 0,0,  5.873535,  5.105588}, sd::DataType::DOUBLE);
    NDArray expQ('c', {4,4}, {1,0,0,0, 0,-0.171956, 0.336675, -0.925787, 0,-0.973988,0.0826795,  0.210976, 0, 0.147574, 0.937984,0.3137}, sd::DataType::DOUBLE);

    ops::helpers::Hessenberg<double> hess(x);

    ASSERT_TRUE(hess._H.isSameShape(&expH));
    ASSERT_TRUE(hess._H.equalsTo(&expH));

    ASSERT_TRUE(hess._Q.isSameShape(&expQ));
    ASSERT_TRUE(hess._Q.equalsTo(&expQ));
}

///////////////////////////////////////////////////////////////////
TEST_F(HelpersTests2, Hessenberg_5) {

    NDArray x('c', {10,10}, {6.9 ,4.8 ,9.5 ,3.1 ,6.5 ,5.8 ,-0.9 ,-7.3 ,-8.1 ,3.0 ,0.1 ,9.9 ,-3.2 ,6.4 ,6.2 ,-7.0 ,5.5 ,-2.2 ,-4.0 ,3.7 ,-3.6 ,9.0 ,-1.4 ,-2.4 ,1.7 ,
                            -6.1 ,-4.2 ,-2.5 ,-5.6 ,-0.4 ,0.4 ,9.1 ,-2.1 ,-5.4 ,7.3 ,3.6 ,-1.7 ,-5.7 ,-8.0 ,8.8 ,-3.0 ,-0.5 ,1.1 ,10.0 ,8.0 ,0.8 ,1.0 ,7.5 ,3.5 ,-1.8 ,
                            0.3 ,-0.6 ,-6.3 ,-4.5 ,-1.1 ,1.8 ,0.6 ,9.6 ,9.2 ,9.7 ,-2.6 ,4.3 ,-3.4 ,0.0 ,-6.7 ,5.0 ,10.5 ,1.5 ,-7.8 ,-4.1 ,-5.3 ,-5.0 ,2.0 ,-4.4 ,-8.4 ,
                            6.0 ,-9.4 ,-4.8 ,8.2 ,7.8 ,5.2 ,-9.5 ,-3.9 ,0.2 ,6.8 ,5.7 ,-8.5 ,-1.9 ,-0.3 ,7.4 ,-8.7 ,7.2 ,1.3 ,6.3 ,-3.7 ,3.9 ,3.3 ,-6.0 ,-9.1 ,5.9}, sd::DataType::DOUBLE);
    NDArray expH('c', {10,10}, {6.9,  6.125208, -8.070945,  7.219828, -9.363308,  2.181236,  5.995414,  3.892612,  4.982657, -2.088574,-12.6412,  1.212547, -6.449684,  5.162879, 0.4341714, -5.278079, -2.624011,  -2.03615,  11.39619, -3.034842,
                                0, -12.71931,   10.1146,  6.494434, -1.062934,  5.668906, -4.672953, -9.319893, -2.023392,  6.090341,0,0, 7.800521,  -1.46286,  1.484626, -10.58252, -3.492978,   2.42187,  5.470045,  1.877265,
                                0,0,0, 14.78259,-0.3147726,  -5.74874, -0.377823,  3.310056,  2.242614, -5.111574,0,0,0,0, -9.709131,  3.885072,  6.762626,  4.509144,  2.390195, -4.991013,
                                0,0,0,0,0,  8.126269, -12.32529,  9.030151,  1.390931, 0.8634045,0,0,0,0,0,0, -12.99477,  9.574299,-0.3098022,  4.910835,0,0,0,0,0,0,0,  14.75256,  18.95723, -5.054717,0,0,0,0,0,0,0,0, -4.577715, -5.440827,}, sd::DataType::DOUBLE);
    NDArray expQ('c', {10,10}, {1,0,0,0,0,0,0,0,0,0,0,-0.0079106,-0.38175,-0.39287,-0.26002,-0.44102,-0.071516,0.12118,0.64392,0.057562,
                                0,0.28478,0.0058784,0.3837,-0.47888,0.39477,0.0036847,-0.24678,0.3229,0.47042,0,-0.031643,-0.61277,0.087648,0.12014,0.47648,-0.5288,0.060599,0.021434,-0.30102,
                                0,0.23732,-0.17801,-0.31809,-0.31267,0.27595,0.30134,0.64555,-0.33392,0.13363,0,-0.023732,-0.40236,0.43089,-0.38692,-0.5178,-0.03957,-0.081667,-0.47515,-0.0077949,
                                0,0.20568,-0.0169,0.36962,0.49669,-0.22475,-0.22199,0.50075,0.10454,0.46112,0,0.41926,0.30243,-0.3714,-0.16795,-0.12969,-0.67572,-0.1205,-0.26047,0.10407,
                                0,-0.41135,-0.28357,-0.33858,0.18836,0.083822,-0.0068213,-0.30161,-0.24956,0.66327,0,0.68823,-0.33616,-0.12129,0.36163,-0.063256,0.34198,-0.37564,-0.048196,-0.058948}, sd::DataType::DOUBLE);

    ops::helpers::Hessenberg<double> hess(x);

    ASSERT_TRUE(hess._H.isSameShape(&expH));
    ASSERT_TRUE(hess._H.equalsTo(&expH));

    ASSERT_TRUE(hess._Q.isSameShape(&expQ));
    ASSERT_TRUE(hess._Q.equalsTo(&expQ));
}

#endif