/*******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
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
 //  @author Oleg Semeniv <oleg.semeniv@gmail.com>
 //
 //

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>
#include <helpers/MKLDNNStream.h>
#include "mkldnnUtils.h"

using namespace dnnl;

namespace sd {
    namespace ops {
        namespace platforms {


            //////////////////////////////////////////////////////////////////////
            static void softmaxMKLDNN(const NDArray* x, NDArray* z, const int axis) {

                const auto xRank = x->rankOf();
                dnnl::memory::dims xShape, zShape;

                mkldnnUtils::getDims(x, xRank, xShape);
                mkldnnUtils::getDims(z, xRank, zShape);


                dnnl::memory::format_tag format = mkldnnUtils::getFormat(xRank);
                // optimized cases
                if (2 == xRank && 0 == axis) {
                    format = dnnl::memory::format_tag::ba;
                }
                else if (4 == xRank && 1 == axis && (x->sizeAt(2) * x->sizeAt(3)) > 1) {
                    format = dnnl::memory::format_tag::acdb;
                }

                dnnl::memory::data_type xType = dnnl::memory::data_type::f32;

                dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xShape, xType, format);
                dnnl::memory::desc x_user_md = dnnl::memory::desc(xShape, xType, format);
                mkldnnUtils::setBlockStrides(x, x_user_md);

                // z
                dnnl::memory::desc z_mkl_md = dnnl::memory::desc(zShape, xType, format);
                dnnl::memory::desc z_user_md = dnnl::memory::desc(zShape, xType, format);
                mkldnnUtils::setBlockStrides(z, z_user_md);

                auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

                // Create attributes (to handle alpha and beta if necessary)
                dnnl::primitive_attr attr; // it is empty since we have usual values for alpha (=1) and beta (=0)

                // operation primitive description
                dnnl::softmax_forward::desc op_desc(dnnl::prop_kind::forward_inference, x_mkl_md, axis);

                dnnl::softmax_forward::primitive_desc op_prim_desc(op_desc, attr, engine);

                // arguments (memory buffers) necessary for calculations
                std::unordered_map<int, dnnl::memory> args;

                dnnl::stream stream(engine);

                // provide memory buffers and check whether reorder is required

                // input
                mkldnnUtils::loadDataToMklStream(x, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

                // z
                auto z_user_mem = dnnl::memory(z_user_md, engine, z->getBuffer());
                const bool zReorder = op_prim_desc.dst_desc() != z_user_mem.get_desc();
                auto z_mkl_mem = zReorder ? dnnl::memory(op_prim_desc.dst_desc(), engine) : z_user_mem;
                args[DNNL_ARG_DST] = z_mkl_mem;

                // run calculations
                dnnl::softmax_forward(op_prim_desc).execute(stream, args);

                // reorder outputs if necessary
                if (zReorder)
                    dnnl::reorder(z_mkl_mem, z_user_mem).execute(stream, z_mkl_mem, z_user_mem);

                stream.wait();
            }


            PLATFORM_IMPL(softmax, ENGINE_CPU) {

                auto input = INPUT_VARIABLE(0);
                auto output = OUTPUT_VARIABLE(0);

                const int rank = input->rankOf();
                int dim = block.numI() > 0 ? INT_ARG(0) : rank - 1;

                if (dim < 0) {
                    dim += rank;
                }

                REQUIRE_TRUE(dim < rank && dim >= 0, 0, "SOFTMAX_MKLDNN OP: the value of input integer parameter (dimension) must be less than input array rank %i, but got dimension = %i instead !", rank, dim);

                REQUIRE_TRUE(rank <= 6, 0, "SOFTMAX_MKLDNN OP: the rank of input must be less or qual 6, but got rank = %i instead !", rank);

                // mkldnnSoftMax
                softmaxMKLDNN(input, output, dim);

                return Status::OK();
            }

            PLATFORM_CHECK(softmax, ENGINE_CPU) {

                auto x = INPUT_VARIABLE(0);
                auto z = OUTPUT_VARIABLE(0);

                const DataType xType = x->dataType();
                const DataType zType = z->dataType();

                const int xRank = x->rankOf();
                bool bSupportedRanks = (xRank > 2 && xRank < 7);
                /*
                Source     Destination
                f32 	    f32
                */
                return  !x->isEmpty() && block.isUseMKLDNN() && bSupportedRanks && (xType == DataType::FLOAT32 && zType == DataType::FLOAT32);

            }

            //////////////////////////////////////////////////////////////////////
            static void softmaxBpMKLDNN(const NDArray* x, const NDArray* dLdz, NDArray* dLdx, const int axis) {

                const auto xRank = x->rankOf();
                const auto dLdzRank = dLdz->rankOf();

                dnnl::memory::dims xShape, dLdxShape, dLdzShape;

                mkldnnUtils::getDims(x, xRank, xShape);
                mkldnnUtils::getDims(dLdx, xRank, dLdxShape);
                mkldnnUtils::getDims(dLdz, dLdzRank, dLdzShape);

                dnnl::memory::format_tag format = mkldnnUtils::getFormat(xRank);

                // x
                dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xShape, dnnl::memory::data_type::f32, format);
                dnnl::memory::desc x_user_md = dnnl::memory::desc(xShape, dnnl::memory::data_type::f32, format);
                mkldnnUtils::setBlockStrides(x, x_user_md);

                // dLdx
                dnnl::memory::desc dLdx_mkl_md = dnnl::memory::desc(dLdxShape, dnnl::memory::data_type::f32, format);
                dnnl::memory::desc dLdx_user_md = dnnl::memory::desc(dLdxShape, dnnl::memory::data_type::f32, format);
                mkldnnUtils::setBlockStrides(dLdx, dLdx_user_md);
                // todo if mkl does not support broadcast we can remove this
                format = mkldnnUtils::getFormat(dLdzRank);

                // dLdz
                dnnl::memory::desc dLdz_mkl_md = dnnl::memory::desc(dLdzShape, dnnl::memory::data_type::f32, format);
                dnnl::memory::desc dLdz_user_md = dnnl::memory::desc(dLdzShape, dnnl::memory::data_type::f32, format);
                mkldnnUtils::setBlockStrides(dLdz, dLdz_user_md);

                auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

                // operation primitive description
                // forward description
                dnnl::softmax_forward::desc op_ff_desc(dnnl::prop_kind::forward_inference, x_mkl_md, axis);
                dnnl::softmax_forward::primitive_desc op_ff_prim_desc(op_ff_desc, engine);

                // backward description
                dnnl::softmax_backward::desc op_bp_desc(dLdz_mkl_md, dLdx_mkl_md, axis);
                dnnl::softmax_backward::primitive_desc op_bp_prim_desc(op_bp_desc, engine, op_ff_prim_desc);

                // arguments (memory buffers) necessary for calculations
                std::unordered_map<int, dnnl::memory> argsbp, argsff;

                dnnl::stream stream(engine);

                // provide memory buffers and check whether reorder is required for forward
                // input
                mkldnnUtils::loadDataToMklStream(x, engine, stream, x_user_md, op_ff_prim_desc.src_desc(), argsff[DNNL_ARG_SRC]);

                // dLdx
                auto dLdx_user_mem = dnnl::memory(dLdx_user_md, engine, dLdx->getBuffer());
                const bool dLdxReorder = op_ff_prim_desc.dst_desc() != dLdx_user_mem.get_desc();
                auto dLdx_mkl_mem = dLdxReorder ? dnnl::memory(op_ff_prim_desc.dst_desc(), engine) : dLdx_user_mem;
                argsff[DNNL_ARG_DST] = dLdx_mkl_mem;

                // check and arg set for backprob
                argsbp[DNNL_ARG_DIFF_SRC] = dLdx_mkl_mem;
                argsbp[DNNL_ARG_DST] = dLdx_mkl_mem;
                // dLdz
                mkldnnUtils::loadDataToMklStream(dLdz, engine, stream, dLdz_user_md, op_bp_prim_desc.diff_dst_desc(), argsbp[DNNL_ARG_DIFF_DST]);

                // run calculations forward
                dnnl::softmax_forward(op_ff_prim_desc).execute(stream, argsff);

                // run calculations backward
                dnnl::softmax_backward(op_bp_prim_desc).execute(stream, argsbp);

                // reorder outputs if necessary
                if (dLdxReorder)
                    dnnl::reorder(dLdx_mkl_mem, dLdx_user_mem).execute(stream, dLdx_mkl_mem, dLdx_user_mem);

                stream.wait();
            }


            PLATFORM_IMPL(softmax_bp, ENGINE_CPU) {

                auto input = INPUT_VARIABLE(0);
                auto dLdz = INPUT_VARIABLE(1);
                auto dLdx = OUTPUT_VARIABLE(0);

                const int rank = input->rankOf();
                const int dLdzRank = dLdz->rankOf();
                int dim = block.numI() > 0 ? INT_ARG(0) : rank - 1;

                if (dim < 0) {
                    dim += rank;
                }

                REQUIRE_TRUE(dim < rank && dim >= 0, 0, "SOFTMAX_MKLDNN_BP OP: the value of input integer parameter (dimension) must be less than input array rank %i, but got dimension = %i instead !", rank, dim);

                REQUIRE_TRUE(rank <= 6 && dLdzRank <= 6, 0, "SOFTMAX_MKLDNN_BP OP: the rank of input and dLdz must be less or qual 6, but got input rank = %i and dLdz rank rank = %i instead !", rank, dLdzRank);

                // mkldnnSoftMax
                softmaxBpMKLDNN(input, dLdz, dLdx, dim);

                return Status::OK();
            }

            PLATFORM_CHECK(softmax_bp, ENGINE_CPU) {

                auto x = INPUT_VARIABLE(0);
                auto dLdz = INPUT_VARIABLE(1);
                auto dLdx = OUTPUT_VARIABLE(0);

                const DataType xType = x->dataType();
                const DataType dLdzType = dLdz->dataType();
                const DataType dLdxType = dLdx->dataType();

                const int xRank = x->rankOf();
                const int dLdzRank = dLdz->rankOf();

                bool bSupportedRanks = xRank < 7 && dLdzRank == xRank && (!x->isEmpty() && !dLdz->isEmpty());

                if (bSupportedRanks) {
                    for (int i = 0; i < xRank; i++) {
                        if (x->sizeAt(i) != dLdz->sizeAt(i)) {
                            bSupportedRanks = false;
                            break;
                        }
                    }
                }

                //Source     Destination
                //f32 	    f32
                return block.isUseMKLDNN() && bSupportedRanks && (xType == DataType::FLOAT32 && dLdzType == DataType::FLOAT32 && dLdxType == DataType::FLOAT32);
            }

        }
    }
}
