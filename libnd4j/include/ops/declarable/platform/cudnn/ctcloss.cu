/*******************************************************************************
 *
 * Copyright (c) 2021 Konduit K.K.
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
 // @author AbdelRauf
 //

#include "cudnnUtils.h"
#include <vector>


namespace sd   {
namespace ops     {
namespace platforms {



    template<typename Op, typename ...Args>
    void callCudnnIfNoErr(cudnnStatus_t &err, Op op, Args&&... args){
        if(err==CUDNN_STATUS_SUCCESS){
            err = op(std::forward<Args>(args)...);
            if(err){
                nd4j_printf("Cudnn error code %s\n",cudnnGetErrorString(err));
            }
        }
    }

    template <typename T>
    const T* bufferInHost( const NDArray *array)  {
        array->syncToHost();
        return reinterpret_cast<const T*>(array->buffer());
    }

    std::vector<int> getConcatTargets(const NDArray &targetLabels, const NDArray &targetLabelLengths){
                //concatenate target labels
                const int32_t *tlabels = bufferInHost<int32_t>(&targetLabels);
                const int32_t *tlens =bufferInHost<int32_t>(&targetLabelLengths);
                int32_t nextOffset = targetLabels.strideAt(0);
                int32_t elStride = targetLabels.strideAt(1);
                int32_t batchCount = targetLabelLengths.lengthOf();
                std::vector<int> labels;
                labels.resize(targetLabels.lengthOf());
                int j=0;
                if(targetLabels.ews()){
                    for(int i=0; i<batchCount;i++){
                        int count = tlens[i];
                        for( int k=0;k<count;k++){
                            labels[j] = tlabels[k];
                            j++;
                        }
                        tlabels+=nextOffset;
                    }
                }else{
                    for(int i=0; i<batchCount;i++){
                        int count = tlens[i];
                        for( int k=0;k<count;k++){
                            labels[j] = tlabels[k*elStride];
                            j++;
                        }
                        tlabels+=nextOffset;
                    }
                }
                return labels;
    }


    PLATFORM_IMPL(ctc_loss, ENGINE_CUDA) {
        auto targetLabels = INPUT_VARIABLE(0);
        auto logitInput = INPUT_VARIABLE(1);
        auto targetLabelLengths = INPUT_VARIABLE(2);
        auto logitInputLengths = INPUT_VARIABLE(3); 
        auto outputLosses = OUTPUT_VARIABLE(0);
        int blankIndex = INT_ARG(0);
        auto context = block.launchContext();
        auto handle = reinterpret_cast<cudnnHandle_t *>(context->getCuDnnHandle());

        cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
        cudnnSetStream(*handle, *context->getCudaStream());

        //in Cudnn inputs are probabilities
        //in Cudnn Batch is in the middle dimension
        auto probs = logitInput->ulike();
        logitInput->applyTransform(sd::transform::Exp, probs);
        probs.permutei({1,0,2}); 
        const int dims[] = {(int)probs.sizeAt(0), (int)probs.sizeAt(1), (int)probs.sizeAt(2)};
        const int strides[] = {(int)probs.strideAt(0), (int)probs.strideAt(1), (int)probs.strideAt(2)};

        //in Cudnn targets are concantenated instead of batched as matrix
        auto labels = getConcatTargets(*targetLabels, *targetLabelLengths);
        const int32_t * ldata= labels.data();

        cudnnCTCLossDescriptor_t  ctcLossDesc;
        cudnnTensorDescriptor_t probsDesc;
        callCudnnIfNoErr(err,cudnnCreateCTCLossDescriptor,&ctcLossDesc);
        callCudnnIfNoErr(err,cudnnCreateTensorDescriptor,&probsDesc);
        callCudnnIfNoErr(err, cudnnSetTensorNdDescriptor,probsDesc, cudnnDataType(logitInput->dataType()), logitInput->rankOf() , dims, strides);


        size_t tempWorkSpaceSize=0;
        callCudnnIfNoErr(err,cudnnGetCTCLossWorkspaceSize, *handle,  probsDesc, nullptr,
            ldata,
            bufferInHost<int32_t>(targetLabelLengths),
            bufferInHost<int32_t>(logitInputLengths),
            CUDNN_CTC_LOSS_ALGO_DETERMINISTIC,
            ctcLossDesc, &tempWorkSpaceSize);

        // Allocate temp tempWorkspace buffer
        void *tempWorkSpace = nullptr;
        cudaMalloc(&tempWorkSpace, tempWorkSpaceSize);

        NDArray::prepareSpecialUse({outputLosses}, {logitInput});
        callCudnnIfNoErr(err, cudnnCTCLoss,*handle,
            probsDesc,
            probs.specialBuffer(),
            ldata,
            bufferInHost<int32_t>(targetLabelLengths),
            bufferInHost<int32_t>(logitInputLengths),
            outputLosses->specialBuffer(),
            nullptr,
            nullptr,
            CUDNN_CTC_LOSS_ALGO_DETERMINISTIC,
            ctcLossDesc,
            tempWorkSpace,
            tempWorkSpaceSize);

        NDArray::registerSpecialUse({outputLosses}, {logitInput});

        cudaFree(tempWorkSpace);
        callCudnnIfNoErr(err, cudnnDestroyTensorDescriptor,probsDesc);
        callCudnnIfNoErr(err, cudnnDestroyCTCLossDescriptor,ctcLossDesc);

        if(err!=CUDNN_STATUS_SUCCESS) throw sd::cuda_exception::build("ctc_loss CUDNN call failure ", err);
        return Status::OK();
    }

    template<typename T>
    bool checkLabelLength(const NDArray &labelLengthArr){
            //check label lengthes
            auto lenBatch = labelLengthArr.lengthOf(); 
            for(int i=0; i < lenBatch; i++){
                // The labelLengths is greater than 256.
                if(labelLengthArr.e<int32_t>(i)>256) return false;
            }
            return true;
    }

    PLATFORM_CHECK(ctc_loss, ENGINE_CUDA) {
        auto targetLabels = INPUT_VARIABLE(0);
        auto logitInput = INPUT_VARIABLE(1);
        auto targetLabelLengths = INPUT_VARIABLE(2);
        auto logitInputLengths = INPUT_VARIABLE(3); 
        auto outputLosses = OUTPUT_VARIABLE(0);
        int blankIndex = INT_ARG(0);

        auto dTypeInput = logitInput->dataType();
        auto intType = targetLabelLengths->dataType();
        auto dTypeOutput = outputLosses->dataType();

        bool is_supported = blankIndex==0 && intType == DataType::INT32  && dTypeInput == DataType::FLOAT32;
        is_supported = is_supported && outputLosses->ews() && targetLabelLengths->ews() && targetLabels->ews() && logitInputLengths->ews();
        if(is_supported){
            is_supported = is_supported && checkLabelLength<int32_t>(*targetLabelLengths);
        }
        return  is_supported; 
    }

    PLATFORM_IMPL(ctc_loss_grad, ENGINE_CUDA) {
        return Status::OK();
    }

    PLATFORM_CHECK(ctc_loss_grad, ENGINE_CUDA) {
        return false;
    } 

}
}
}