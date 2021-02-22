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

package org.nd4j.linalg.api.ops.impl.image;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;

import java.util.Collections;
import java.util.List;

/**
 * Non max suppression with overlaps
 *
 * @author raver119@gmail.com
 */
public class NonMaxSuppressionWithOverlaps extends DynamicCustomOp {

    public NonMaxSuppressionWithOverlaps() {}

    public NonMaxSuppressionWithOverlaps(SameDiff sameDiff,
                                         @NonNull SDVariable boxes,
                                         @NonNull SDVariable scores,
                                         @NonNull SDVariable maxOutSize,
                                         @NonNull SDVariable iouThreshold,
                                         @NonNull SDVariable scoreThreshold) {
        super(null, sameDiff, new SDVariable[]{boxes, scores, maxOutSize, iouThreshold, scoreThreshold}, false);
    }

    public NonMaxSuppressionWithOverlaps(SameDiff sameDiff,
                                         SDVariable boxes,
                                         SDVariable scores,
                                         int maxOutSize,
                                         double iouThreshold,
                                         double scoreThreshold) {
        super(null, sameDiff, new SDVariable[]{boxes, scores}, false);
        addIArgument(maxOutSize);
        addTArgument(iouThreshold, scoreThreshold);
    }

    public NonMaxSuppressionWithOverlaps(INDArray boxes, INDArray scores,
                                         int maxOutSize,
                                         double iouThreshold,
                                         double scoreThreshold) {
        addInputArgument(boxes,scores);
        addIArgument(maxOutSize);
        addTArgument(iouThreshold, scoreThreshold);
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx name found for shape " + opName());
    }

    @Override
    public String tensorflowName() {
        return "NonMaxSuppressionWithOverlaps";
    }


    @Override
    public String opName() {
        return "non_max_suppression_overlaps";
    }

    @Override
    public Op.Type opType() {
        return Op.Type.CUSTOM;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        return Collections.singletonList(sameDiff.zerosLike(arg()));
    }

    @Override
    public List<DataType> calculateOutputDataTypes(List<DataType> inputDataTypes){
        //Always 1D integer tensor (indices)
        return Collections.singletonList(DataType.INT);
    }
}