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

package org.nd4j.linalg.api.ops.impl.transforms.same;

import lombok.NoArgsConstructor;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformSameOp;

import java.util.Arrays;
import java.util.List;

/**
 * Rounding function
 *
 * @author Adam Gibson
 */
@NoArgsConstructor
public class Round extends BaseTransformSameOp {

    public Round(SameDiff sameDiff, SDVariable i_v) {
        this(sameDiff, i_v, false);
    }

    public Round(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public Round(INDArray x, INDArray z) {
        super(x, z);
    }

    public Round(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 4;
    }

    @Override
    public String opName() {
        return "round";
    }

    @Override
    public String onnxName() {
        return "Round";
    }

    @Override
    public String tensorflowName() {
        return "Round";
    }



    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Arrays.asList(sameDiff.zerosLike(arg()));
    }
}
