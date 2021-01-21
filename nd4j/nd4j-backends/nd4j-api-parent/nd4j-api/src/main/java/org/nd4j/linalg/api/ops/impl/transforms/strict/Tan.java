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

package org.nd4j.linalg.api.ops.impl.transforms.strict;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseTransformStrictOp;

import java.util.Collections;
import java.util.List;

/**
 * Tanh elementwise function
 *
 * @author raver119@gmail.com
 */
public class Tan extends BaseTransformStrictOp {
    public Tan(SameDiff sameDiff, SDVariable i_v, boolean inPlace) {
        super(sameDiff, i_v, inPlace);
    }

    public Tan(SameDiff sameDiff, SDVariable i_v) {
        this(sameDiff, i_v, false);
    }

    public Tan() {
    }

    public Tan(INDArray x, INDArray z) {
        super(x, z);
    }

    public Tan(INDArray x) {
        super(x);
    }

    @Override
    public int opNum() {
        return 41;
    }

    @Override
    public String opName() {
        return "tan";
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "Tan";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        //d(tan(x))/dx = (sec(x))^2 = 1 / (cos(x))^2
        SDVariable cosx = sameDiff.math.cos(arg());
        SDVariable cosSqx = sameDiff.math().square(cosx);
        return Collections.singletonList(i_v.get(0).div(cosSqx));
    }
}
