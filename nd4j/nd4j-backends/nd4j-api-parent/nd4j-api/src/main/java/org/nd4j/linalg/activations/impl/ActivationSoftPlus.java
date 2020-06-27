/* *****************************************************************************
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

package org.nd4j.linalg.activations.impl;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.BaseActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.gradient.SoftPlusBp;
import org.nd4j.linalg.api.ops.impl.transforms.strict.SoftPlus;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;

/**
 * f(x) = log(1+e^x)
 */
@EqualsAndHashCode(callSuper = false)
@Getter
public class ActivationSoftPlus extends BaseActivationFunction {

    @Override
    public INDArray getActivation(INDArray in, boolean training) {
        Nd4j.getExecutioner().execAndReturn(new SoftPlus(in));
        return in;
    }

    @Override
    public Pair<INDArray, INDArray> backprop(INDArray in, INDArray epsilon) {
        assertShape(in, epsilon);

        Nd4j.getExecutioner().execAndReturn(new SoftPlusBp(in, epsilon, in));

        return new Pair<>(in, null);
    }

    @Override
    public SDVariable defineActivation(@NonNull SameDiff sameDiff, @NonNull SDVariable input) {
        return sameDiff.nn.softplus(input);
    }

    @Override
    public String toString() {
        return "softplus";
    }
}
