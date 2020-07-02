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

package org.deeplearning4j.nn.conf.layers.misc;

import java.util.Map;
import lombok.Data;
import lombok.NonNull;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.params.FrozenLayerWithBackpropParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.shade.jackson.annotation.JsonProperty;

import java.util.Collection;
import java.util.List;

/**
 * Frozen layer freezes parameters of the layer it wraps, but allows the backpropagation to continue.
 * 
 * @author Ugljesa Jovanovic (jovanovic.ugljesa@gmail.com) on 06/05/2018.
 * @see FrozenLayer
 */
@Data
public class FrozenLayerWithBackprop extends BaseWrapperLayer {

    public FrozenLayerWithBackprop(@JsonProperty("layer") Layer layer) {
        super(layer);
    }

    public NeuralNetConfiguration getInnerConf(NeuralNetConfiguration conf) {
        NeuralNetConfiguration nnc = conf.clone();
        nnc.setLayer(underlying);
        return nnc;
    }

    @Override
    public Layer clone() {
        FrozenLayerWithBackprop l = (FrozenLayerWithBackprop) super.clone();
        l.underlying = underlying.clone();
        return l;
    }

    @Override
    public org.deeplearning4j.nn.api.Layer instantiate(NeuralNetConfiguration conf,
                                                       Collection<TrainingListener> trainingListeners, int layerIndex, INDArray layerParamsView,
                                                       boolean initializeParams, DataType networkDataType) {

        //Need to be able to instantiate a layer, from a config - for JSON -> net type situations
        org.deeplearning4j.nn.api.Layer underlying = getUnderlying().instantiate(getInnerConf(conf), trainingListeners,
                        layerIndex, layerParamsView, initializeParams, networkDataType);

        NeuralNetConfiguration nncUnderlying = underlying.conf();

        if (nncUnderlying.variables() != null) {
            List<String> vars = nncUnderlying.variables(true);
            nncUnderlying.clearVariables();
            conf.clearVariables();
            for (String s : vars) {
                conf.variables(false).add(s);
                nncUnderlying.variables(false).add(s);
            }
        }

        return new org.deeplearning4j.nn.layers.FrozenLayerWithBackprop(underlying);
    }

    /**
     * Will freeze any params passed to it.
     *  @param sameDiff SameDiff instance
     * @param layerInput Input to the layer
     * @param mask Optional, maybe null. Mask to apply if supported
     * @param paramTable Parameter table - keys and shapes as defined in the layer implementation class.
     */
    @Override
    public SDVariable defineLayer(@NonNull SameDiff sameDiff, @NonNull SDVariable layerInput,
            SDVariable mask, @NonNull Map<String, SDVariable> paramTable) {
        for(SDVariable variable : paramTable.values()){
            variable.convertToConstant();
        }
        return defineUnderlying(sameDiff, layerInput, paramTable, mask);
    }

    @Override
    public ParamInitializer initializer() {
        return FrozenLayerWithBackpropParamInitializer.getInstance();
    }

    @Override
    public List<Regularization> getRegularizationByParam(String paramName){
        //No regularization for frozen layers
        return null;
    }

    @Override
    public boolean isPretrainParam(String paramName) {
        return false;
    }

    @Override
    public IUpdater getUpdaterByParam(String paramName) {
        return null;
    }

    @Override
    public void setLayerName(String layerName) {
        super.setLayerName(layerName);
        underlying.setLayerName(layerName);
    }

    @Override
    public void setConstraints(List<LayerConstraint> constraints) {
        this.constraints = constraints;
        this.underlying.setConstraints(constraints);
    }
}
