/*
 * ******************************************************************************
 *  * Copyright (c) 2020 Konduit K.K.
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.util;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.regularization.Regularization;

/**
 * Utilities for use in {@link org.deeplearning4j.nn.graph.ComputationGraph#toSameDiff(SameDiff, Map, boolean, boolean)} and {@link org.deeplearning4j.nn.multilayer.MultiLayerNetwork#toSameDiff(SameDiff, InputType, boolean, boolean)}.
 */
@Slf4j
public class ToSameDiffUtils {


    /**
     * Get the updater for a network.  If updaters aren't the same on all layers, throws an exception or returns null depending on skipErrors.
     * @param layers The layers of the network.
     * @param skipErrors If true, returns null if updaters aren't the same for all layers.  Otherwise, throws an error.
     */
    public static IUpdater getUpdater(Layer[] layers, boolean skipErrors){
        IUpdater iUpdater = null;
        for(Layer l : layers) {
            org.deeplearning4j.nn.conf.layers.Layer conf = l.conf().getLayer();
            if (conf instanceof BaseLayer) {
                IUpdater u = ((BaseLayer) conf).getIUpdater();
                if (iUpdater == null) {
                    iUpdater = u;
                } else {
                    if (u != null && u != iUpdater) {
                        if (skipErrors) {
                            iUpdater = null;
                            log.warn("Ignoring updater config: Can not convert to SameDiff with different IUpdaters. Expected {}, but was {} for {}", iUpdater, u, conf);
                            break;
                        } else {
                            throw new IllegalStateException(
                                    "Can not convert to SameDiff with different IUpdaters.  Ensure all layers have the same updater.  Expected "
                                            + iUpdater + ", but was " + u + " different for " + conf);
                        }
                    }
                }

                u = ((BaseLayer) conf).getBiasUpdater();
                if (iUpdater == null) {
                    iUpdater = u;
                } else {
                    if (u != null && u != iUpdater) {
                        if (skipErrors) {
                            iUpdater = null;
                            log.warn("Ignoring updater config: Can not convert to SameDiff when layers have different IUpdaters. Expected {}, but was {} for {}", iUpdater, u, conf);
                            break;
                        } else {
                            throw new IllegalStateException(
                                    "Can not convert to SameDiff with different IUpdaters.  Ensure all layers have the same updater.  Expected "
                                            + iUpdater + ", but was " + u + " for " + conf);
                        }
                    }
                }
            }
        }
        return iUpdater;
    }

    /**
     * Get the regularizations of a network.  If regularizations aren't the same on all layers, throws an exception or returns null depending on skipErrors.
     * @param layers The layers of the network.
     * @param skipErrors If true, returns null if regularizations aren't the same for all layers.  Otherwise, throws an error.
     */
    public static List<Regularization> getRegularizations(Layer[] layers, boolean skipErrors){
        List<Regularization> regularizations = null;

        for(Layer l : layers){
            org.deeplearning4j.nn.conf.layers.Layer conf = l.conf().getLayer();
            if(conf instanceof BaseLayer){
                if(regularizations == null){
                    regularizations = ((BaseLayer) conf).getRegularization();
                } else {
                    if(((BaseLayer) conf).getRegularization() != regularizations) {
                        if(skipErrors){
                            regularizations = null;
                            log.warn("Ignoring regularization config: Can not convert to SameDiff when layers have different regularizations. Expected {}, but was {} for {}",
                                    regularizations, ((BaseLayer) conf).getRegularization(), conf);
                            break;
                        } else {
                            throw new IllegalStateException(
                                    "Can not convert to SameDiff with different regularizations.  Ensure all layers have the same regularizations, and that bias and weight regularizations are the same.  "
                                            + "Expected " + regularizations + ", but was " + ((BaseLayer) conf)
                                            .getRegularization() + " for " + conf);
                        }
                    }
                }

                if(regularizations == null){
                    regularizations = ((BaseLayer) conf).getRegularizationBias();
                } else {
                    if(((BaseLayer) conf).getRegularizationBias() != regularizations) {
                        if(skipErrors){
                            regularizations = null;
                            log.warn("Ignoring regularization config: Can not convert to SameDiff when layers have different regularizations. Expected {}, but was {} for {}",
                                    regularizations, ((BaseLayer) conf).getRegularization(), conf);
                            break;
                        } else {
                            throw new IllegalStateException(
                                    "Can not convert to SameDiff with different regularizations.  Ensure all layers have the same regularizations, and that bias and weight regularizations are the same.  "
                                            + "Expected " + regularizations + ", but was " + ((BaseLayer) conf)
                                            .getRegularizationBias() + " for bias in " + conf);
                        }
                    }
                }
            }
        }
        return regularizations;
    }

    /**
     * Define the parameters of a layer, transforming them if necessary using {@link org.deeplearning4j.nn.conf.layers.Layer#transformParamsForSameDiff(Map)}.
     *
     * @param sameDiff The SameDiff to define the parameters in.
     * @param layer The layer whose parameters we are defining.
     * @param useView Whether to use the param view directly (if true) or dup it.
     * @return The SDVariable parameters of the layer.
     */
    public static Map<String, SDVariable> defineParams(SameDiff sameDiff, Layer layer, boolean useView){
        Map<String, INDArray> params = new HashMap<>(layer.paramTable(false));
        layer.conf().getLayer().transformParamsForSameDiff(params);
        return defineTransformedParams(sameDiff, params, (int) layer.numParams(), useView);
    }


    /**
     * Define the parameters of a vertex, transforming them if necessary using {@link GraphVertex#transformParamsForSameDiff(Map)}.
     *
     * @param sameDiff The SameDiff to define the parameters in.
     * @param vertex The vertex whose parameters we are defining.
     * @param useView Whether to use the param view directly (if true) or dup it.
     * @return The SDVariable parameters of the vertex.
     */
    public static Map<String, SDVariable> defineParams(SameDiff sameDiff, GraphVertex vertex, boolean useView){
        Map<String, INDArray> params = new HashMap<>(vertex.paramTable(false));
        vertex.transformParamsForSameDiff(params);
        return defineTransformedParams(sameDiff, params, (int) vertex.numParams(), useView);
    }

    /**
     * A helper for parameter definition.
     */
    private static Map<String, SDVariable> defineTransformedParams(SameDiff sameDiff, Map<String, INDArray> params, int numParams, boolean useView){
        Map<String, SDVariable> newParams = new HashMap<>(numParams);
        for (Map.Entry<String, INDArray> entry : params.entrySet()) {
            INDArray value = entry.getValue();
            if (!useView) {
                value = value.dup();
            }
            newParams.put(entry.getKey(), sameDiff.var(entry.getKey(), value));
        }
        return newParams;
    }

}
