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

package org.nd4j.imports.keras.constraints;

import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.nd4j.imports.keras.deserialize.KerasWrappedJson;
import org.nd4j.imports.keras.regularizers.L1;
import org.nd4j.imports.keras.regularizers.L1L2;
import org.nd4j.imports.keras.regularizers.L2;
import org.nd4j.shade.jackson.annotation.JsonSubTypes;
import org.nd4j.shade.jackson.annotation.JsonSubTypes.Type;

@KerasWrappedJson
@Getter
@NoArgsConstructor
@ToString
@EqualsAndHashCode
@JsonSubTypes({
        @Type(MaxNorm.class),
        @Type(NonNeg.class),
        @Type(UnitNorm.class),
        @Type(MinMaxNorm.class),
        @Type(RadialConstraint.class)
})
public abstract class KerasConstraint {

}
