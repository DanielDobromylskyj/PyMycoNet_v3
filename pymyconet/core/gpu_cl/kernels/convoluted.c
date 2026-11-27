
__kernel void forward(
    __global float* inputs,
    __global float* outputs,
    __global float* weights,
    __global float* biases,

    int input_width,
    int input_height,
    int output_width,
    int output_height,
    int kernel_width,
    int kernel_height,
    int step_x,
    int step_y,
    int filter_count
) {
    int filter_index   = get_global_id(0);
    int output_y_index = get_global_id(1);
    int output_x_index = get_global_id(2);

    int output_index = (filter_index * output_height * output_width) +
                          (output_y_index * output_width) + output_x_index;

    int filter_weight_offset = filter_index * kernel_height * kernel_width;

    int input_x_offset = step_x * output_x_index;
    int input_y_offset = step_y * output_y_index;


    float total = biases[filter_index];

    for (int ky = 0; ky < kernel_height; ky++) {
        for (int kx = 0; kx < kernel_width; kx++) {

            int weight_index = filter_weight_offset + (ky * kernel_width) + kx;

            int input_x = input_x_offset + kx;
            int input_y = input_y_offset + ky;

            // Validate weight index
            if (weight_index < 0 ||
                weight_index >= filter_count * kernel_height * kernel_width)
            {
                return;
            }

            float weight = weights[weight_index];

            float input_value = 0.0f;
            if (!(input_x < 0 || input_y < 0 || input_x >= input_width || input_y >= input_height)) {
                int input_index = (input_y * input_width) + input_x;

                if (input_index < 0 || input_index >= input_width * input_height) {
                    return;
                }

                input_value = inputs[input_index];
            }

            total += input_value * weight;
        }
    }

    outputs[output_index] = total;
}


__kernel void backward(
    __global float* inputs,
    __global float* previous_errors,
    __global float* weights,

    __global float* weight_gradients_unreduced,
    __global float* bias_gradients_unreduced,
    __global float* next_errors_unreduced,

    int input_width,
    int input_height,
    int output_width,
    int output_height,
    int kernel_width,
    int kernel_height,
    int step_x,
    int step_y,
    int filter_count,

    float learning_rate
) {
    int filter_index   = get_global_id(0);
    int output_y_index = get_global_id(1);
    int output_x_index = get_global_id(2);

    /* Python Code:

    # output_{x/y}_index = {x/y}_offset//self.step_{x/y}
    # {x/y}_offset is a input buffer offset

    delta = previous_errors[filter_index, y_offset//self.step_y, x_offset//self.step_x]
    bais_grads[filter_index] += delta * learning_rate

    for kernel_x in range(self.kernel_width):
        for kernel_y in range(self.kernel_height):
            weight_grad = delta * inputs[y_offset + kernel_y, x_offset + kernel_x] * learning_rate
            next_error_grad = delta * self.__weights[filter_index][kernel_y, kernel_x]

            weight_grads[filter_index, kernel_y, kernel_x] += weight_grad
            next_grads[y_offset + kernel_y, x_offset + kernel_x] += next_error_grad
    */

    // Calculate Indexes / Offsets
    int output_index = (filter_index * output_height * output_width) + (output_y_index * output_width) + output_x_index;
    int unreduced_base_offset = output_index; // Its the same calculation as above. ^^^

    // Spatial position (independent of filter)
    int spatial_index = (output_y_index * output_width) + output_x_index;

    // Each spatial position contains ALL filters and their kernels
    int unreduced_weight_offset = spatial_index * filter_count * kernel_height * kernel_width;
    int unreduced_next_error_offset = unreduced_weight_offset; // Same calc as above ^^^

    int unreduced_bias_index = spatial_index * filter_count + filter_index;

    // Filter-specific weight offsets
    int filter_weight_offset = filter_index * kernel_height * kernel_width;


    int input_x_offset = step_x * output_x_index;
    int input_y_offset = step_y * output_y_index;

    // Get Delta
    float delta = previous_errors[output_index];

    // Set Bias Gradient
    bias_gradients_unreduced[unreduced_bias_index] = delta * learning_rate;

    // Loop over all kernel weights / inputs
    for (int ky = 0; ky < kernel_height; ky++) {
        for (int kx = 0; kx < kernel_width; kx++) {
            /* Python Version (Difference: Inputs buffer is padded on cpu version)
            weight_grad = delta * inputs[y_offset + kernel_y, x_offset + kernel_x] * learning_rate
            next_error_grad = delta * self.__weights[filter_index][kernel_y, kernel_x]
            */
            int weight_index = filter_weight_offset + (ky * kernel_width) + kx;

            int input_x = input_x_offset + kx;
            int input_y = input_y_offset + ky;

            // Get Input Value, Ensuring we dont index out of bounds of the array
            int input_index = (input_y * input_width) + input_x;

            float input_value = 0.0f;
            if (!(input_x < 0 || input_y < 0 || input_x >= input_width || input_y >= input_height)) {

                if (input_index < 0 || input_index >= input_width * input_height) {
                    continue; // Skip, Its out of bounds, Somehow? This should never get reached?
                }

                input_value = inputs[input_index];

            }

            float weight_grad = delta * input_value * learning_rate;
            float next_error_grad = delta * weights[weight_index];

            /* Python Version (Different Indexing, as we cant use += on a buffer)
            weight_grads[filter_index, kernel_y, kernel_x] += weight_grad
            next_grads[y_offset + kernel_y, x_offset + kernel_x] += next_error_grad
            */

            weight_gradients_unreduced[unreduced_weight_offset + weight_index] = weight_grad;
            next_errors_unreduced[unreduced_next_error_offset + weight_index] = next_error_grad;
        }
    }
}

__kernel void reduce_generic(
    __global const float *unreduced_values,
    __global float *reduced_values,

    const int reduction_count,
    const int output_count
) {
    int idx = get_global_id(0);

    if (idx >= output_count)
        return;

    float sum = 0.0f;

    for (int i = 0; i < reduction_count; i++) {
        sum += unreduced_values[i * output_count + idx];
    }

    reduced_values[idx] = sum;
}

// This is kinda confusing, So many indexes
__kernel void reduce_next_errors(
    __global const float* unreduced_values,
    __global float* reduced_values,

    int input_width,
    int input_height,

    int output_width,
    int output_height,

    int kernel_width,
    int kernel_height,

    int step_x,
    int step_y,

    int filter_count
) {
    int input_x = get_global_id(0);
    int input_y = get_global_id(1);

    if (input_x >= input_width || input_y >= input_height)
        return;

    float sum = 0.0f;

    // Iterate through every kernel position
    for (int ky = 0; ky < kernel_height; ky++) {
        for (int kx = 0; kx < kernel_width; kx++) {

            // Reverse the forward convolution mapping
            int offset_x = input_x - kx;
            int offset_y = input_y - ky;

            // Invalid spatial mapping
            if (offset_x < 0 || offset_y < 0)
                continue;

            if (offset_x % step_x != 0 ||
                offset_y % step_y != 0)
                continue;

            int output_x = offset_x / step_x;
            int output_y = offset_y / step_y;

            // Ensure the output position is valid
            if (output_x >= output_width ||
                output_y >= output_height)
                continue;

            int spatial_index = output_y * output_width + output_x;

            // Sum contributions from every filter
            for (int f = 0; f < filter_count; f++) {

                int index =
                    spatial_index * filter_count * kernel_height * kernel_width
                    + f * kernel_height * kernel_width
                    + ky * kernel_width
                    + kx;

                sum += unreduced_values[index];
            }
        }
    }

    int output_index = input_y * input_width + input_x;
    reduced_values[output_index] = sum;
}

__kernel void reduce_next_errors_debug(
    __global const float* unreduced_values,
    __global float* reduced_values,

    int input_width,
    int input_height,

    int output_width,
    int output_height,

    int kernel_width,
    int kernel_height,

    int step_x,
    int step_y,

    int filter_count
) {
    int input_x = get_global_id(0);
    int input_y = get_global_id(1);

    if (input_x >= input_width || input_y >= input_height)
        return;

    float sum = 0.0f;

    for (int oy = 0; oy < output_height; oy++) {
        for (int ox = 0; ox < output_width; ox++) {

            int spatial_index = oy * output_width + ox;

            int base_offset =
                spatial_index * filter_count * kernel_height * kernel_width;

            int origin_x = ox * step_x;
            int origin_y = oy * step_y;

            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {

                    if (origin_x + kx != input_x ||
                        origin_y + ky != input_y)
                        continue;

                    for (int f = 0; f < filter_count; f++) {

                        int index =
                            base_offset
                            + f * kernel_height * kernel_width
                            + ky * kernel_width
                            + kx;

                        sum += unreduced_values[index];
                    }
                }
            }
        }
    }

    reduced_values[input_y * input_width + input_x] = sum;
}