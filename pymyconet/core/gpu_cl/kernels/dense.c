
// Possible todo
// Add partial summing, so we will do a loop in forward() that will partially sum say every 256/100k nodes,
// instead of doing this current approach

__kernel void forward(
    __global float *inputs,
    __global float *unreduced_outputs,
    __global float *weights,
    __global float *biases,
    int input_size,
    int output_size
) {
    int input_index = get_global_id(0);
    int output_index = get_global_id(1);

    int weight_index = output_index * input_size + input_index;

    float weight = weights[weight_index];

    float value = weights[weight_index] * inputs[input_index];

    if (input_index == 0) {
        value += biases[output_index];
    }

    // OpenCL 2.0 doesn't guarantee float atomic adding. (And my hardware cant use it, so we reduce later)
    unreduced_outputs[weight_index] = value;

    // We could also add our biases here, by checking that input_index == 0, but that adds branching,
    // so we can also do it in the reduce function as it only loops over each output index.
    // - No clue what this actually does performance wise, just a guess.
}


__kernel void reduce(
    __global float *unreduced,
    __global float *outputs,
    int reduce_size,
    int output_size
) {
    int output_index = get_global_id(0);

    float total = 0.0f;
    for (int i = 0; i < reduce_size; i++) {
        total += unreduced[output_index * reduce_size + i];
    }

    outputs[output_index] = total;
}

// Its time for my fav section!
//   > Back Propagation


__kernel void backward(
    __global float *inputs,
    __global float *weights,
    __global float *previous_errors,

    __global float *next_errors_unreduced,
    __global float *bias_gradients,
    __global float *weight_gradients,

    int input_size,
    int output_size,
    float learning_rate
) {
    int input_index = get_global_id(0);
    int output_index = get_global_id(1);

    // Index Calcs
    int weight_index = output_index * input_size + input_index;
    int next_error_index = input_index * output_size + output_index;

    // Delta
    float delta = previous_errors[output_index];

    // Calculate Gradients
    float weight_grad = delta * inputs[input_index] * learning_rate;
    float bias_grad = delta * learning_rate;
    float next_error_grad = delta * weights[weight_index];

    // Set Gradients
    weight_gradients[weight_index] = weight_grad;
    bias_gradients[output_index] = bias_grad;
    next_errors_unreduced[next_error_index] = next_error_grad;
}

