Notes:

-   Each Node must have as many weights as nodes in the next layer.
    For example, if layers[0] has 2 nodes, and layers[1] has 4 nodes,
    each node in layer[0] must have 4 weights.


back prop:

l2_delta = l2_err * d_sigmoid(layer2)
l1_err = l2_delta.dot( l1_weights )
l1_delta = l1_err * d_sigmoid(l1)