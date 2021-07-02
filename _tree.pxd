import numpy as np
cimport numpy as np
ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer


cdef struct SplitRecord:
    SIZE_t feature         # Which feature to split on.
    SIZE_t pos             # Split samples array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos is >= end if the node is a leaf.
    double threshold       # Threshold to split at.
    double improvement     # Impurity improvement given parent node.
    double impurity_left   # Impurity of the left split.
    double impurity_right  # Impurity of the right split.


cdef struct StackRecord:
    SIZE_t start
    SIZE_t end
    SIZE_t depth
    SIZE_t parent
    bint is_left
    double impurity
    SIZE_t n_constant_features


cdef struct Node:
    SIZE_t left_child                    # id of the left child of the node
    SIZE_t right_child                   # id of the right child of the node
    SIZE_t feature                       # Feature used for splitting the node
    DOUBLE_t threshold                   # Threshold value at the node
    DOUBLE_t impurity                    # Impurity of the node
    SIZE_t n_node_samples                # Number of samples at the node
    DOUBLE_t weighted_n_node_samples     # Weighted number of samples at node


cdef class Criterion:
    cdef const DOUBLE_t[:, ::1] y        # Values of y
    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t start                    # samples[start:pos] left node
    cdef SIZE_t pos                      # samples[pos:end] right node
    cdef SIZE_t end
    cdef SIZE_t n_outputs                # Number of outputs
    cdef SIZE_t n_samples                # Number of samples
    cdef SIZE_t n_node_samples           # Number of samples in node end-start
    cdef double weighted_n_samples       # Weighted number of samples (total)
    cdef double weighted_n_node_samples  # Weighted number of samples in node
    cdef double weighted_n_left          # Weighted number of samples in left
    cdef double weighted_n_right         # Weighted number of samples in right
    cdef double* sum_total          # The sum of the weighted count of each
                                    # label.
    cdef double* sum_left           # Same as above, but for the left
    cdef double* sum_right          # same as above, but for the right
    cdef SIZE_t* n_classes
    cdef SIZE_t sum_stride
    cdef int init(self, const DOUBLE_t[:, ::1] y,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1
    cdef int reset(self) nogil except -1
    cdef int reverse_reset(self) nogil except -1
    cdef int update(self, SIZE_t new_pos) nogil except -1
    cdef double node_impurity(self) nogil
    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil
    cdef void node_value(self, double* dest) nogil
    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil
    cdef double proxy_impurity_improvement(self) nogil


cdef class Splitter:
    cdef public Criterion criterion      # Impurity criterion
    cdef public SIZE_t max_features      # Number of features to test
    cdef public SIZE_t min_samples_leaf  # Min samples in a leaf
    cdef public double min_weight_leaf   # Minimum weight in a leaf
    cdef object random_state             # Random state
    cdef UINT32_t rand_r_state           # sklearn_rand_r random number state
    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t n_samples                # X.shape[0]
    cdef double weighted_n_samples       # Weighted number of samples
    cdef SIZE_t* features                # Feature indices in X
    cdef SIZE_t* constant_features       # Constant features indices
    cdef SIZE_t n_features               # X.shape[1]
    cdef DTYPE_t* feature_values         # temp. array holding feature values
    cdef SIZE_t start                    # Start position for the current node
    cdef SIZE_t end                      # End position for the current node
    cdef const DTYPE_t[:, :] X
    cdef const DOUBLE_t[:, ::1] y
    cdef SIZE_t n_total_samples
    cdef int init(self, object X, const DOUBLE_t[:, ::1] y) except -1
    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1
    cdef int node_split(self,
                        double impurity,   # Impurity of the node
                        SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1
    cdef void node_value(self, double* dest) nogil
    cdef double node_impurity(self) nogil


cdef class Stack:
    cdef SIZE_t capacity
    cdef SIZE_t top
    cdef StackRecord* stack_
    cdef bint is_empty(self) nogil
    cdef int push(self, SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent,
                  bint is_left, double impurity,
                  SIZE_t n_constant_features) nogil except -1
    cdef int pop(self, StackRecord* res) nogil


cdef class Tree:
    cdef public SIZE_t n_features        # Number of features in X
    cdef SIZE_t* n_classes               # Number of classes in y[:, k]
    cdef public SIZE_t n_outputs         # Number of outputs in y
    cdef public SIZE_t max_n_classes     # max(n_classes)
    cdef public SIZE_t max_depth         # Max depth of the tree
    cdef public SIZE_t node_count        # Counter for node IDs
    cdef public SIZE_t capacity          # Capacity of tree, in terms of nodes
    cdef Node* nodes                     # Array of nodes
    cdef double* value                   # (capacity, n_outputs, max_n_classes)
    cdef SIZE_t value_stride             # = n_outputs * max_n_classes
    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples, double weighted_n_samples) nogil except -1
    cdef int _resize(self, SIZE_t capacity) nogil except -1
    cdef int _resize_c(self, SIZE_t capacity=*) nogil except -1
    cdef np.ndarray _get_value_ndarray(self)
    cdef np.ndarray _get_node_ndarray(self)
    cpdef np.ndarray predict(self, object X)
    cpdef np.ndarray apply(self, object X)


cdef class TreeBuilder:
    cdef Splitter splitter              # Splitting algorithm
    cdef SIZE_t min_samples_split       # Minimum number of samples in an internal node
    cdef SIZE_t min_samples_leaf        # Minimum number of samples in a leaf
    cdef double min_weight_leaf         # Minimum weight in a leaf
    cdef SIZE_t max_depth               # Maximal tree depth
    cdef double min_impurity_split
    cdef double min_impurity_decrease   # Impurity threshold for early stopping
    cpdef build(self, Tree tree, object X, np.ndarray y)
    cdef _check_input(self, object X, np.ndarray y)
