# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from cpython cimport Py_INCREF, PyObject, PyTypeObject
from libc.math cimport fabs
from libc.math cimport log as ln
from libc.stdint cimport SIZE_MAX
from libc.stdlib cimport calloc
from libc.stdlib cimport free
from libc.stdlib cimport malloc
from libc.stdlib cimport qsort
from libc.stdlib cimport realloc
from libc.string cimport memset
from libc.string cimport memcpy
import numpy as np
cimport numpy as np
from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE
np.import_array()
cdef inline UINT32_t DEFAULT_SEED = 1
cdef double INFINITY = np.inf
cdef double EPSILON = np.finfo('double').eps
cdef int IS_LEFT = 1
TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10
cdef Node dummy;
NODE_DTYPE = np.asarray(<Node[:1]>(&dummy)).dtype
cdef enum:
    RAND_R_MAX = 0x7FFFFFFF
ctypedef fused realloc_ptr:
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)
    (DOUBLE_t*)
    (DOUBLE_t**)
    (Node*)
    (Node**)
    (StackRecord*)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *:
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        with gil:
            raise MemoryError("could not allocate (%d * %d) bytes"
                              % (nelems, sizeof(p[0][0])))
    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        with gil:
            raise MemoryError("could not allocate %d bytes" % nbytes)
    p[0] = tmp
    return tmp


cdef inline double log(double x) nogil:
    return ln(x) / ln(2.0)


cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    if (seed[0] == 0):
        seed[0] = DEFAULT_SEED
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)
    return seed[0] % <UINT32_t>(RAND_R_MAX + 1)


cdef inline SIZE_t rand_int(SIZE_t low, SIZE_t high,
                            UINT32_t* random_state) nogil:
    return low + our_rand_r(random_state) % (high - low)


cdef DTYPE_t FEATURE_THRESHOLD = 1e-7
cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY


cdef void introsort(DTYPE_t* Xf, SIZE_t *samples, SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r
    while n > 1:
        if maxd <= 0:
            heapsort(Xf, samples, n)
            return
        maxd -= 1
        pivot = median3(Xf, n)
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1
        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples, SIZE_t start,
                           SIZE_t end) nogil:
    cdef SIZE_t child, maxind, root
    root = start
    while True:
        child = root * 2 + 1
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1
        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1


cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    if n == 0:
      return
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples, SIZE_t i, SIZE_t j) nogil:
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    cdef DTYPE_t a = Xf[0], b = Xf[n / 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(PyTypeObject* subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)
    int PyArray_SetBaseObject(np.ndarray arr, PyObject* obj)


cdef class Criterion():
    def __cinit__(self, SIZE_t n_outputs,
                  np.ndarray[SIZE_t, ndim=1] n_classes):
        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0
        self.n_outputs = n_outputs
        self.n_samples = 0
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0
        self.sum_total = NULL
        self.sum_left = NULL
        self.sum_right = NULL
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)
        cdef SIZE_t k = 0
        cdef SIZE_t sum_stride = 0
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]
            if n_classes[k] > sum_stride:
                sum_stride = n_classes[k]
        self.sum_stride = sum_stride
        cdef SIZE_t n_elements = n_outputs * sum_stride
        self.sum_total = <double*> calloc(n_elements, sizeof(double))
        self.sum_left = <double*> calloc(n_elements, sizeof(double))
        self.sum_right = <double*> calloc(n_elements, sizeof(double))
        if (self.sum_total == NULL or
                self.sum_left == NULL or
                self.sum_right == NULL):
            raise MemoryError()

    def __dealloc__(self):
        free(self.sum_total)
        free(self.sum_left)
        free(self.sum_right)
        free(self.n_classes)

    cdef int init(self, const DOUBLE_t[:, ::1] y, double weighted_n_samples,
                  SIZE_t* samples, SIZE_t start, SIZE_t end) nogil except -1:
        self.y = y
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        self.weighted_n_node_samples = 0.0
        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t offset = 0
        for k in range(self.n_outputs):
            memset(sum_total + offset, 0, n_classes[k] * sizeof(double))
            offset += self.sum_stride
        for p in range(start, end):
            i = samples[p]
            for k in range(self.n_outputs):
                c = <SIZE_t> self.y[i, k]
                sum_total[k * self.sum_stride + c] += w
            self.weighted_n_node_samples += w
        self.reset()
        return 0

    cdef int reset(self) nogil except -1:
        self.pos = self.start
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples
        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k
        for k in range(self.n_outputs):
            memset(sum_left, 0, n_classes[k] * sizeof(double))
            memcpy(sum_right, sum_total, n_classes[k] * sizeof(double))
            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride
        return 0

    cdef int reverse_reset(self) nogil except -1:
        self.pos = self.end
        self.weighted_n_left = self.weighted_n_node_samples
        self.weighted_n_right = 0.0
        cdef double* sum_total = self.sum_total
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k
        for k in range(self.n_outputs):
            memset(sum_right, 0, n_classes[k] * sizeof(double))
            memcpy(sum_left, sum_total, n_classes[k] * sizeof(double))
            sum_total += self.sum_stride
            sum_left += self.sum_stride
            sum_right += self.sum_stride
        return 0

    cdef double proxy_impurity_improvement(self) nogil:
        cdef double impurity_left
        cdef double impurity_right
        self.children_impurity(&impurity_left, &impurity_right)
        return (- self.weighted_n_right * impurity_right
                - self.weighted_n_left * impurity_left)

    cdef double impurity_improvement(self, double impurity_parent,
                                     double impurity_left,
                                     double impurity_right) nogil:
        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity_parent-(self.weighted_n_right /
                                  self.weighted_n_node_samples*impurity_right)
                                -(self.weighted_n_left /
                                  self.weighted_n_node_samples*impurity_left)))

    cdef int update(self, SIZE_t new_pos) nogil except -1:
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t c
        cdef SIZE_t label_index
        cdef DOUBLE_t w = 1.0
        if (new_pos - pos) <= (end - new_pos):
            for p in range(pos, new_pos):
                i = samples[p]
                for k in range(self.n_outputs):
                    label_index = k * self.sum_stride + <SIZE_t> self.y[i, k]
                    sum_left[label_index] += w
                self.weighted_n_left += w
        else:
            self.reverse_reset()
            for p in range(end - 1, new_pos - 1, -1):
                i = samples[p]
                for k in range(self.n_outputs):
                    label_index = k * self.sum_stride + <SIZE_t> self.y[i, k]
                    sum_left[label_index] -= w
                self.weighted_n_left -= w
        self.weighted_n_right = (self.weighted_n_node_samples -
                                 self.weighted_n_left)
        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                sum_right[c] = sum_total[c] - sum_left[c]
            sum_right += self.sum_stride
            sum_left += self.sum_stride
            sum_total += self.sum_stride
        self.pos = new_pos
        return 0

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        cdef double* sum_total = self.sum_total
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t k
        for k in range(self.n_outputs):
            memcpy(dest, sum_total, n_classes[k] * sizeof(double))
            dest += self.sum_stride
            sum_total += self.sum_stride


cdef class Entropy(Criterion):
    cdef double node_impurity(self) nogil:
        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double entropy = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c
        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_total[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_node_samples
                    entropy -= count_k * log(count_k)
            sum_total += self.sum_stride
        return entropy / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double entropy_left = 0.0
        cdef double entropy_right = 0.0
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c
        for k in range(self.n_outputs):
            for c in range(n_classes[k]):
                count_k = sum_left[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_left
                    entropy_left -= count_k * log(count_k)
                count_k = sum_right[c]
                if count_k > 0.0:
                    count_k /= self.weighted_n_right
                    entropy_right -= count_k * log(count_k)
            sum_left += self.sum_stride
            sum_right += self.sum_stride
        impurity_left[0] = entropy_left / self.n_outputs
        impurity_right[0] = entropy_right / self.n_outputs


cdef class Gini(Criterion):
    cdef double node_impurity(self) nogil:
        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_total = self.sum_total
        cdef double gini = 0.0
        cdef double sq_count
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c
        for k in range(self.n_outputs):
            sq_count = 0.0
            for c in range(n_classes[k]):
                count_k = sum_total[c]
                sq_count += count_k * count_k
            gini += 1.0 - sq_count / (self.weighted_n_node_samples *
                                      self.weighted_n_node_samples)
            sum_total += self.sum_stride
        return gini / self.n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        cdef SIZE_t* n_classes = self.n_classes
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double gini_left = 0.0
        cdef double gini_right = 0.0
        cdef double sq_count_left
        cdef double sq_count_right
        cdef double count_k
        cdef SIZE_t k
        cdef SIZE_t c
        for k in range(self.n_outputs):
            sq_count_left = 0.0
            sq_count_right = 0.0
            for c in range(n_classes[k]):
                count_k = sum_left[c]
                sq_count_left += count_k * count_k
                count_k = sum_right[c]
                sq_count_right += count_k * count_k
            gini_left += 1.0 - sq_count_left / (self.weighted_n_left *
                                                self.weighted_n_left)
            gini_right += 1.0 - sq_count_right / (self.weighted_n_right *
                                                  self.weighted_n_right)
            sum_left += self.sum_stride
            sum_right += self.sum_stride
        impurity_left[0] = gini_left / self.n_outputs
        impurity_right[0] = gini_right / self.n_outputs


cdef class AveragePrecision(Criterion):
    # Assuming there are only two classes in y, and one output
    cdef double node_impurity(self) nogil:
        cdef double ap = self.sum_total[1]/(self.sum_total[0] +
                                            self.sum_total[1])
        return 1-ap

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        cdef double npl = self.sum_left[1]
        cdef double npr = self.sum_right[1]
        cdef double npo = npr + npl
        cdef double nl = self.pos-self.start
        cdef double nr = self.end-self.pos
        cdef double n = self.end-self.start
        cdef double ap_left = 0.0
        cdef double ap_right = 0.0
        if npo == 0:
            ap_left = 1
            ap_right = 1
        else:
            ap_left = (npl*npl)/(npo*nl)+npr/n
            ap_right = (npr*npr)/(npo*nr)+npl/n
        impurity_left[0] = 1-ap_left
        impurity_right[0] = 1-ap_right


cdef class Splitter:
    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state):
        self.criterion = criterion
        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state

    def __dealloc__(self):
        free(self.samples)
        free(self.features)
        free(self.constant_features)
        free(self.feature_values)

    cdef int init(self, object X, const DOUBLE_t[:, ::1] y) except -1:
        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)
        cdef SIZE_t i, j
        cdef double weighted_n_samples = 0.0
        j = 0
        for i in range(n_samples):
            samples[j] = i
            j += 1
            weighted_n_samples += 1.0
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples
        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)
        for i in range(n_features):
            features[i] = i
        self.n_features = n_features
        safe_realloc(&self.feature_values, n_samples)
        safe_realloc(&self.constant_features, n_features)
        self.X = X
        self.y = y
        return 0

    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1:
        self.start = start
        self.end = end
        self.criterion.init(self.y, self.weighted_n_samples,
                            self.samples, start, end)
        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef void node_value(self, double* dest) nogil:
        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        return self.criterion.node_impurity()

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features
        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state
        cdef SplitRecord best, current
        cdef double current_proxy_improvement = -INFINITY
        cdef double best_proxy_improvement = -INFINITY
        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t p
        cdef SIZE_t feature_idx_offset
        cdef SIZE_t feature_offset
        cdef SIZE_t i
        cdef SIZE_t j
        cdef SIZE_t n_visited_features = 0
        cdef SIZE_t n_found_constants = 0
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end
        _init_split(&best, end)
        while (f_i > n_total_constants and
                (n_visited_features < max_features or
                 n_visited_features <= n_found_constants + n_drawn_constants)):
            n_visited_features += 1
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)
            if f_j < n_known_constants:
                features[n_drawn_constants], features[f_j] = (
                                    features[f_j], features[n_drawn_constants])
                n_drawn_constants += 1
            else:
                f_j += n_found_constants
                current.feature = features[f_j]
                for i in range(start, end):
                    Xf[i] = self.X[samples[i], current.feature]
                sort(Xf + start, samples + start, end - start)
                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j], features[n_total_constants] = (
                                    features[n_total_constants], features[f_j])
                    n_found_constants += 1
                    n_total_constants += 1
                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]
                    self.criterion.reset()
                    p = start
                    while p < end:
                        while (p + 1 < end and
                               Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                            p += 1
                        p += 1
                        if p < end:
                            current.pos = p
                            if (((current.pos - start) < min_samples_leaf) or
                                    ((end - current.pos) < min_samples_leaf)):
                                continue
                            self.criterion.update(current.pos)
                            if ((self.criterion.weighted_n_left <
                                 min_weight_leaf) or
                                (self.criterion.weighted_n_right <
                                 min_weight_leaf)):
                                continue
                            current_proxy_improvement = (
                                   self.criterion.proxy_impurity_improvement())
                            if (current_proxy_improvement >
                               best_proxy_improvement):
                                best_proxy_improvement = (
                                                     current_proxy_improvement)
                                current.threshold = Xf[p-1] / 2.0 + Xf[p] / 2.0
                                if ((current.threshold == Xf[p]) or
                                    (current.threshold == INFINITY) or
                                    (current.threshold == -INFINITY)):
                                    current.threshold = Xf[p - 1]
                                best = current
        if best.pos < end:
            partition_end = end
            p = start
            while p < partition_end:
                if self.X[samples[p], best.feature] <= best.threshold:
                    p += 1
                else:
                    partition_end -= 1
                    samples[p], samples[partition_end] = (
                                            samples[partition_end], samples[p])
            self.criterion.reset()
            self.criterion.update(best.pos)
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)
            best.improvement = self.criterion.impurity_improvement(
                impurity, best.impurity_left, best.impurity_right)
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0


cdef class Stack:
    def __cinit__(self, SIZE_t capacity):
        self.capacity = capacity
        self.top = 0
        self.stack_ = <StackRecord*> malloc(capacity * sizeof(StackRecord))

    def __dealloc__(self):
        free(self.stack_)

    cdef bint is_empty(self) nogil:
        return self.top <= 0

    cdef int push(self, SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent,
                  bint is_left, double impurity,
                  SIZE_t n_constant_features) nogil except -1:
        cdef SIZE_t top = self.top
        cdef StackRecord* stack = NULL
        if top >= self.capacity:
            self.capacity *= 2
            safe_realloc(&self.stack_, self.capacity)
        stack = self.stack_
        stack[top].start = start
        stack[top].end = end
        stack[top].depth = depth
        stack[top].parent = parent
        stack[top].is_left = is_left
        stack[top].impurity = impurity
        stack[top].n_constant_features = n_constant_features
        self.top = top + 1
        return 0

    cdef int pop(self, StackRecord* res) nogil:
        cdef SIZE_t top = self.top
        cdef StackRecord* stack = self.stack_
        if top <= 0:
            return -1
        res[0] = stack[top - 1]
        self.top = top - 1
        return 0


cdef class TreeBuilder():
    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth, double min_impurity_decrease,
                  double min_impurity_split):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

    cdef inline _check_input(self, object X, np.ndarray y):
        if X.dtype != DTYPE:
            X = np.asfortranarray(X, dtype=DTYPE)
        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)
        return X, y

    cpdef build(self, Tree tree, object X, np.ndarray y):
        X, y = self._check_input(X, y)
        cdef int init_capacity
        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047
        tree._resize(init_capacity)
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split
        cdef double min_impurity_decrease = self.min_impurity_decrease
        cdef double min_impurity_split = self.min_impurity_split
        splitter.init(X, y)
        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double weighted_n_node_samples
        cdef SplitRecord split
        cdef SIZE_t node_id
        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0
        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record
        with nogil:
            rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY,
                            0)
            if rc == -1:
                with gil:
                    raise MemoryError()
            while not stack.is_empty():
                stack.pop(&stack_record)
                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features
                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples)
                is_leaf = (depth >= max_depth or
                           n_node_samples < min_samples_split or
                           n_node_samples < 2 * min_samples_leaf or
                           weighted_n_node_samples < 2 * min_weight_leaf)
                if first:
                    impurity = splitter.node_impurity()
                    first = 0
                is_leaf = (is_leaf or
                           (impurity <= min_impurity_split))
                if not is_leaf:
                    splitter.node_split(impurity, &split, &n_constant_features)
                    is_leaf = (is_leaf or split.pos >= end or
                               (split.improvement + EPSILON <
                                min_impurity_decrease))
                node_id = tree._add_node(parent, is_left, is_leaf,
                                         split.feature, split.threshold,
                                         impurity, n_node_samples,
                                         weighted_n_node_samples)
                if node_id == SIZE_MAX:
                    rc = -1
                    break
                splitter.node_value(tree.value + node_id * tree.value_stride)
                if not is_leaf:
                    rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                                    split.impurity_right, n_constant_features)
                    if rc == -1:
                        break
                    rc = stack.push(start, split.pos, depth + 1, node_id, 1,
                                    split.impurity_left, n_constant_features)
                    if rc == -1:
                        break
                if depth > max_depth_seen:
                    max_depth_seen = depth
            if rc >= 0:
                rc = tree._resize_c(tree.node_count)
            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()


cdef class Tree:
    property n_classes:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.n_outputs
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INTP,
                                                self.n_classes).copy()

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

    property n_leaves:
        def __get__(self):
            return np.sum(np.logical_and(
                self.children_left == -1,
                self.children_right == -1))

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.node_count]

    property impurity:
        def __get__(self):
            return self._get_node_ndarray()['impurity'][:self.node_count]

    property n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['n_node_samples'][:self.node_count]

    property weighted_n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['weighted_n_node_samples'][
                                                              :self.node_count]

    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count]

    def __cinit__(self, int n_features, np.ndarray[SIZE_t, ndim=1] n_classes,
                  int n_outputs):
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)
        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes
        cdef SIZE_t k
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL

    def __dealloc__(self):
        free(self.n_classes)
        free(self.value)
        free(self.nodes)

    cdef int _resize(self, SIZE_t capacity) nogil except -1:
        if self._resize_c(capacity) != 0:
            with gil:
                raise MemoryError()

    cdef int _resize_c(self, SIZE_t capacity=SIZE_MAX) nogil except -1:
        if capacity == self.capacity and self.nodes != NULL:
            return 0
        if capacity == SIZE_MAX:
            if self.capacity == 0:
                capacity = 3
            else:
                capacity = 2 * self.capacity
        safe_realloc(&self.nodes, capacity)
        safe_realloc(&self.value, capacity * self.value_stride)
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride *
                   sizeof(double))
        if capacity < self.node_count:
            self.node_count = capacity
        self.capacity = capacity
        return 0

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples,
                          double weighted_n_node_samples) nogil except -1:
        cdef SIZE_t node_id = self.node_count
        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return SIZE_MAX
        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples
        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id
        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED
        else:
            node.feature = feature
            node.threshold = threshold
        self.node_count += 1
        return node_id

    cpdef np.ndarray predict(self, object X):
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    cpdef np.ndarray apply(self, object X):
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))
        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)
        cdef const DTYPE_t[:, :] X_ndarray = X
        cdef SIZE_t n_samples = X.shape[0]
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data
        cdef Node* node = NULL
        cdef SIZE_t i = 0
        with nogil:
            for i in range(n_samples):
                node = self.nodes
                while node.left_child != _TREE_LEAF:
                    if X_ndarray[i, node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]
                out_ptr[i] = <SIZE_t>(node - self.nodes)
        return out

    cdef np.ndarray _get_value_ndarray(self):
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr

    cdef np.ndarray _get_node_ndarray(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(<PyTypeObject *> np.ndarray,
                                   <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        if PyArray_SetBaseObject(arr, <PyObject*> self) < 0:
            raise ValueError("Can't initialize array.")
        return arr
