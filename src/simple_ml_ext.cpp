#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


float* batch_dot(size_t s, size_t e, size_t n, size_t k, const float *X, float *theta) {
    float * res = new float[(e-s)*k];
    for (size_t i = 0; i < (e-s)*k; i++)
    {
        res[i] = 0;
    }
    
    for (size_t i=s ; i < e; ++i) {
        for (size_t j=0 ; j < k ; ++j) {
            for (size_t t =0; t<n ; ++t) {
                res[(i-s)*k +j] += X[i*n + t]*theta[t*k + j];
            }
        }
    }
    
    return res;
}

void z_normalized_minus_Iy(int m, int n, int batch_start, float *x, const unsigned char *y) {
    float sum;
    for (int i=0 ; i < m; ++i) {
        sum = 0;
        for (int j=0 ; j < n ; ++j) {
            sum += exp(x[i*n + j]);
        }
        for (int j=0 ; j < n ; ++j) {
            x[i*n+j] = exp(x[i*n + j])/sum;
            if (y[batch_start + i] == j) x[i*n+j] -= 1;
        }
        
    }
}

float* batch_transpose(int s, int e, int n, const float *x){
    float *xt = new float[n*(e-s)];
    for (int i=s ; i < e; ++i) {
        for (int j=0 ; j < n ; ++j) {
            xt[j*(e-s) + i-s] = x[i*n + j];
        }
    }
    return xt;
}

void update_theta(int m, int n, float *x, float *grad, float lr, float bs) {
    for (int i=0 ; i < m*n; ++i) x[i] -= lr*grad[i]/bs;
}

void print(const float *X, size_t m, size_t n) {
    for (size_t i=0 ; i < m; ++i) {
        for (size_t j=0 ; j < n ; ++j) {
            std::cout<<X[i*n + j] <<"  ";
        }
        std::cout<<"\n\n";
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    
    size_t n_batches = (m + batch - 1)/batch;
    size_t num = n_batches*batch;
    for (size_t i = 0; i < num; i+=batch)
    {
        size_t batch_start = i;
        size_t batch_end = i + batch;
        if (batch_end > m) batch_end = m;
        size_t bs = batch_end - batch_start;
        float * H = batch_dot(batch_start, batch_end, n, k, X, theta);
        
        z_normalized_minus_Iy(bs, k, batch_start, H, y);
        float *bt = batch_transpose(batch_start, batch_end, n, X);
        
        float *grad = batch_dot(0, n, bs, k, bt, H);
        update_theta(n, k, theta, grad, (float)lr, (float)bs);
    }
    
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
