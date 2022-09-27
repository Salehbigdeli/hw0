#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


double* batch_dot(size_t s, size_t e, size_t n, size_t k, const double *X, double *theta) {
    double * res = new double[(e-s)*k];
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
    double * res2 = new double[(e-s)*k];
    for (size_t i = 0; i < (e-s)*k; i++) res2[i] = res[i];
    return res2;
}

void z_normalized_minus_Iy(int m, int n, double *x, const unsigned char *y) {
    double sum = 0;
    // for (size_t i = 0; i < m; i++) sums[i] = 0;

    for (int i=0 ; i < m; ++i) {
        sum = 0;
        for (int j=0 ; j < n ; ++j) {
            // x[i*n+j] = exp(x[i*n + j]);
            sum += exp(x[i*n + j]);
        }
        for (int j=0 ; j < n ; ++j) {
            x[i*n+j] = exp(x[i*n + j])/sum;
            if (y[i] == j) x[i*n+j] -= 1;
        }
        
    }
    // for (size_t i = 0; i < m; i++) std::cout<<"\n"<<sums[i] <<"\n";
}

double* batch_transpose(int s, int e, int n, const double *x){
    double *xt = new double[n*(e-s)];
    // for (size_t i = 0; i < (e-s)*n; i++) xt[i] = 0;
    for (int i=s ; i < e; ++i) {
        for (int j=0 ; j < n ; ++j) {
            xt[j*(e-s) + i-s] = x[i*n + j];
        }
    }
    return xt;
}

void update_theta(int m, int n, double *x, double *grad, double lr, int batch) {
    for (int i=0 ; i < m; ++i) {
        for (int j=0 ; j < n ; ++j) {
            x[i*n + j] -= lr*(grad[i*n + j]/(double)batch);
        }
    }
}

void print(const double *X, size_t m, size_t n) {
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
    double * X2 = new double[m*n];
    for (size_t i = 0; i < m*n; i++) X2[i] = X[i];

    double * theta2 = new double[k*n];
    for (size_t i = 0; i < k*n; i++) theta2[i] = theta[i];
    double lr2 = lr;
    
    size_t n_batches = (m + batch - 1)/batch;
    size_t num = n_batches*batch;
    for (size_t i = 0; i < num; i+=batch)
    {
        std::cout<<"hi "<<i<<"\n";
        size_t batch_start = i;
        size_t batch_end = i + batch;
        if (batch_end > m) batch_end = m;
        double * H = batch_dot(batch_start, batch_end, n, k, X2, theta2);
        // std::cout<<"\n\n\nhello\n\n\n\n";
        // print(H, 50, k);
        // print(X, m, n);
        z_normalized_minus_Iy(batch_end-batch_start, k, H, y);
        // print()
        // std::cout<<"\n\n\nhello "<< m <<"   " <<n <<"   " << k << "\n\n\n\n";
        double *bt = batch_transpose(batch_start, batch_end, n, X2);
        // std::cout<<"\n\n\nhello "<< batch_start <<"   " <<batch_end << "\n\n\n\n";
        // print(bt, n, batch_end-batch_start);
        double *grad = batch_dot(0, n, batch_end-batch_start, k, bt, H);
        update_theta(n, k, theta2, grad, lr2, batch_end-batch_start);
    }
    for (size_t i = 0; i < k*n; i++) theta[i] = (float)theta2[i];
    
    
    /*
    sample_size = len(X)
    n_batches = (sample_size + batch - 1)//batch
    n = n_batches*batch
    n_clss = theta.shape[1]
    count = 0
    for i in range(0, n, batch):
        count += 1
        batch_start, batch_end = i, min(i + batch, len(y))
        X_batch, y_batch = X[batch_start: batch_end, ...], y[batch_start: batch_end, ...]
        m = len(y_batch)
        H = X_batch.dot(theta)
        Z = np.exp(H)/np.sum(np.exp(H), axis=1).reshape(m, 1)
        Iy = np.eye(n_clss)[y_batch]
        grad = np.transpose(X_batch).dot(Z-Iy)/m
        theta -= lr*grad  #   IMPORTANT: theta = theta - lr*grad is not working!!!

    */
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
