// -*- lsst-c++ -*-

#include <boost/numeric/ublas/io.hpp>
#include <boost/timer.hpp> 
#include <boost/numeric/ublas/matrix.hpp>

#include <vw/Core/Exception.h> 
#include <vw/Math/Matrix.h> 
#include <vw/Math/Vector.h> 
#include <vw/Math/LinearAlgebra.h> 

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_machine.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_linalg.h>

#include "Eigen/Cholesky"
#include "Eigen/Core"
#include "Eigen/LU"
#include "Eigen/QR"
#include "Eigen/SVD"

using namespace boost::numeric;
using namespace std;

/************************************************************************************************************/

template<typename T>
class RHLMatrix {
public:
    RHLMatrix(int n, int m=1) :
        _n(n), _m(m), _rows(*new std::vector<T*>(m)), _data(new T[n*m]) {

        for (int i = 0; i != m; ++i) {
            _rows[i] = &_data[i*n];
        }
    }

    ~RHLMatrix() {
        delete &_rows;
        delete[] _data;
    }

    T& operator()(int i, int j) {
        return _rows[i][j];
    }
    T const& operator()(int i, int j) const {
        return _rows[i][j];
    }
    T& operator()(int j) {
        return _rows[0][j];
    }
    T const& operator()(int j) const {
        return _rows[0][j];
    }
    T* operator[](int i) {
        return _rows[i];
    }
    T const* operator[](int i) const {
        return _rows[i];
    }

    int getN() const { return _n; }
    int getM() const { return _m; }
private:
    int _n, _m;
    std::vector<T*>& _rows;
    T *_data;
};

/************************************************************************************************************/

namespace {
    void result(std::string const& msg) {
        printf("%-50s\n", msg.c_str());
    }
    void result(std::string const& msg, double t) {
        printf("%-50s %7.4fs\n", msg.c_str(), t);
    }
    void result(std::string const& msg, double t, double v1, double v2) {
        printf("%-50s %7.4fs (%10.4g ... %10.4g)\n", msg.c_str(), t, v1, v2);
    }

    //
    // boost::ublas
    //
    void test_set_ublas(boost::timer &t,
                        RHLMatrix<double> &in,
                        int const Niter) {
        int const N = in.getN();
        assert(N == in.getM());

        ublas::matrix<double> m1(N, N);

        t.restart();
        for (int n = 0; n != Niter; ++n) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    m1(i, j) = in[i][j];
                }
            }
        }

        result("Boost::ublas fill", t.elapsed());
    }
    //
    // VW
    //
    void test_set_vw(boost::timer& t,
                     RHLMatrix<double> &in,
                     int const Niter) {
        int const N = in.getN();
        assert(N == in.getM());
        vw::Matrix<double> m1(N, N);

        t.restart();
        for (int n = 0; n != Niter; ++n) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    m1(i, j) = in[i][j];
                }
            }
        }

        result("VW matrix fill", t.elapsed());
    }

    void test_lsq_vw(boost::timer& t,
                     RHLMatrix<double> &min,
                     RHLMatrix<double> &bin
                    ) {
        int const N = min.getN(), M = min.getM();
        assert (N == M);
                     
        vw::Matrix<double> m1(N, N);
        vw::Vector<double> v1(N);

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                m1(i, j) = min[i][j];
            }
        }
        for (int i = 0; i < N; ++i) {
            v1(i) = bin[0][i];
        }
        
        t.restart();

        /* use vw's internal least squares mechanism that uses SVD */
        vw::math::Vector<double> vs1 = vw::math::least_squares(m1, v1);
        result("VW least squared 1", t.elapsed(), vs1[0], vs1[N-1]);
        
        /* explicitly use pseudoinverse, which also uses SVD */
        t.restart();
        vw::math::Matrix<double> m1t = vw::math::pseudoinverse(m1);
        vw::math::Vector<double> vs2  = m1t*v1;
        result("VW least squared 2", t.elapsed(), vs2[0], vs2[N-1]);
    }
    //
    // GSL
    //
    void test_set_gsl(boost::timer &t,
                      RHLMatrix<double> &in,
                      int const Niter
                     ) {
        int const N = in.getN();
        assert(N == in.getM());

        gsl_matrix *gm1 = gsl_matrix_alloc(N, N);

        t.restart();
        for (int n = 0; n != Niter; ++n) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    gsl_matrix_set(gm1, i, j, in[i][j]);
                }
            }
        }
        result("GSL matrix fill 1", t.elapsed());

        gsl_matrix_set_zero(gm1);
        t.restart();
        for (int n = 0; n != Niter; ++n) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    *gsl_matrix_ptr(gm1, i, j) = in[i][j];
                }
            }
        }
        result("GSL matrix fill 2", t.elapsed());

        /* Stride over columns */
        gsl_matrix_set_zero(gm1);
        t.restart();
        for (int n = 0; n != Niter; ++n) {
            for (int i = 0; i < N; ++i) {
                gsl_vector_view row1 = gsl_matrix_row(gm1, i);
                for (int j = 0; j < N; ++j) {
                    *gsl_vector_ptr(&row1.vector, j) = in[i][j];
                }
            }
        }
        result("GSL matrix fill 3", t.elapsed());

        /* Stride over rows */
        gsl_matrix_set_zero(gm1);
        t.restart();
        for (int n = 0; n != Niter; ++n) {
            for (int i = 0; i < N; ++i) {
                gsl_vector_view col1 = gsl_matrix_column(gm1, i);
                for (int j = 0; j < N; ++j) {
                    *gsl_vector_ptr(&col1.vector, j) = in[i][j];
                }
            }
        }
        result("GSL matrix fill 4", t.elapsed());
    }
    
    void test_lsq_gsl(boost::timer& t,
                      RHLMatrix<double> &min,
                      RHLMatrix<double> &bin
                     ) {
        int const N = min.getN(), M = min.getM();
        assert (N == M);

        gsl_matrix *gm1 = gsl_matrix_alloc(N, N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                gsl_matrix_set(gm1, i, j, min[i][j]);
            }
        }
        
        gsl_vector *v1 = gsl_vector_alloc(N);
        for (int i = 0; i < N; ++i) {
            gsl_vector_set(v1, i, bin[0][i]);
        }

        gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc (N, N);
        gsl_vector *gs1                     = gsl_vector_alloc (N);
        gsl_matrix *gc1                     = gsl_matrix_alloc (N, N);
        double chi2;
        size_t rank;
        t.restart();
        gsl_multifit_linear_svd(gm1, v1, GSL_DBL_EPSILON, &rank, gs1, gc1, &chi2, work);
        result("GSL least squared 1", t.elapsed(), gsl_vector_get(gs1, 0), gsl_vector_get(gs1, N-1));

        /* only use the GSL parts necessary to get the solution */
        gsl_vector *gs2 = gsl_vector_alloc (N);
        t.restart();
        gsl_matrix *A   = work->A;
        gsl_matrix *Q   = work->Q;
        gsl_matrix *QSI = work->QSI;
        gsl_vector *S   = work->S;
        gsl_vector *xt  = work->xt;
        gsl_vector *D   = work->D;
        gsl_matrix_memcpy (A, gm1);
        gsl_linalg_balance_columns (A, D);
        gsl_linalg_SV_decomp_mod (A, QSI, Q, S, xt);
        gsl_blas_dgemv (CblasTrans, 1.0, A, v1, 0.0, xt);
        gsl_matrix_memcpy (QSI, Q);
        {
            double alpha0 = gsl_vector_get (S, 0);
            size_t p_eff = 0;

            const size_t p = gm1->size2;

            for (size_t j = 0; j < p; j++)
            {
                gsl_vector_view column = gsl_matrix_column (QSI, j);
                double alpha = gsl_vector_get (S, j);
            
                if (alpha <= GSL_DBL_EPSILON * alpha0) {
                    alpha = 0.0;
                } else {
                    alpha = 1.0 / alpha;
                    p_eff++;
                }
            
                gsl_vector_scale (&column.vector, alpha);
            }
        
            rank = p_eff;
        }
        gsl_vector_set_zero (gs2);
        gsl_blas_dgemv (CblasNoTrans, 1.0, QSI, xt, 0.0, gs2);
        gsl_vector_div (gs2, D);
        result("GSL least squared 2", t.elapsed() ,gsl_vector_get(gs2, 0), gsl_vector_get(gs2, N-1));
    }
    //
    // Eigen
    //
    void test_set_eigen(boost::timer &t,
                        RHLMatrix<double> &in,                        
                        int const Niter) {
        int const N = in.getN();
        assert(N == in.getM());

        Eigen::MatrixXd m1(N, N);

        t.restart();
        for (int n = 0; n != Niter; ++n) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    m1(i, j) = in[i][j];
                }
            }
        }
        result("Eigen matrix fill", t.elapsed());

        t.restart();
        for (int n = 0; n != Niter; ++n) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    m1.coeffRef(i, j) = in[i][j];
                }
            }
        }

        result("Eigen matrix fill (no bounds check)", t.elapsed());

        t.restart();
        for (int n = 0; n != Niter; ++n) {
            for (int i = 0; i < N; ++i) {
                int index = i*N;
                for (int j = 0; j < N; ++j) {
                    m1(index + j) = in[i][j];
                }
            }
        }

        result("Eigen matrix fill (per row)", t.elapsed());
        t.restart();
        for (int n = 0; n != Niter; ++n) {
            for (int i = 0; i < N; ++i) {
                int index = i*N;
                for (int j = 0; j < N; ++j) {
                    m1.coeffRef(index + j) = in[i][j];
                }
            }
        }

        result("Eigen matrix fill (per row, no bounds check)", t.elapsed());
    }

    void test_lsq_eigen(boost::timer& t,
                        RHLMatrix<double> &min,
                        RHLMatrix<double> &bin
                       ) {
        int const N = min.getN(), M = min.getM();
        assert (N == M);

        Eigen::MatrixXd m1(N, N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                m1(i, j) = min[i][j];
            }
        }

        Eigen::VectorXd v1(N);
        for (int i = 0; i != N; ++i) {
            v1(i) = bin[0][i];
        }
        
        Eigen::VectorXd b;
        // LU
        t.restart();
        if (m1.lu().solve(v1, &b)) {
            result("Eigen LU", t.elapsed(), b[0], b[N-1]);
        } else {
            result("Eigen failed to solve by LU");
        }
        // Cholesky
        t.restart();
        if (m1.llt().solve(v1, &b)) {
            result("Eigen Cholesky LL^T", t.elapsed(), b[0], b[N-1]);
        } else {
            result("Eigen failed to solve by Cholesky LL^T");
        }
        // Cholesky (no sqrt)
        t.restart();
        if (m1.ldlt().solve(v1, &b)) {
            result("Eigen Cholesky LDL^T", t.elapsed(), b[0], b[N-1]);
        } else {
            result("Eigen failed to solve by Cholesky LDL^T");
        }
        // Eigenvalues/vectors
        t.restart();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eVecValues(m1);
        Eigen::MatrixXd const& R = eVecValues.eigenvectors();
        Eigen::VectorXd eValues = eVecValues.eigenvalues();

        for (int i = 0; i != eValues.rows(); ++i) {
            if (eValues(i) != 0.0) {
                eValues(i) = 1.0/eValues(i);
            }
        }

        b = R*eValues.asDiagonal()*R.transpose()*v1;
        result("Eigen eigen-values", t.elapsed(), b[0], b[N-1]);
        // SVD
        t.restart();
        Eigen::SVD<Eigen::MatrixXd> svdOfM1(m1);
        svdOfM1.solve(v1, &b);
        result("Eigen SVD", t.elapsed(), b[0], b[N-1]);
    }
}

int main(int argc, char** argv) {
    if (argc <= 1) {
        cerr << "Usage: " << argv[0] << " N [niter]" << endl;
        cerr << "Default: niter = 1000" << endl;
        return 1;
    }

    int const N = atoi(argv[1]);
    int const Niter = (argc <= 2) ? 10 : atoi(argv[2]);

    boost::timer t;                     // Boost timing
    /* 
     * First test, fill and increment elements of a matrix 
     *
     * Start by initialising the matrix;  we'll use our own
     * class for this
     */
    RHLMatrix<double> A(N, N);
    {
        srand(12345);
        RHLMatrix<double> sqrtA(N, N);  // we'll square this matrix to get a postitive definite matrix
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                sqrtA(i, j) = ((N*i + j) + rand())/(N*N);
            }
        }

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double a = 0.0;
                for (int k = 0; k != N; ++k) {
                    a += sqrtA(i, k)*sqrtA(j, k);
                }
                A(i, j) = a/N;
            }
        }
    }

    RHLMatrix<double> b(N);
    for (int i = 0; i < N; ++i) {
        b(i) = N*(i + rand());
    }

    // boost::ublas
#if 1
    test_set_ublas(t, A, Niter);
#endif

    // vw
    test_set_vw(t, A, Niter);
    
    // GSL
#define TEST_GSL 1
#if TEST_GSL
    test_set_gsl(t, A, Niter);
#endif

    // Eigen
    test_set_eigen(t, A, Niter);
    /* 
     * Second test, find linear algebra solution using PCA
     */
    cout << endl;

    test_lsq_vw(t, A, b);
#if TEST_GSL
    test_lsq_gsl(t, A, b);
#endif
    test_lsq_eigen(t, A, b);
   
    return 0;
}
