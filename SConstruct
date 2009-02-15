# -*- python -*-
#
# Setup our environment
#
import glob, os.path
import lsst.SConsUtils as scons

env = scons.makeEnv(
    "matrix",
    r"$HeadURL: svn+ssh://svn.lsstcorp.org/DMS/gil/trunk/SConstruct $",
    [["boost", "boost/numeric/ublas/blas.hpp"],
     ["gsl", "gsl/gsl_version.h", "gsl"],
     #["gsl", "gsl/gsl_version.h", "gslcblas"],
     ["lapack", None, "lapack", "dgesdd_"],
     #["lapack", None, "BLAS", "ilaenv_"],
     ["vw", "vw/Core.h", "vwCore:C++"],
     ["vw", "vw/Core.h", "vw:C++"],
     ["eigen", "Eigen/Core.h"],
    ],
)
#
# Libraries needed to link libraries/executables
#
env.libs["matrix"] += env.getlibs("boost gsl vw lapack")

#
# Build/install things
#
for d in (
    ".",
    "tests",
):
    if d != ".":
        SConscript(os.path.join(d, "SConscript"))
    Clean(d, Glob(os.path.join(d, "*~")))
    Clean(d, Glob(os.path.join(d, "*.pyc")))
