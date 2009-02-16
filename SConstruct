# -*- python -*-
#
# Setup our environment
#
import glob, os.path
import lsst.SConsUtils as scons

dependencies = [
    ["boost", "boost/numeric/ublas/blas.hpp"],
    ]

if os.uname()[0] == 'Darwin':           # already has blas
    dependencies += [
        ["gsl", "gsl/gsl_version.h", "gsl"],
        ]
else:
    blas = "gslcblas"                   # assuming that we build gsl using gslcblas
        
    dependencies += [
        ["gsl", "gsl/gsl_version.h", blas],
        ["gsl", "gsl/gsl_version.h", "%s gsl" % blas],
        ]
    
dependencies += [
    ["lapack", None, "lapack", "dgesdd_"],
    ["vw", "vw/Core.h", "vwCore:C++"],
    ["vw", "vw/Core.h", "vw:C++"],
    ["eigen", "Eigen/Core.h"],
    ]

env = scons.makeEnv(
    "matrix",
    r"$HeadURL: svn+ssh://svn.lsstcorp.org/DMS/gil/trunk/SConstruct $",
    dependencies
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
