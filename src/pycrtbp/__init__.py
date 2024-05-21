#############################################################
# LOADING PACKAGE
#############################################################
from pycrtbp.version import *

#############################################################
# SYSTEM UTILS
#############################################################
import os
try:
    FILE=__file__
    ROOTDIR=os.path.abspath(os.path.dirname(FILE))
except:
    FILE=""
    ROOTDIR=os.path.abspath('')

#############################################################
# PACKAGES MODULES
#############################################################
from pycrtbp.particles import *
from pycrtbp.system import *
from pycrtbp.periodicorbits import *

#############################################################
# EXECUTE
#############################################################
print(f"Running pycrtbp {version} from {ROOTDIR}")
