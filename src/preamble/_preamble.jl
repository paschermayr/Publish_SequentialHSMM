################################################################################
#plotting defaults
plot_default_color = :rainbow_bgyrm_35_85_c71_n256
plot_default_size = (1000,1000)
_fontsize = 16
_axissize = 16

################################################################################
#constants
import BaytesCore: ByRows, ByCols
n = 1000 #number of simulated data points
latent_type = Int32
cvg = 1.0 #pf coverage
_rng = Random.Xoshiro(1)

_N1 = 500 #Initial data points for SMC run
_Npart = 8*12 #SMC particles
_TuningIter = 400 #SMC tuning iterations

################################################################################
#Load prepared data in real domain
include("_data.jl")

################################################################################
# Define all models and load corresponding functions
include(string("models/models.jl"))
