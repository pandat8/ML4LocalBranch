import ecole
import numpy as np
import pyscipopt
from mllocalbranch import MlLocalbranch
from utility import instancetypes, instancesizes, incumbent_modes, lbconstraint_modes

# instance_type = instancetypes[1]
instance_size = instancesizes[0]
# incumbent_mode = 'firstsol'
lbconstraint_mode = 'symmetric'
samples_time_limit = 3
node_time_limit = 10

total_time_limit = 60
reset_k_at_2nditeration = True


for i in range(5, 6):
    for j in range(0, 1):
        instance_type = instancetypes[i]
        instance_size = instancesizes[j]

        if instance_type == instancetypes[0]:
            lbconstraint_mode = 'asymmetric'
        else:
            lbconstraint_mode = 'symmetric'

        print(instance_type + instance_size)
        print(lbconstraint_mode)

        mllb = MlLocalbranch(instance_type, instance_size, lbconstraint_mode, seed=100)
        mllb.generate_instances_miplib2017_binary()
