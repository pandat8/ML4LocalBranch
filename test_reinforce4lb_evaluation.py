import ecole
import numpy as np
import pyscipopt
from mllocalbranch_fromfiles import RlLocalbranch
from utility import instancetypes, instancesizes, incumbent_modes, lbconstraint_modes
import torch
import random

seed = 100
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


# instance_type = instancetypes[1]
instance_size = instancesizes[1]
# incumbent_mode = 'firstsol'
lbconstraint_mode = 'symmetric'
samples_time_limit = 3

total_time_limit = 60
node_time_limit = 10

reset_k_at_2nditeration = True
use_checkpoint = True
# lr_list = [0.01] # 0.1, 0.05, 0.01, 0.001,0.0001,1e-5, 1e-6,1e-8
# eps_list = [0, 0.02]
epsilon = 0.0
lr = 0.01
# for lr in lr_list:
#     print('learning rate = ', lr)
#     print('epsilon = ', epsilon)
for i in range(0, 3, 2):
    instance_type = instancetypes[i]
    if instance_type == instancetypes[0]:
        lbconstraint_mode = 'asymmetric'
    else:
        lbconstraint_mode = 'symmetric'
    for j in range(1, 2):
        incumbent_mode = incumbent_modes[j]

        print(instance_type + instance_size)
        print(incumbent_mode)
        print(lbconstraint_mode)

        reinforce_localbranch = RlLocalbranch(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=seed)

        # reinforce_localbranch.train_agent(train_instance_size='-small', total_time_limit=total_time_limit,
        #                                   node_time_limit=node_time_limit, reset_k_at_2nditeration=reset_k_at_2nditeration,
        #                                   lr=lr, n_epochs=100, epsilon=epsilon, use_checkpoint=use_checkpoint)

        # reinforce_localbranch.evaluate_localbranching(evaluation_instance_size=instance_size, total_time_limit=total_time_limit, node_time_limit=node_time_limit, reset_k_at_2nditeration=reset_k_at_2nditeration)

        # reinforce_localbranch.evaluate_localbranching_rlactive(
        #     evaluation_instance_size=instance_size,
        #     total_time_limit=total_time_limit,
        #     node_time_limit=node_time_limit,
        #     reset_k_at_2nditeration=reset_k_at_2nditeration,
        #     lr=lr
        #                                                        )

        reinforce_localbranch.primal_integral(test_instance_size=instance_size, total_time_limit=total_time_limit, node_time_limit=node_time_limit)
        # regression_init_k.solve2opt_evaluation(test_instance_size='-small')
