import pyscipopt
from pyscipopt import Model
import ecole
import numpy
import pathlib
import matplotlib.pyplot as plt
from geco.mips.miplib.base import Loader
from improvelp import improvelp
import gzip
import pickle
from utility import instancetypes, generator_switcher, instancesizes

# def generator_switcher(instancetype):
#     switcher = {
#         instancetypes[0]: lambda : ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000, density=0.05),
#         instancetypes[1]: lambda : ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=100),
#         instancetypes[2]: lambda : ecole.instance.IndependentSetGenerator(n_nodes=1000),
#         instancetypes[3]: lambda : ecole.instance.CombinatorialAuctionGenerator(n_items=300, n_bids=300),
#         instancetypes[4]: lambda: ecole.instance.SetCoverGenerator(n_rows=1000, n_cols=2000, density=0.05),
#         instancetypes[5]: lambda : ecole.instance.SetCoverGenerator(n_rows=2000, n_cols=4000, density=0.05),
#         instancetypes[6]: lambda: ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=200, n_facilities=200),
#         instancetypes[7]: lambda: ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=400, n_facilities=400),
#         instancetypes[8]: lambda: ecole.instance.IndependentSetGenerator(n_nodes=2000),
#         instancetypes[9]: lambda: ecole.instance.IndependentSetGenerator(n_nodes=4000),
#     }
#     return switcher.get(instancetype, lambda : "invalide argument")()

# instancetypes = ['setcovering', 'capacitedfacility', 'independentset', 'combinatorialauction','setcovering-row1000col2000', 'setcovering-row2000col4000', 'capacitedfacility-c200-f200', 'capacitedfacility-c400-f400', 'independentset-n2000', 'independentset-n4000']
# modes = ['improve-supportbinvars', 'improve-binvars']
# instancetype = instancetypes[2]


instance_size = instancesizes[0]

for t in range(0, 3):
    instance_type = instancetypes[t]

    dataset = instance_type + instance_size
    directory_opt = './result/generated_instances/' + instance_type + '/' + instance_size + '/' + 'opt_solution' + '/'
    pathlib.Path(directory_opt).mkdir(parents=True, exist_ok=True)

    generator = generator_switcher(dataset)
    generator.seed(100)

    for i in range(100, 200):
        instance = next(generator)
        MIP_model = instance.as_pyscipopt()
        MIP_model.setProbName(instance_type + instance_size + '-' + str(i))
        instance_name = MIP_model.getProbName()
        # if 13 < i:
        MIP_model.setParam('presolving/maxrounds', 0)
        MIP_model.setParam('presolving/maxrestarts', 0)
        MIP_model.setParam("display/verblevel", 0)
        MIP_model.optimize()
        status = MIP_model.getStatus()
        if status == 'optimal':
            obj = MIP_model.getObjVal()
            time = MIP_model.getSolvingTime()
            data = [obj, time]

            filename = f'{directory_opt}{instance_name}-optimal-obj-time.pkl'
            with gzip.open(filename, 'wb') as f:
                pickle.dump(data, f)

        print("instance:", MIP_model.getProbName(),
              "status:", MIP_model.getStatus(),
              "best obj: ", MIP_model.getObjVal(),
              "solving time: ", MIP_model.getSolvingTime())

        MIP_model.freeProb()
        del MIP_model
        del instance







