import pyscipopt
from pyscipopt import Model
import ecole
import numpy as np
import random
import pathlib
import gzip
import pickle
import json
import matplotlib.pyplot as plt
from geco.mips.loading.miplib import Loader
from utility import lbconstraint_modes, instancetypes, instancesizes, generator_switcher, binary_support, copy_sol, mean_filter,mean_forward_filter, imitation_accuracy, haming_distance_solutions, haming_distance_solutions_asym
from localbranching import addLBConstraint, addLBConstraintAsymmetric
from ecole_extend.environment_extend import SimpleConfiguring, SimpleConfiguringEnablecuts, SimpleConfiguringEnableheuristics
from models import GraphDataset, GNNPolicy, BipartiteNodeData
import torch.nn.functional as F
import torch_geometric
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from scipy.interpolate import interp1d

from localbranching import LocalBranching

import gc
import sys
from memory_profiler import profile

from models_rl import SimplePolicy, ImitationLbDataset, AgentReinforce

class MlLocalbranch:
    def __init__(self, instance_type, instance_size, lbconstraint_mode, incumbent_mode='firstsol', seed=100):
        self.instance_type = instance_type
        self.instance_size = instance_size
        self.incumbent_mode = incumbent_mode
        self.lbconstraint_mode = lbconstraint_mode
        self.seed = seed
        self.directory = './result/generated_instances/' + self.instance_type + '/' + self.instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # self.generator = generator_switcher(self.instance_type + self.instance_size)

        self.initialize_ecole_env()

        self.env.seed(self.seed)  # environment (SCIP)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

    def initialize_ecole_env(self):

        if self.incumbent_mode == 'firstsol':

            self.env = ecole.environment.Configuring(

                # set up a few SCIP parameters
                scip_params={
                    "presolving/maxrounds": 0,  # deactivate presolving
                    "presolving/maxrestarts": 0,
                },

                observation_function=ecole.observation.MilpBipartite(),

                reward_function=None,

                # collect additional metrics for information purposes
                information_function={
                    'time': ecole.reward.SolvingTime().cumsum(),
                }
            )

        elif self.incumbent_mode == 'rootsol':

            if self.instance_type == 'independentset':
                self.env = SimpleConfiguring(

                    # set up a few SCIP parameters
                    scip_params={
                        "presolving/maxrounds": 0,  # deactivate presolving
                        "presolving/maxrestarts": 0,
                    },

                    observation_function=ecole.observation.MilpBipartite(),

                    reward_function=None,

                    # collect additional metrics for information purposes
                    information_function={
                        'time': ecole.reward.SolvingTime().cumsum(),
                    }
                )
            else:
                self.env = SimpleConfiguringEnablecuts(

                    # set up a few SCIP parameters
                    scip_params={
                        "presolving/maxrounds": 0,  # deactivate presolving
                        "presolving/maxrestarts": 0,
                    },

                    observation_function=ecole.observation.MilpBipartite(),

                    reward_function=None,

                    # collect additional metrics for information purposes
                    information_function={
                        'time': ecole.reward.SolvingTime().cumsum(),
                    }
                )
            # elif self.instance_type == 'capacitedfacility':
            #     self.env = SimpleConfiguringEnableheuristics(
            #
            #         # set up a few SCIP parameters
            #         scip_params={
            #             "presolving/maxrounds": 0,  # deactivate presolving
            #             "presolving/maxrestarts": 0,
            #         },
            #
            #         observation_function=ecole.observation.MilpBipartite(),
            #
            #         reward_function=None,
            #
            #         # collect additional metrics for information purposes
            #         information_function={
            #             'time': ecole.reward.SolvingTime().cumsum(),
            #         }
            #     )

    def set_and_optimize_MIP(self, MIP_model, incumbent_mode):

        preprocess_off = True
        if incumbent_mode == 'firstsol':
            heuristics_off = False
            cuts_off = False
        elif incumbent_mode == 'rootsol':
            if self.instance_type == 'independentset':
                heuristics_off = True
                cuts_off = True
            elif self.instance_type == 'combinatorialauction':
                heuristics_off = True
                cuts_off = True
            elif self.instance_type == 'generalized_independentset':
                heuristics_off = True
                cuts_off = True
            elif self.instance_type == 'miplib2017_binary':
                heuristics_off = True
                cuts_off = True
            else:
                heuristics_off = True
                cuts_off = False
            # elif self.instance_type == 'capacitedfacility':
            #     heuristics_off = False
            #     cuts_off = True

        if preprocess_off:
            MIP_model.setParam('presolving/maxrounds', 0)
            MIP_model.setParam('presolving/maxrestarts', 0)

        if heuristics_off:
            MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
        # else:
        #     MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.FAST)

        if cuts_off:
            MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
        # else:
        #     MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.FAST)

        if incumbent_mode == 'firstsol':
            MIP_model.setParam('limits/solutions', 1)
        elif incumbent_mode == 'rootsol':
            MIP_model.setParam("limits/nodes", 1)

        MIP_model.setParam("display/verblevel", 0)
        MIP_model.optimize()

        t = MIP_model.getSolvingTime()
        status = MIP_model.getStatus()
        lp_status = MIP_model.getLPSolstat()
        stage = MIP_model.getStage()
        n_sols = MIP_model.getNSols()
        if n_sols == 0:
            MIP_model.freeTransform()
            MIP_model.resetParams()
            if preprocess_off:
                MIP_model.setParam('presolving/maxrounds', 0)
                MIP_model.setParam('presolving/maxrestarts', 0)
            # MIP_model.setParam("limits/nodes", -1)
            MIP_model.setParam('limits/solutions', 10)
            MIP_model.setParam("display/verblevel", 0)
            MIP_model.optimize()
            n_sols = MIP_model.getNSols()
            t = MIP_model.getSolvingTime()
            status = MIP_model.getStatus()


        print("* Model status: %s" % status)
        # print("* LP status: %s" % lp_status)
        # print("* Solve stage: %s" % stage)
        print("* Solving time: %s" % t)
        print('* number of sol : ', n_sols)

        incumbent_solution = MIP_model.getBestSol()
        feasible = MIP_model.checkSol(solution=incumbent_solution)

        return status, feasible, MIP_model, incumbent_solution

    def initialize_MIP(self, MIP_model):

        MIP_model_2, MIP_2_vars, success = MIP_model.createCopy(origcopy=True)

        incumbent_mode = self.incumbent_mode
        if self.incumbent_mode == 'firstsol':
            incumbent_mode_2 = 'rootsol'
        elif self.incumbent_mode == 'rootsol':
            incumbent_mode_2 = 'firstsol'

        status, feasible, MIP_model, incumbent_solution = self.set_and_optimize_MIP(MIP_model, incumbent_mode)
        status_2, feasible_2, MIP_model_2, incumbent_solution_2 = self.set_and_optimize_MIP(MIP_model_2,
                                                                                            incumbent_mode_2)

        feasible = (feasible and feasible_2)

        if (not status == 'optimal') and (not status_2 == 'optimal'):
            not_optimal = True
        else:
            not_optimal = False

        if not_optimal and feasible:
            valid = True
        else:
            valid = False

        return valid, MIP_model, incumbent_solution

    def generate_instances(self, instance_type, instance_size):

        directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
        directory_transformedmodel = directory + 'transformedmodel' + '/'
        directory_firstsol = directory +'firstsol' + '/'
        directory_rootsol = directory + 'rootsol' + '/'
        pathlib.Path(directory_transformedmodel).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_firstsol).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_rootsol).mkdir(parents=True, exist_ok=True)

        generator = generator_switcher(instance_type + instance_size)
        generator.seed(self.seed)

        index_instance = 0
        while index_instance < 200: # 200
            instance = next(generator)
            MIP_model = instance.as_pyscipopt()
            MIP_model.setProbName(instance_type + '-' + str(index_instance))
            instance_name = MIP_model.getProbName()
            print('\n')
            print(instance_name)

            # initialize MIP
            MIP_model_2, MIP_2_vars, success = MIP_model.createCopy(
                problemName='Baseline', origcopy=True)

            # MIP_model_orig, MIP_vars_orig, success = MIP_model.createCopy(
            #     problemName='Baseline', origcopy=True)

            incumbent_mode = 'firstsol'
            incumbent_mode_2 = 'rootsol'

            status, feasible, MIP_model, incumbent_solution = self.set_and_optimize_MIP(MIP_model, incumbent_mode)

            status_2, feasible_2, MIP_model_2, incumbent_solution_2 = self.set_and_optimize_MIP(MIP_model_2,
                                                                                                incumbent_mode_2)

            feasible = feasible and feasible_2

            if (not status == 'optimal') and (not status_2 == 'optimal'):
                not_optimal = True
            else:
                not_optimal = False

            if not_optimal and feasible:
                valid = True
            else:
                valid = False

            if valid:

                MIP_model.resetParams()
                MIP_model_transformed, MIP_copy_vars, success = MIP_model.createCopy(
                    problemName='transformed', origcopy=False)
                MIP_model_transformed, sol_MIP_first = copy_sol(MIP_model, MIP_model_transformed, incumbent_solution,
                                                        MIP_copy_vars)
                MIP_model_transformed, sol_MIP_root = copy_sol(MIP_model_2, MIP_model_transformed, incumbent_solution_2,
                                                        MIP_copy_vars)

                transformed_model_name = MIP_model_transformed.getProbName()
                filename = f'{directory_transformedmodel}{transformed_model_name}.cip'
                MIP_model_transformed.writeProblem(filename=filename, trans=False)

                firstsol_filename = f'{directory_firstsol}firstsol-{transformed_model_name}.sol'
                MIP_model_transformed.writeSol(solution=sol_MIP_first, filename=firstsol_filename)

                rootsol_filename = f'{directory_rootsol}rootsol-{transformed_model_name}.sol'
                MIP_model_transformed.writeSol(solution=sol_MIP_root, filename=rootsol_filename)

                model = Model()
                model.readProblem(filename)
                sol = model.readSolFile(rootsol_filename)

                feas = model.checkSol(sol)
                if not feas:
                    print('the root solution of '+ model.getProbName()+ 'is not feasible!')

                model.addSol(sol, False)
                print(model.getSolObjVal(sol))
                instance = ecole.scip.Model.from_pyscipopt(model)
                scipMIP = instance.as_pyscipopt()
                sol2 = scipMIP.getBestSol()
                print(scipMIP.getSolObjVal(sol2))

                # MIP_model_2.resetParams()
                # MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model_2.createCopy(
                #     problemName='rootsol',
                #     origcopy=False)
                # MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model_2, MIP_model_copy2, incumbent_solution_2,
                #                                           MIP_copy_vars2)

                MIP_model.freeProb()
                MIP_model_2.freeProb()
                MIP_model_transformed.freeProb()
                model.freeProb()
                del MIP_model
                del MIP_model_2
                del MIP_model_transformed
                del model

                index_instance += 1

    def generate_instances_generalized_independentset(self, instance_type='generalized_independentset', instance_size='-small'):

        directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
        directory_transformedmodel = directory + 'transformedmodel' + '/'
        directory_firstsol = directory +'firstsol' + '/'
        directory_rootsol = directory + 'rootsol' + '/'
        pathlib.Path(directory_transformedmodel).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_firstsol).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_rootsol).mkdir(parents=True, exist_ok=True)

        instance_directory = './data/generated_instances/generalized_independentset/original_lp_instances/'
        filename = '*.lp'
        # print(filename)
        sample_files = [str(path) for path in pathlib.Path(instance_directory).glob(filename)]

        index_instance = 0
        for i, instance in enumerate(sample_files):
            print(instance)
            MIP_model = Model()
            MIP_model.readProblem(instance)

            MIP_model.setProbName(instance_type + '-' + str(index_instance))
            instance_name = MIP_model.getProbName()
            print('\n')
            print(instance_name)
            print('Number of variables', MIP_model.getNVars())
            print('Number of binary variables', MIP_model.getNBinVars())

            # initialize MIP
            MIP_model_2, MIP_2_vars, success = MIP_model.createCopy(
                problemName='Baseline', origcopy=True)

            # MIP_model_orig, MIP_vars_orig, success = MIP_model.createCopy(
            #     problemName='Baseline', origcopy=True)

            incumbent_mode = 'firstsol'
            incumbent_mode_2 = 'rootsol'

            status, feasible, MIP_model, incumbent_solution = self.set_and_optimize_MIP(MIP_model, incumbent_mode)

            status_2, feasible_2, MIP_model_2, incumbent_solution_2 = self.set_and_optimize_MIP(MIP_model_2,
                                                                                                incumbent_mode_2)

            feasible = feasible and feasible_2

            if (not status == 'optimal') and (not status_2 == 'optimal'):
                not_optimal = True
            else:
                not_optimal = False

            if not_optimal and feasible:
                valid = True
            else:
                valid = False

            if valid:

                MIP_model.resetParams()
                MIP_model_transformed, MIP_copy_vars, success = MIP_model.createCopy(
                    problemName='transformed', origcopy=False)
                MIP_model_transformed, sol_MIP_first = copy_sol(MIP_model, MIP_model_transformed, incumbent_solution,
                                                        MIP_copy_vars)
                MIP_model_transformed, sol_MIP_root = copy_sol(MIP_model_2, MIP_model_transformed, incumbent_solution_2,
                                                        MIP_copy_vars)

                transformed_model_name = MIP_model_transformed.getProbName()
                filename = f'{directory_transformedmodel}{transformed_model_name}.cip'
                MIP_model_transformed.writeProblem(filename=filename, trans=False)

                firstsol_filename = f'{directory_firstsol}firstsol-{transformed_model_name}.sol'
                MIP_model_transformed.writeSol(solution=sol_MIP_first, filename=firstsol_filename)

                rootsol_filename = f'{directory_rootsol}rootsol-{transformed_model_name}.sol'
                MIP_model_transformed.writeSol(solution=sol_MIP_root, filename=rootsol_filename)

                model = Model()
                model.readProblem(filename)
                sol = model.readSolFile(rootsol_filename)

                feas = model.checkSol(sol)
                if not feas:
                    print('the root solution of '+ model.getProbName()+ 'is not feasible!')

                model.addSol(sol, False)
                print(model.getSolObjVal(sol))
                instance = ecole.scip.Model.from_pyscipopt(model)
                scipMIP = instance.as_pyscipopt()
                sol2 = scipMIP.getBestSol()
                print(scipMIP.getSolObjVal(sol2))

                # MIP_model_2.resetParams()
                # MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model_2.createCopy(
                #     problemName='rootsol',
                #     origcopy=False)
                # MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model_2, MIP_model_copy2, incumbent_solution_2,
                #                                           MIP_copy_vars2)

                MIP_model.freeProb()
                MIP_model_2.freeProb()
                MIP_model_transformed.freeProb()
                model.freeProb()
                del MIP_model
                del MIP_model_2
                del MIP_model_transformed
                del model

                index_instance += 1

    def generate_instances_miplib2017_binary(self, instance_type='miplib2017_binary', instance_size='-small'):

        directory = './data/generated_instances/' + instance_type + '/' + instance_size + '/'
        directory_transformedmodel = directory + 'transformedmodel' + '/'
        directory_firstsol = directory +'firstsol' + '/'
        directory_rootsol = directory + 'rootsol' + '/'
        pathlib.Path(directory_transformedmodel).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_firstsol).mkdir(parents=True, exist_ok=True)
        pathlib.Path(directory_rootsol).mkdir(parents=True, exist_ok=True)

        file_directory = './result/miplib2017/miplib2017_purebinary_solved.txt'
        index_instance = 0
        with open(file_directory) as fp:
            Lines = fp.readlines()
            for line in Lines:

                instance_str = line.strip()
                MIP_model = Loader().load_instance(instance_str)
                original_name = MIP_model.getProbName()
                print(original_name)

                MIP_model.setProbName(instance_type + '-' + str(index_instance))
                instance_name = MIP_model.getProbName()
                print('\n')
                print(instance_name)
                print('Number of variables', MIP_model.getNVars())
                print('Number of binary variables', MIP_model.getNBinVars())

                # initialize MIP
                MIP_model_2, MIP_2_vars, success = MIP_model.createCopy(
                    problemName='Baseline', origcopy=True)

                # MIP_model_orig, MIP_vars_orig, success = MIP_model.createCopy(
                #     problemName='Baseline', origcopy=True)

                incumbent_mode = 'firstsol'
                incumbent_mode_2 = 'rootsol'

                status, feasible, MIP_model, incumbent_solution = self.set_and_optimize_MIP(MIP_model, incumbent_mode)

                status_2, feasible_2, MIP_model_2, incumbent_solution_2 = self.set_and_optimize_MIP(MIP_model_2,
                                                                                                    incumbent_mode_2)

                feasible = feasible and feasible_2

                if (not status == 'optimal') and (not status_2 == 'optimal'):
                    not_optimal = True
                else:
                    not_optimal = False

                if not_optimal and feasible:
                    valid = True
                else:
                    valid = False

                if valid:

                    MIP_model.resetParams()
                    MIP_model_transformed, MIP_copy_vars, success = MIP_model.createCopy(
                        problemName='transformed', origcopy=False)
                    MIP_model_transformed, sol_MIP_first = copy_sol(MIP_model, MIP_model_transformed, incumbent_solution,
                                                            MIP_copy_vars)
                    MIP_model_transformed, sol_MIP_root = copy_sol(MIP_model_2, MIP_model_transformed, incumbent_solution_2,
                                                            MIP_copy_vars)

                    transformed_model_name = MIP_model_transformed.getProbName()
                    MIP_model_transformed.setProbName(transformed_model_name + '_' + original_name)

                    filename = f'{directory_transformedmodel}{transformed_model_name}.cip'
                    MIP_model_transformed.writeProblem(filename=filename, trans=False)

                    firstsol_filename = f'{directory_firstsol}firstsol-{transformed_model_name}.sol'
                    MIP_model_transformed.writeSol(solution=sol_MIP_first, filename=firstsol_filename)

                    rootsol_filename = f'{directory_rootsol}rootsol-{transformed_model_name}.sol'
                    MIP_model_transformed.writeSol(solution=sol_MIP_root, filename=rootsol_filename)

                    model = Model()
                    model.readProblem(filename)
                    sol = model.readSolFile(rootsol_filename)

                    feas = model.checkSol(sol)
                    if not feas:
                        print('the root solution of '+ model.getProbName()+ 'is not feasible!')

                    model.addSol(sol, False)
                    print(model.getSolObjVal(sol))
                    instance = ecole.scip.Model.from_pyscipopt(model)
                    scipMIP = instance.as_pyscipopt()
                    sol2 = scipMIP.getBestSol()
                    print(scipMIP.getSolObjVal(sol2))

                    # MIP_model_2.resetParams()
                    # MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model_2.createCopy(
                    #     problemName='rootsol',
                    #     origcopy=False)
                    # MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model_2, MIP_model_copy2, incumbent_solution_2,
                    #                                           MIP_copy_vars2)

                    MIP_model.freeProb()
                    MIP_model_2.freeProb()
                    MIP_model_transformed.freeProb()
                    model.freeProb()
                    del MIP_model
                    del MIP_model_2
                    del MIP_model_transformed
                    del model

                    index_instance += 1

    def evaluate_lb_per_instance(self, node_time_limit, total_time_limit, index_instance, reset_k_at_2nditeration=False, policy=None,
                             criterion=None, device=None):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        instance = next(self.generator)
        MIP_model = instance.as_pyscipopt()
        MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
        instance_name = MIP_model.getProbName()
        print('\n')
        print(instance_name)

        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        valid, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)
        conti =99
        # if self.incumbent_mode == 'rootsol' and self.instance_type == 'independentset':
        #     conti = 196

        if valid:
            if index_instance > 99 and index_instance > conti:
                gc.collect()
                observation, _, _, done, _ = self.env.reset(instance)
                del observation
                # print(observation)

                if self.incumbent_mode == 'firstsol':
                    action = {'limits/solutions': 1}
                elif self.incumbent_mode == 'rootsol':
                    action = {'limits/nodes': 1}  #
                sample_observation, _, _, done, _ = self.env.step(action)

                # print(sample_observation)
                graph = BipartiteNodeData(sample_observation.constraint_features,
                                          sample_observation.edge_features.indices,
                                          sample_observation.edge_features.values,
                                          sample_observation.variable_features)

                # We must tell pytorch geometric how many nodes there are, for indexing purposes
                graph.num_nodes = sample_observation.constraint_features.shape[0] + \
                                  sample_observation.variable_features.shape[
                                      0]

                # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
                # MIP_model = instance

                # MIP_model.optimize()
                # print("Status:", MIP_model.getStatus())
                # print("best obj: ", MIP_model.getObjVal())
                # print("Solving time: ", MIP_model.getSolvingTime())

                initial_obj = MIP_model.getSolObjVal(incumbent_solution)
                print("Initial obj before LB: {}".format(initial_obj))

                binary_supports = binary_support(MIP_model, incumbent_solution)
                print('binary support: ', binary_supports)

                model_gnn = GNNPolicy()

                model_gnn.load_state_dict(torch.load(
                    self.saved_gnn_directory + 'trained_params_' + self.train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '.pth'))

                # model_gnn.load_state_dict(torch.load(
                #      'trained_params_' + self.instance_type + '.pth'))

                k_model = model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                    graph.variable_features)

                k_pred = k_model.item() * n_binvars
                print('GNN prediction: ', k_model.item())

                if self.is_symmetric == False:
                    k_pred = k_model.item() * binary_supports

                k_pred = np.ceil(k_pred)

                del k_model
                del graph
                del sample_observation
                del model_gnn

                # create a copy of MIP
                MIP_model.resetParams()
                MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
                    problemName='Baseline', origcopy=False)
                MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
                    problemName='GNN',
                    origcopy=False)
                MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
                    problemName='GNN+reset',
                    origcopy=False)

                print('MIP copies are created')

                MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent_solution,
                                                        MIP_copy_vars)
                MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent_solution,
                                                          MIP_copy_vars2)
                MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
                                                          MIP_copy_vars3)

                print('incumbent solution is copied to MIP copies')
                MIP_model.freeProb()
                del MIP_model
                del incumbent_solution

                # sol = MIP_model_copy.getBestSol()
                # initial_obj = MIP_model_copy.getSolObjVal(sol)
                # print("Initial obj before LB: {}".format(initial_obj))

                # # execute local branching baseline heuristic by Fischetti and Lodi
                # lb_model = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy, k=self.k_baseline,
                #                           node_time_limit=node_time_limit,
                #                           total_time_limit=total_time_limit)
                # status, obj_best, elapsed_time, lb_bits, times, objs = lb_model.search_localbranch(is_symmeric=self.is_symmetric,
                #                                                              reset_k_at_2nditeration=False)
                # print("Instance:", MIP_model_copy.getProbName())
                # print("Status of LB: ", status)
                # print("Best obj of LB: ", obj_best)
                # print("Solving time: ", elapsed_time)
                # print('\n')
                #
                # MIP_model_copy.freeProb()
                # del sol_MIP_copy
                # del MIP_model_copy

                # sol = MIP_model_copy2.getBestSol()
                # initial_obj = MIP_model_copy2.getSolObjVal(sol)
                # print("Initial obj before LB: {}".format(initial_obj))

                # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
                lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                                           node_time_limit=node_time_limit,
                                           total_time_limit=total_time_limit)
                status, obj_best, elapsed_time, lb_bits_pred_reset, times_reset_imitation, objs_reset_imitation, loss_instance, accu_instance = lb_model3.mdp_localbranch(
                    is_symmetric=self.is_symmetric,
                    reset_k_at_2nditeration=reset_k_at_2nditeration,
                    policy=policy,
                    optimizer=None,
                    criterion=criterion,
                    device=device
                    )
                print("Instance:", MIP_model_copy3.getProbName())
                print("Status of LB: ", status)
                print("Best obj of LB: ", obj_best)
                print("Solving time: ", elapsed_time)
                print('\n')

                MIP_model_copy3.freeProb()
                del sol_MIP_copy3
                del MIP_model_copy3

                # execute local branching with 1. first k predicted by GNN; 2. for 2nd iteration of lb, continue lb algorithm with no further injection
                lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=k_pred,
                                           node_time_limit=node_time_limit,
                                           total_time_limit=total_time_limit)
                status, obj_best, elapsed_time, lb_bits_pred, times_reset_vanilla, objs_reset_vanilla, _, _ = lb_model2.mdp_localbranch(
                    is_symmetric=self.is_symmetric,
                    reset_k_at_2nditeration=True,
                    policy=None,
                    optimizer=None,
                    criterion=None,
                    device=None
                )

                print("Instance:", MIP_model_copy2.getProbName())
                print("Status of LB: ", status)
                print("Best obj of LB: ", obj_best)
                print("Solving time: ", elapsed_time)
                print('\n')

                MIP_model_copy2.freeProb()
                del sol_MIP_copy2
                del MIP_model_copy2

                data = [objs_reset_vanilla, times_reset_vanilla, objs_reset_imitation, times_reset_imitation]
                filename = f'{self.directory_lb_test}lb-test-{instance_name}.pkl'  # instance 100-199
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)

                del data
                del lb_model2
                del lb_model3

            index_instance += 1
        del instance
        return index_instance

    def evaluate_localbranching(self, test_instance_size='-small', total_time_limit=60, node_time_limit=30, reset_k_at_2nditeration=False):

        self.train_dataset = self.instance_type + self.instance_size
        self.evaluation_dataset = self.instance_type + test_instance_size

        self.generator = generator_switcher(self.evaluation_dataset)
        self.generator.seed(self.seed)

        self.k_baseline = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_gnn_directory = './result/saved_models/'

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/'
        self.directory_lb_test = directory + 'imitation4lb-from-' + self.incumbent_mode + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        rl_policy = SimplePolicy(7, 4)

        rl_policy.load_state_dict(torch.load(
            self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_imitation.pth'))

        criterion = nn.CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        index_instance = 0
        while index_instance < 200:
            index_instance = self.evaluate_lb_per_instance(node_time_limit=node_time_limit, total_time_limit=total_time_limit, index_instance=index_instance, reset_k_at_2nditeration=reset_k_at_2nditeration,
                                                           policy=rl_policy, criterion=criterion, device=device
                                                           )

    def compute_primal_integral(self, times, objs, total_time_limit=60):

        obj_opt = objs.min()
        times = np.append(times, total_time_limit)
        objs = np.append(objs, objs[-1])

        gamma_baseline = np.zeros(len(objs))
        for j in range(len(objs)):
            if objs[j] == 0 and obj_opt == 0:
                gamma_baseline[j] = 0
            elif objs[j] * obj_opt < 0:
                gamma_baseline[j] = 1
            else:
                gamma_baseline[j] = np.abs(objs[j] - obj_opt) / np.maximum(np.abs(objs[j]), np.abs(obj_opt))  #

        # compute the primal gap of last objective
        primal_gap_final = np.abs(objs[-1] - obj_opt) / np.abs(obj_opt)

        # create step line
        stepline = interp1d(times, gamma_baseline, 'previous')


        # compute primal integral
        primal_integral = 0
        for j in range(len(objs) - 1):
            primal_integral += gamma_baseline[j] * (times[j + 1] - times[j])

        return primal_integral, primal_gap_final, stepline


    def primal_integral(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/'
        directory_lb_test = directory + 'imitation4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/' + 'rl/'
            directory_lb_test_2 = directory_2 + 'imitation4lb-from-' +  'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/' + 'rl/'
            directory_lb_test_2 = directory_2 + 'imitation4lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        # primal_int_baselines = []
        primal_int_reset_vanillas = []
        primal_in_reset_imitations = []
        # primal_gap_final_baselines = []
        primal_gap_final_reset_vanillas = []
        primal_gap_final_reset_imitations = []
        # steplines_baseline = []
        steplines_reset_vanillas = []
        steplines_reset_imitations = []

        for i in range(100,200):
            instance_name = self.instance_type + '-' + str(i)  # instance 100-199

            filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'

            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_reset_vanilla, times_reset_vanilla, objs_reset_imitation, times_reset_imitation = data  # objs contains objs of a single instance of a lb test

            filename_2 = f'{directory_lb_test_2}lb-test-{instance_name}.pkl'

            with gzip.open(filename_2, 'rb') as f:
                data = pickle.load(f)
            objs_reset_vanilla_2, times_reset_vanilla_2, objs_reset_imitation_2, times_reset_imitation_2 = data  # objs contains objs of a single instance of a lb test

            a = [objs_reset_vanilla.min(), objs_reset_imitation.min(), objs_reset_vanilla_2.min(), objs_reset_imitation_2.min()]
            # a = [objs.min(), objs_reset_vanilla.min(), objs_reset_imitation.min()]
            obj_opt = np.amin(a)

            # # compute primal gap for baseline localbranching run
            # # if times[-1] < total_time_limit:
            # times = np.append(times, total_time_limit)
            # objs = np.append(objs, objs[-1])
            #
            # gamma_baseline = np.zeros(len(objs))
            # for j in range(len(objs)):
            #     if objs[j] == 0 and obj_opt == 0:
            #         gamma_baseline[j] = 0
            #     elif objs[j] * obj_opt < 0:
            #         gamma_baseline[j] = 1
            #     else:
            #         gamma_baseline[j] = np.abs(objs[j] - obj_opt) / np.maximum(np.abs(objs[j]), np.abs(obj_opt)) #
            #
            # # compute the primal gap of last objective
            # primal_gap_final_baseline = np.abs(objs[-1] - obj_opt) / np.abs(obj_opt)
            # primal_gap_final_baselines.append(primal_gap_final_baseline)
            #
            # # create step line
            # stepline_baseline = interp1d(times, gamma_baseline, 'previous')
            # steplines_baseline.append(stepline_baseline)
            #
            # # compute primal integral
            # primal_int_baseline = 0
            # for j in range(len(objs) - 1):
            #     primal_int_baseline += gamma_baseline[j] * (times[j + 1] - times[j])
            # primal_int_baselines.append(primal_int_baseline)
            #


            # lb-gnn
            # if times_reset_vanilla[-1] < total_time_limit:
            times_reset_vanilla = np.append(times_reset_vanilla, total_time_limit)
            objs_reset_vanilla = np.append(objs_reset_vanilla, objs_reset_vanilla[-1])

            gamma_reset_vanilla = np.zeros(len(objs_reset_vanilla))
            for j in range(len(objs_reset_vanilla)):
                if objs_reset_vanilla[j] == 0 and obj_opt == 0:
                    gamma_reset_vanilla[j] = 0
                elif objs_reset_vanilla[j] * obj_opt < 0:
                    gamma_reset_vanilla[j] = 1
                else:
                    gamma_reset_vanilla[j] = np.abs(objs_reset_vanilla[j] - obj_opt) / np.maximum(np.abs(objs_reset_vanilla[j]), np.abs(obj_opt)) #

            primal_gap_final_vanilla = np.abs(objs_reset_vanilla[-1] - obj_opt) / np.abs(obj_opt)
            primal_gap_final_reset_vanillas.append(primal_gap_final_vanilla)

            stepline_reset_vanilla = interp1d(times_reset_vanilla, gamma_reset_vanilla, 'previous')
            steplines_reset_vanillas.append(stepline_reset_vanilla)

            #
            # t = np.linspace(start=0.0, stop=total_time_limit, num=1001)
            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of primal gap")
            # fig.subplots_adjust(top=0.5)
            # # ax.set_title(instance_name, loc='right')
            # ax.plot(t, stepline_baseline(t), label='lb baseline')
            # ax.plot(t, stepline_reset_vanilla(t), label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()

            # compute primal interal
            primal_int_reset_vanilla = 0
            for j in range(len(objs_reset_vanilla) - 1):
                primal_int_reset_vanilla += gamma_reset_vanilla[j] * (times_reset_vanilla[j + 1] - times_reset_vanilla[j])
            primal_int_reset_vanillas.append(primal_int_reset_vanilla)

            # lb-gnn-reset
            times_reset_imitation = np.append(times_reset_imitation, total_time_limit)
            objs_reset_imitation = np.append(objs_reset_imitation, objs_reset_imitation[-1])

            gamma_reset_imitation = np.zeros(len(objs_reset_imitation))
            for j in range(len(objs_reset_imitation)):
                if objs_reset_imitation[j] == 0 and obj_opt == 0:
                    gamma_reset_imitation[j] = 0
                elif objs_reset_imitation[j] * obj_opt < 0:
                    gamma_reset_imitation[j] = 1
                else:
                    gamma_reset_imitation[j] = np.abs(objs_reset_imitation[j] - obj_opt) / np.maximum(np.abs(objs_reset_imitation[j]), np.abs(obj_opt)) #

            primal_gap_final_imitation = np.abs(objs_reset_imitation[-1] - obj_opt) / np.abs(obj_opt)
            primal_gap_final_reset_imitations.append(primal_gap_final_imitation)

            stepline_reset_imitation = interp1d(times_reset_imitation, gamma_reset_imitation, 'previous')
            steplines_reset_imitations.append(stepline_reset_imitation)

            # compute primal interal
            primal_int_reset_imitation = 0
            for j in range(len(objs_reset_imitation) - 1):
                primal_int_reset_imitation += gamma_reset_imitation[j] * (times_reset_imitation[j + 1] - times_reset_imitation[j])
            primal_in_reset_imitations.append(primal_int_reset_imitation)

            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of objective")
            # fig.subplots_adjust(top=0.5)
            # ax.set_title(instance_name, loc='right')
            # ax.plot(times, objs, label='lb baseline')
            # ax.plot(times_reset_vanilla, objs_reset_vanilla, label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()
            #
            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of primal gap")
            # fig.subplots_adjust(top=0.5)
            # ax.set_title(instance_name, loc='right')
            # ax.plot(times, gamma_baseline, label='lb baseline')
            # ax.plot(times_reset_vanilla, gamma_reset_vanilla, label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()


        # primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_reset_vanilla = np.array(primal_int_reset_vanillas).reshape(-1)
        primal_in_reset_imitation = np.array(primal_in_reset_imitations).reshape(-1)

        # primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_reset_vanilla = np.array(primal_gap_final_reset_vanillas).reshape(-1)
        primal_gap_final_reset_imitation = np.array(primal_gap_final_reset_imitations).reshape(-1)

        # avarage primal integral over test dataset
        # primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_reset_vanilla_ave = primal_int_reset_vanilla.sum() / len(primal_int_reset_vanilla)
        primal_int_reset_imitation_ave = primal_in_reset_imitation.sum() / len(primal_in_reset_imitation)

        # primal_gap_final_baselines = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_reset_vanilla = primal_gap_final_reset_vanilla.sum() / len(primal_gap_final_reset_vanilla)
        primal_gap_final_reset_imitation = primal_gap_final_reset_imitation.sum() / len(primal_gap_final_reset_imitation)

        print(self.instance_type + self.instance_size)
        print(self.incumbent_mode + 'Solution')
        # print('baseline primal integral: ', primal_int_base_ave)
        print('baseline primal integral: ', primal_int_reset_vanilla_ave)
        print('imitation primal integral: ', primal_int_reset_imitation_ave)
        print('\n')
        # print('baseline primal gap: ',primal_gap_final_baselines)
        print('baseline primal gap: ', primal_gap_final_reset_vanilla)
        print('imitation primal gap: ', primal_gap_final_reset_imitation)

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)

        # primalgaps_baseline = None
        # for n, stepline_baseline in enumerate(steplines_baseline):
        #     primal_gap = stepline_baseline(t)
        #     if n==0:
        #         primalgaps_baseline = primal_gap
        #     else:
        #         primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        # primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)

        primalgaps_reset_vanilla = None
        for n, stepline_reset_vanilla in enumerate(steplines_reset_vanillas):
            primal_gap = stepline_reset_vanilla(t)
            if n == 0:
                primalgaps_reset_vanilla = primal_gap
            else:
                primalgaps_reset_vanilla = np.vstack((primalgaps_reset_vanilla, primal_gap))
        primalgap_reset_vanilla_ave = np.average(primalgaps_reset_vanilla, axis=0)

        primalgaps_reset_imitation = None
        for n, stepline_reset_imitation in enumerate(steplines_reset_imitations):
            primal_gap = stepline_reset_imitation(t)
            if n == 0:
                primalgaps_reset_imitation = primal_gap
            else:
                primalgaps_reset_imitation = np.vstack((primalgaps_reset_imitation, primal_gap))
        primalgap_reset_imitation_ave = np.average(primalgaps_reset_imitation, axis=0)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle("Normalized primal gap")
        # fig.subplots_adjust(top=0.5)
        ax.set_title(self.instance_type + '-' + self.incumbent_mode, loc='right')
        # ax.plot(t, primalgap_baseline_ave, label='lb-baseline')
        ax.plot(t, primalgap_reset_vanilla_ave, label='lb-gnn-baseline')
        ax.plot(t, primalgap_reset_imitation_ave,'--', label='lb-gnn-imitation')
        ax.set_xlabel('time /s')
        ax.set_ylabel("normalized primal gap")
        ax.legend()
        plt.show()

class RegressionInitialK:

    def __init__(self, instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=100):
        self.instance_type = instance_type
        self.instance_size = instance_size
        self.incumbent_mode = incumbent_mode
        self.lbconstraint_mode = lbconstraint_mode
        self.seed = seed
        self.directory = './result/generated_instances/' + self.instance_type + '/' + self.instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # self.generator = generator_switcher(self.instance_type + self.instance_size)

        self.initialize_ecole_env()

        self.env.seed(self.seed)  # environment (SCIP)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def initialize_ecole_env(self):

        if self.incumbent_mode == 'firstsol':

            self.env = ecole.environment.Configuring(

                # set up a few SCIP parameters
                scip_params={
                    "presolving/maxrounds": 0,  # deactivate presolving
                    "presolving/maxrestarts": 0,
                },

                observation_function=ecole.observation.MilpBipartite(),

                reward_function=None,

                # collect additional metrics for information purposes
                information_function={
                    'time': ecole.reward.SolvingTime().cumsum(),
                }
            )

        elif self.incumbent_mode == 'rootsol':

            if self.instance_type == 'independentset':
                self.env = SimpleConfiguring(

                    # set up a few SCIP parameters
                    scip_params={
                        "presolving/maxrounds": 0,  # deactivate presolving
                        "presolving/maxrestarts": 0,
                    },

                    observation_function=ecole.observation.MilpBipartite(),

                    reward_function=None,

                    # collect additional metrics for information purposes
                    information_function={
                        'time': ecole.reward.SolvingTime().cumsum(),
                    }
                )
            else:
                self.env = SimpleConfiguringEnablecuts(

                    # set up a few SCIP parameters
                    scip_params={
                        "presolving/maxrounds": 0,  # deactivate presolving
                        "presolving/maxrestarts": 0,
                    },

                    observation_function=ecole.observation.MilpBipartite(),

                    reward_function=None,

                    # collect additional metrics for information purposes
                    information_function={
                        'time': ecole.reward.SolvingTime().cumsum(),
                    }
                )
            # elif self.instance_type == 'capacitedfacility':
            #     self.env = SimpleConfiguringEnableheuristics(
            #
            #         # set up a few SCIP parameters
            #         scip_params={
            #             "presolving/maxrounds": 0,  # deactivate presolving
            #             "presolving/maxrestarts": 0,
            #         },
            #
            #         observation_function=ecole.observation.MilpBipartite(),
            #
            #         reward_function=None,
            #
            #         # collect additional metrics for information purposes
            #         information_function={
            #             'time': ecole.reward.SolvingTime().cumsum(),
            #         }
            #     )

    def set_and_optimize_MIP(self, MIP_model, incumbent_mode):

        preprocess_off = True
        if incumbent_mode == 'firstsol':
            heuristics_off = False
            cuts_off = False
        elif incumbent_mode == 'rootsol':
            if self.instance_type == 'independentset':
                heuristics_off = True
                cuts_off = True
            else:
                heuristics_off = True
                cuts_off = False
            # elif self.instance_type == 'capacitedfacility':
            #     heuristics_off = False
            #     cuts_off = True

        if preprocess_off:
            MIP_model.setParam('presolving/maxrounds', 0)
            MIP_model.setParam('presolving/maxrestarts', 0)

        if heuristics_off:
            MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)

        if cuts_off:
            MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)

        if incumbent_mode == 'firstsol':
            MIP_model.setParam('limits/solutions', 1)
        elif incumbent_mode == 'rootsol':
            MIP_model.setParam("limits/nodes", 1)

        MIP_model.optimize()

        t = MIP_model.getSolvingTime()
        status = MIP_model.getStatus()
        lp_status = MIP_model.getLPSolstat()
        stage = MIP_model.getStage()
        n_sols = MIP_model.getNSols()

        # print("* Model status: %s" % status)
        # print("* LP status: %s" % lp_status)
        # print("* Solve stage: %s" % stage)
        # print("* Solving time: %s" % t)
        # print('* number of sol : ', n_sols)

        incumbent_solution = MIP_model.getBestSol()
        feasible = MIP_model.checkSol(solution=incumbent_solution)

        return status, feasible, MIP_model, incumbent_solution

    def initialize_MIP(self, MIP_model):

        MIP_model_2, MIP_2_vars, success = MIP_model.createCopy(
            problemName='Baseline', origcopy=False)

        incumbent_mode = self.incumbent_mode
        if self.incumbent_mode == 'firstsol':
            incumbent_mode_2 = 'rootsol'
        elif self.incumbent_mode == 'rootsol':
            incumbent_mode_2 = 'firstsol'

        status, feasible, MIP_model, incumbent_solution = self.set_and_optimize_MIP(MIP_model, incumbent_mode)
        status_2, feasible_2, MIP_model_2, incumbent_solution_2 = self.set_and_optimize_MIP(MIP_model_2, incumbent_mode_2)

        feasible = feasible and feasible_2

        if (not status == 'optimal') and (not status_2 == 'optimal'):
            not_optimal = True
        else:
            not_optimal = False

        if not_optimal and feasible:
            valid = True
        else:
            valid = False

        return valid, MIP_model, incumbent_solution

    def sample_k_per_instance(self, t_limit, index_instance):

        instance = next(self.generator)
        MIP_model = instance.as_pyscipopt()
        MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
        instance_name = MIP_model.getProbName()
        print(instance_name)

        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        valid, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)
        if valid:
            if index_instance > -1:
                initial_obj = MIP_model.getObjVal()
                print("Initial obj before LB: {}".format(initial_obj))
                print('Relative gap: ', MIP_model.getGap())

                n_supportbinvars = binary_support(MIP_model, incumbent_solution)
                print('binary support: ', n_supportbinvars)

                MIP_model.resetParams()

                neigh_sizes = []
                objs = []
                t = []
                n_supportbins = []
                statuss = []

                nsample = 101
                # create a copy of the MIP to be 'locally branched'
                MIP_copy, subMIP_copy_vars, success = MIP_model.createCopy(problemName='MIPCopy',
                                                                              origcopy=False)
                sol_MIP_copy = MIP_copy.createSol()

                # create a primal solution for the copy MIP by copying the solution of original MIP
                n_vars = MIP_model.getNVars()
                subMIP_vars = MIP_model.getVars()

                for j in range(n_vars):
                    val = MIP_model.getSolVal(incumbent_solution, subMIP_vars[j])
                    MIP_copy.setSolVal(sol_MIP_copy, subMIP_copy_vars[j], val)
                feasible = MIP_copy.checkSol(solution=sol_MIP_copy)

                if feasible:
                    # print("the trivial solution of subMIP is feasible ")
                    MIP_copy.addSol(sol_MIP_copy, False)
                    # print("the feasible solution of subMIP_copy is added to subMIP_copy")
                else:
                    print("Warn: the trivial solution of subMIP_copy is not feasible!")

                n_supportbinvars = binary_support(MIP_copy, sol_MIP_copy)
                print('binary support: ', n_supportbinvars)

                MIP_model.freeProb()
                del MIP_model

                for i in range(nsample):

                    # create a copy of the MIP to be 'locally branched'
                    subMIP_copy = MIP_copy
                    sol_subMIP_copy =  sol_MIP_copy

                    # add LB constraint to subMIP model
                    alpha = 0.01 * i
                    # if nsample == 41:
                    #     if i<11:
                    #         alpha = 0.01*i
                    #     elif i<31:
                    #         alpha = 0.02*(i-5)
                    #     else:
                    #         alpha = 0.05*(i-20)

                    if self.lbconstraint_mode == 'asymmetric':
                        neigh_size = alpha * n_supportbinvars
                        subMIP_copy, constraint_lb = addLBConstraintAsymmetric(subMIP_copy, sol_subMIP_copy, neigh_size)
                    else:
                        neigh_size = alpha * n_binvars
                        subMIP_copy, constraint_lb = addLBConstraint(subMIP_copy, sol_subMIP_copy, neigh_size)

                    subMIP_copy.setParam('limits/time', t_limit)
                    subMIP_copy.optimize()

                    status = subMIP_copy.getStatus()
                    best_obj = subMIP_copy.getSolObjVal(subMIP_copy.getBestSol())
                    solving_time = subMIP_copy.getSolvingTime()  # total time used for solving (including presolving) the current problem

                    best_sol = subMIP_copy.getBestSol()

                    vars_subMIP = subMIP_copy.getVars()
                    n_binvars_subMIP = subMIP_copy.getNBinVars()
                    n_supportbins_subMIP = 0
                    for i in range(n_binvars_subMIP):
                        val = subMIP_copy.getSolVal(best_sol, vars_subMIP[i])
                        assert subMIP_copy.isFeasIntegral(val), "Error: Value of a binary varialbe is not integral!"
                        if subMIP_copy.isFeasEQ(val, 1.0):
                            n_supportbins_subMIP += 1

                    neigh_sizes.append(alpha)
                    objs.append(best_obj)
                    t.append(solving_time)
                    n_supportbins.append(n_supportbins_subMIP)
                    statuss.append(status)

                    MIP_copy.freeTransform()
                    MIP_copy.delCons(constraint_lb)
                    MIP_copy.releasePyCons(constraint_lb)
                    del constraint_lb

                for i in range(len(t)):
                    print('Neighsize: {:.4f}'.format(neigh_sizes[i]),
                          'Best obj: {:.4f}'.format(objs[i]),
                          'Binary supports:{}'.format(n_supportbins[i]),
                          'Solving time: {:.4f}'.format(t[i]),
                          'Status: {}'.format(statuss[i])
                          )

                neigh_sizes = np.array(neigh_sizes).reshape(-1)
                t = np.array(t).reshape(-1)
                objs = np.array(objs).reshape(-1)

                f = self.k_samples_directory + instance_name
                np.savez(f, neigh_sizes=neigh_sizes, objs=objs, t=t)
            index_instance += 1

        del instance
        return index_instance

    def generate_k_samples(self, t_limit):
        """
        For each MIP instance, sample k from [0,1] * n_binary(symmetric) or [0,1] * n_binary_support(asymmetric),
        and evaluate the performance of 1st round of local-branching
        :param t_limit:
        :param k_samples_directory:
        :return:
        """

        self.k_samples_directory = self.directory + 'k_samples' + '/'
        pathlib.Path(self.k_samples_directory).mkdir(parents=True, exist_ok=True)

        self.generator = generator_switcher(self.instance_type + self.instance_size)
        self.generator.seed(self.seed)

        index_instance = 0

        # while index_instance < 86:
        #     instance = next(self.generator)
        #     MIP_model = instance.as_pyscipopt()
        #     MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
        #     instance_name = MIP_model.getProbName()
        #     print(instance_name)
        #     index_instance += 1

        while index_instance < 100:
            index_instance = self.sample_k_per_instance(t_limit, index_instance)

            # instance = next(self.generator)
            # MIP_model = instance.as_pyscipopt()
            # MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
            # instance_name = MIP_model.getProbName()
            # print(instance_name)
            #
            # n_vars = MIP_model.getNVars()
            # n_binvars = MIP_model.getNBinVars()
            # print("N of variables: {}".format(n_vars))
            # print("N of binary vars: {}".format(n_binvars))
            # print("N of constraints: {}".format(MIP_model.getNConss()))
            #
            # status, feasible, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)
            # if (not status == 'optimal') and feasible:
            #     initial_obj = MIP_model.getObjVal()
            #     print("Initial obj before LB: {}".format(initial_obj))
            #     print('Relative gap: ', MIP_model.getGap())
            #
            #     n_supportbinvars = binary_support(MIP_model, incumbent_solution)
            #     print('binary support: ', n_supportbinvars)
            #
            #
            #     MIP_model.resetParams()
            #
            #     neigh_sizes = []
            #     objs = []
            #     t = []
            #     n_supportbins = []
            #     statuss = []
            #     MIP_model.resetParams()
            #     nsample = 101
            #     for i in range(nsample):
            #
            #         # create a copy of the MIP to be 'locally branched'
            #         subMIP_copy, subMIP_copy_vars, success = MIP_model.createCopy(problemName='subMIPmodelCopy',
            #                                                                       origcopy=False)
            #         sol_subMIP_copy = subMIP_copy.createSol()
            #
            #         # create a primal solution for the copy MIP by copying the solution of original MIP
            #         n_vars = MIP_model.getNVars()
            #         subMIP_vars = MIP_model.getVars()
            #
            #         for j in range(n_vars):
            #             val = MIP_model.getSolVal(incumbent_solution, subMIP_vars[j])
            #             subMIP_copy.setSolVal(sol_subMIP_copy, subMIP_copy_vars[j], val)
            #         feasible = subMIP_copy.checkSol(solution=sol_subMIP_copy)
            #
            #         if feasible:
            #             # print("the trivial solution of subMIP is feasible ")
            #             subMIP_copy.addSol(sol_subMIP_copy, False)
            #             # print("the feasible solution of subMIP_copy is added to subMIP_copy")
            #         else:
            #             print("Warn: the trivial solution of subMIP_copy is not feasible!")
            #
            #         # add LB constraint to subMIP model
            #         alpha = 0.01 * i
            #         # if nsample == 41:
            #         #     if i<11:
            #         #         alpha = 0.01*i
            #         #     elif i<31:
            #         #         alpha = 0.02*(i-5)
            #         #     else:
            #         #         alpha = 0.05*(i-20)
            #
            #         if self.lbconstraint_mode == 'asymmetric':
            #             neigh_size = alpha * n_supportbinvars
            #             subMIP_copy = addLBConstraintAsymmetric(subMIP_copy, sol_subMIP_copy, neigh_size)
            #         else:
            #             neigh_size = alpha * n_binvars
            #             subMIP_copy = addLBConstraint(subMIP_copy, sol_subMIP_copy, neigh_size)
            #
            #         subMIP_copy.setParam('limits/time', t_limit)
            #         subMIP_copy.optimize()
            #
            #         status = subMIP_copy.getStatus()
            #         best_obj = subMIP_copy.getSolObjVal(subMIP_copy.getBestSol())
            #         solving_time = subMIP_copy.getSolvingTime()  # total time used for solving (including presolving) the current problem
            #
            #         best_sol = subMIP_copy.getBestSol()
            #
            #         vars_subMIP = subMIP_copy.getVars()
            #         n_binvars_subMIP = subMIP_copy.getNBinVars()
            #         n_supportbins_subMIP = 0
            #         for i in range(n_binvars_subMIP):
            #             val = subMIP_copy.getSolVal(best_sol, vars_subMIP[i])
            #             assert subMIP_copy.isFeasIntegral(val), "Error: Value of a binary varialbe is not integral!"
            #             if subMIP_copy.isFeasEQ(val, 1.0):
            #                 n_supportbins_subMIP += 1
            #
            #         neigh_sizes.append(alpha)
            #         objs.append(best_obj)
            #         t.append(solving_time)
            #         n_supportbins.append(n_supportbins_subMIP)
            #         statuss.append(status)
            #
            #     for i in range(len(t)):
            #         print('Neighsize: {:.4f}'.format(neigh_sizes[i]),
            #               'Best obj: {:.4f}'.format(objs[i]),
            #               'Binary supports:{}'.format(n_supportbins[i]),
            #               'Solving time: {:.4f}'.format(t[i]),
            #               'Status: {}'.format(statuss[i])
            #               )
            #
            #     neigh_sizes = np.array(neigh_sizes).reshape(-1).astype('float64')
            #     t = np.array(t).reshape(-1)
            #     objs = np.array(objs).reshape(-1)
            #     f = self.k_samples_directory + instance_name
            #     np.savez(f, neigh_sizes=neigh_sizes, objs=objs, t=t)
            #     index_instance += 1

    def generate_regression_samples(self, t_limit):

        self.k_samples_directory = self.directory + 'k_samples' + '/'
        self.regression_samples_directory = self.directory + 'regression_samples' + '/'
        pathlib.Path(self.regression_samples_directory).mkdir(parents=True, exist_ok=True)

        self.generator = generator_switcher(self.instance_type + self.instance_size)
        self.generator.seed(self.seed)

        index_instance = 0
        while index_instance < 100:

            instance = next(self.generator)
            MIP_model = instance.as_pyscipopt()
            MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
            instance_name = MIP_model.getProbName()
            print(instance_name)

            n_vars = MIP_model.getNVars()
            n_binvars = MIP_model.getNBinVars()
            print("N of variables: {}".format(n_vars))
            print("N of binary vars: {}".format(n_binvars))
            print("N of constraints: {}".format(MIP_model.getNConss()))

            valid, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)
            if valid:
                if index_instance > -1:
                    data = np.load(self.k_samples_directory + instance_name + '.npz')
                    k = data['neigh_sizes']
                    t = data['t']
                    objs_abs = data['objs']

                    # normalize the objective and solving time
                    t = t / t_limit
                    objs = (objs_abs - np.min(objs_abs))
                    objs = objs / np.max(objs)

                    t = mean_filter(t, 5)
                    objs = mean_filter(objs, 5)

                    # t = mean_forward_filter(t,10)
                    # objs = mean_forward_filter(objs, 10)

                    # compute the performance score
                    alpha = 1 / 2
                    perf_score = alpha * t + (1 - alpha) * objs
                    k_bests = k[np.where(perf_score == perf_score.min())]
                    k_init = k_bests[0]

                    # plt.clf()
                    # fig, ax = plt.subplots(3, 1, figsize=(6.4, 6.4))
                    # fig.suptitle("Evaluation of size of lb neighborhood")
                    # fig.subplots_adjust(top=0.5)
                    # ax[0].plot(k, objs)
                    # ax[0].set_title(instance_name, loc='right')
                    # ax[0].set_xlabel(r'$\ r $   ' + '(Neighborhood size: ' + r'$K = r \times N$)') #
                    # ax[0].set_ylabel("Objective")
                    # ax[1].plot(k, t)
                    # # ax[1].set_ylim([0,31])
                    # ax[1].set_ylabel("Solving time")
                    # ax[2].plot(k, perf_score)
                    # ax[2].set_ylabel("Performance score")
                    # plt.show()

                    # instance = ecole.scip.Model.from_pyscipopt(MIP_model)
                    observation, _, _, done, _ = self.env.reset(instance)

                    if self.incumbent_mode == 'firstsol':
                        action = {'limits/solutions': 1}
                    elif self.incumbent_mode == 'rootsol':
                        action = {'limits/nodes': 1}

                    observation, _, _, done, _ = self.env.step(action)

                    data_sample = [observation, k_init]
                    filename = f'{self.regression_samples_directory}regression-{instance_name}.pkl'
                    with gzip.open(filename, 'wb') as f:
                        pickle.dump(data_sample, f)

                index_instance += 1

    def load_dataset(self, dataset_directory=None):

        self.regression_samples_directory = dataset_directory
        filename = 'regression-' + self.instance_type + '-*.pkl'
        # print(filename)
        sample_files = [str(path) for path in pathlib.Path(self.regression_samples_directory).glob(filename)]
        train_files = sample_files[:int(0.7 * len(sample_files))]
        valid_files = sample_files[int(0.7 * len(sample_files)):int(0.8 * len(sample_files))]
        test_files =  sample_files[int(0.8 * len(sample_files)):]

        train_data = GraphDataset(train_files)
        train_loader = torch_geometric.data.DataLoader(train_data, batch_size=1, shuffle=True)
        valid_data = GraphDataset(valid_files)
        valid_loader = torch_geometric.data.DataLoader(valid_data, batch_size=1, shuffle=False)
        test_data = GraphDataset(test_files)
        test_loader = torch_geometric.data.DataLoader(test_data, batch_size=1, shuffle=False)

        return train_loader, valid_loader, test_loader

    def train(self, gnn_model, data_loader, optimizer=None):
        """
        training function
        :param gnn_model:
        :param data_loader:
        :param optimizer:
        :return:
        """
        mean_loss = 0
        n_samples_precessed = 0
        with torch.set_grad_enabled(optimizer is not None):
            for batch in data_loader:
                k_model = gnn_model(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
                k_init = batch.k_init
                loss = F.l1_loss(k_model.float(), k_init.float())
                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                mean_loss += loss.item() * batch.num_graphs
                n_samples_precessed += batch.num_graphs
        mean_loss /= n_samples_precessed

        return mean_loss

    def test(self, gnn_model, data_loader):
        n_samples_precessed = 0
        loss_list = []
        k_model_list = []
        k_init_list = []
        graph_index = []
        for batch in data_loader:
            k_model = gnn_model(batch.constraint_features, batch.edge_index, batch.edge_attr, batch.variable_features)
            k_init = batch.k_init
            loss = F.l1_loss(k_model, k_init)

            if batch.num_graphs == 1:
                loss_list.append(loss.item())
                k_model_list.append(k_model.item())
                k_init_list.append(k_init)
                graph_index.append(n_samples_precessed)
                n_samples_precessed += 1

            else:

                for g in range(batch.num_graphs):
                    loss_list.append(loss.item()[g])
                    k_model_list.append(k_model[g])
                    k_init_list.append(k_init(g))
                    graph_index.append(n_samples_precessed)
                    n_samples_precessed += 1

        loss_list = np.array(loss_list).reshape(-1)
        k_model_list = np.array(k_model_list).reshape(-1)
        k_init_list = np.array(k_init_list).reshape(-1)
        graph_index = np.array(graph_index).reshape(-1)

        loss_ave = loss_list.mean()
        k_model_ave = k_model_list.mean()
        k_init_ave = k_init_list.mean()

        return loss_ave, k_model_ave, k_init_ave

    def execute_regression(self, lr=0.0000001, n_epochs=20):

        saved_gnn_directory = './result/saved_models/'
        pathlib.Path(saved_gnn_directory).mkdir(parents=True, exist_ok=True)

        train_loaders = {}
        val_loaders = {}
        test_loaders = {}

        # load the small dataset
        small_dataset = self.instance_type + "-small"
        small_directory = './result/generated_instances/' + self.instance_type + '/' + '-small' + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'

        small_regression_samples_directory = self.directory + 'regression_samples' + '/'
        train_loader, valid_loader, test_loader = self.load_dataset(dataset_directory=small_regression_samples_directory)
        train_loaders[small_dataset] = train_loader
        val_loaders[small_dataset] = valid_loader
        test_loaders[small_dataset] = test_loader

        large_dataset = self.instance_type + "-large"
        large_directory = './result/generated_instances/' + self.instance_type + '/' + '-large' + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        test_regression_samples_directory = large_directory + 'regression_samples' + '/'
        train_loader, valid_loader, test_loader = self.load_dataset(dataset_directory=test_regression_samples_directory)
        train_loaders[large_dataset] = train_loader
        val_loaders[large_dataset] = valid_loader
        test_loaders[large_dataset] = test_loader

        model_gnn = GNNPolicy()
        train_dataset = small_dataset
        valid_dataset = small_dataset
        test_dataset = large_dataset
        # LEARNING_RATE = 0.0000001  # setcovering:0.0000005 cap-loc: 0.00000005 independentset: 0.0000001

        optimizer = torch.optim.Adam(model_gnn.parameters(), lr=lr)
        k_init = []
        k_model = []
        loss = []
        epochs = []
        for epoch in range(n_epochs):
            print(f"Epoch {epoch}")

            if epoch == 0:
                optim = None
            else:
                optim = optimizer

            train_loader = train_loaders[train_dataset]
            train_loss = self.train(model_gnn, train_loader, optim)
            print(f"Train loss: {train_loss:0.6f}")

            # torch.save(model_gnn.state_dict(), 'trained_params_' + train_dataset + '.pth')
            # model_gnn2.load_state_dict(torch.load('trained_params_' + train_dataset + '.pth'))

            valid_loader = val_loaders[valid_dataset]
            valid_loss = self.train(model_gnn, valid_loader, None)
            print(f"Valid loss: {valid_loss:0.6f}")

            test_loader = test_loaders[test_dataset]
            loss_ave, k_model_ave, k_init_ave = self.test(model_gnn, test_loader)

            loss.append(loss_ave)
            k_model.append(k_model_ave)
            k_init.append(k_init_ave)
            epochs.append(epoch)

        loss_np = np.array(loss).reshape(-1)
        k_model_np = np.array(k_model).reshape(-1)
        k_init_np = np.array(k_init).reshape(-1)
        epochs_np = np.array(epochs).reshape(-1)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(2, 1, figsize=(8, 6.4))
        fig.suptitle("Test Result: prediction of initial k")
        fig.subplots_adjust(top=0.5)
        ax[0].set_title(valid_dataset, loc='right')
        ax[0].plot(epochs_np, loss_np)
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel("loss")
        ax[1].plot(epochs_np, k_model_np, label='k-prediction')

        ax[1].plot(epochs_np, k_init_np, label='k-label')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel("k")
        ax[1].set_ylim([0, 1.1])
        ax[1].legend()
        plt.show()

        torch.save(model_gnn.state_dict(),
                   saved_gnn_directory + 'trained_params_mean' + train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '.pth')

    def evaluate_lb_per_instance(self, node_time_limit, total_time_limit, index_instance, reset_k_at_2nditeration=False):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        instance = next(self.generator)
        MIP_model = instance.as_pyscipopt()
        MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
        instance_name = MIP_model.getProbName()
        print('\n')
        print(instance_name)

        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        valid, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)
        conti = -1
        # if self.incumbent_mode == 'rootsol' and self.instance_type == 'independentset':
        #     conti = 196

        if valid:
            if index_instance > -1 and index_instance > conti:
                gc.collect()
                observation, _, _, done, _ = self.env.reset(instance)
                del observation
                # print(observation)

                if self.incumbent_mode == 'firstsol':
                    action = {'limits/solutions': 1}
                elif self.incumbent_mode == 'rootsol':
                    action = {'limits/nodes': 1}  #
                sample_observation, _, _, done, _ = self.env.step(action)


                # print(sample_observation)
                graph = BipartiteNodeData(sample_observation.constraint_features,
                                          sample_observation.edge_features.indices,
                                          sample_observation.edge_features.values,
                                          sample_observation.variable_features)

                # We must tell pytorch geometric how many nodes there are, for indexing purposes
                graph.num_nodes = sample_observation.constraint_features.shape[0] + \
                                  sample_observation.variable_features.shape[
                                      0]

                filename = f'{self.directory_transformedmodel}{self.instance_type}-{str(index_instance)}_transformed.cip'
                firstsol_filename = f'{self.directory_sol}{self.incumbent_mode}-{self.instance_type}-{str(index_instance)}_transformed.sol'

                model = Model()
                model.readProblem(filename)
                sol = model.readSolFile(firstsol_filename)

                feas = model.checkSol(sol)
                try:
                    model.addSol(sol, False)
                except:
                    print('Error: the root solution of ' + model.getProbName() + ' is not feasible!')

                instance2 = ecole.scip.Model.from_pyscipopt(model)
                observation, _, _, done, _ = self.env.reset(instance2)
                graph2 = BipartiteNodeData(observation.constraint_features,
                                          observation.edge_features.indices,
                                          observation.edge_features.values,
                                          observation.variable_features)

                # We must tell pytorch geometric how many nodes there are, for indexing purposes
                graph2.num_nodes = observation.constraint_features.shape[0] + \
                                  observation.variable_features.shape[
                                      0]

                # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
                # MIP_model = instance

                # MIP_model.optimize()
                # print("Status:", MIP_model.getStatus())
                # print("best obj: ", MIP_model.getObjVal())
                # print("Solving time: ", MIP_model.getSolvingTime())

                initial_obj = MIP_model.getSolObjVal(incumbent_solution)
                print("Initial obj before LB: {}".format(initial_obj))

                binary_supports = binary_support(MIP_model, incumbent_solution)
                print('binary support: ', binary_supports)

                model_gnn = GNNPolicy()

                model_gnn.load_state_dict(torch.load(
                    self.saved_gnn_directory + 'trained_params_mean_' + self.train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '.pth'))

                # model_gnn.load_state_dict(torch.load(
                #      'trained_params_' + self.instance_type + '.pth'))

                k_model = model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                    graph.variable_features)

                k_pred = k_model.item() * n_binvars
                print('GNN prediction: ', k_model.item())

                k_model2 = model_gnn(graph2.constraint_features, graph2.edge_index, graph2.edge_attr,
                                    graph2.variable_features)

                print('GNN prediction of model2: ', k_model2.item())

                if self.is_symmetric == False:
                    k_pred = k_model.item() * binary_supports

                del graph
                del sample_observation
                del model_gnn

                MIP_model, MIP_copy_vars, success = MIP_model.createCopy(
                    problemName='Baseline', origcopy=False)
                status = MIP_model.getStatus()
                print("* Model status: %s" % status)
                MIP_model.resetParams()
                MIP_model.setPresolve(pyscipopt.SCIP_PARAMSETTING.OFF)
                MIP_model.setHeuristics(pyscipopt.SCIP_PARAMSETTING.OFF)
                MIP_model.setSeparating(pyscipopt.SCIP_PARAMSETTING.OFF)
                MIP_model.setIntParam("lp/solvefreq", 0)
                MIP_model.setParam("limits/nodes", 1)
                MIP_model.setParam("limits/solutions", 1)

                # MIP_model.setParam("limits/solutions", 1)
                MIP_model.optimize()
                #
                status = MIP_model.getStatus()
                lp_status = MIP_model.getLPSolstat()
                stage = MIP_model.getStage()
                n_sols = MIP_model.getNSols()
                t = MIP_model.getSolvingTime()
                print("* Model status: %s" % status)
                print("* Solve stage: %s" % stage)
                print("* LP status: %s" % lp_status)
                print('* number of sol : ', n_sols)

                sol_lp = MIP_model.createLPSol()

                k_prime = haming_distance_solutions(MIP_model, incumbent_solution, sol_lp)
                n_bins = MIP_model.getNBinVars()
                k_base = n_bins

                if self.is_symmetric == False:
                    k_prime = haming_distance_solutions_asym(MIP_model, incumbent_solution, sol_lp)
                    binary_supports = binary_support(MIP_model, incumbent_solution)
                    k_base = binary_supports

                print('k_lp =', k_prime/k_base)
                print('GNN prediction: ', k_model.item())

                # # create a copy of MIP
                # MIP_model.resetParams()
                # MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
                #     problemName='Baseline', origcopy=False)
                # MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
                #     problemName='GNN',
                #     origcopy=False)
                # MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
                #     problemName='GNN+reset',
                #     origcopy=False)
                #
                # print('MIP copies are created')
                #
                # MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent_solution,
                #                                         MIP_copy_vars)
                # MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent_solution,
                #                                           MIP_copy_vars2)
                # MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
                #                                           MIP_copy_vars3)
                #
                # print('incumbent solution is copied to MIP copies')
                # MIP_model.freeProb()
                # del MIP_model
                # del incumbent_solution
                #
                # # sol = MIP_model_copy.getBestSol()
                # # initial_obj = MIP_model_copy.getSolObjVal(sol)
                # # print("Initial obj before LB: {}".format(initial_obj))
                #
                # # execute local branching baseline heuristic by Fischetti and Lodi
                # lb_model = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy, k=self.k_baseline,
                #                           node_time_limit=node_time_limit,
                #                           total_time_limit=total_time_limit)
                # status, obj_best, elapsed_time, lb_bits, times, objs = lb_model.search_localbranch(is_symmetric=self.is_symmetric,
                #                                                              reset_k_at_2nditeration=False)
                # print("Instance:", MIP_model_copy.getProbName())
                # print("Status of LB: ", status)
                # print("Best obj of LB: ", obj_best)
                # print("Solving time: ", elapsed_time)
                # print('\n')
                #
                # MIP_model_copy.freeProb()
                # del sol_MIP_copy
                # del MIP_model_copy
                #
                # # sol = MIP_model_copy2.getBestSol()
                # # initial_obj = MIP_model_copy2.getSolObjVal(sol)
                # # print("Initial obj before LB: {}".format(initial_obj))
                #
                # # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
                # lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                #                            node_time_limit=node_time_limit,
                #                            total_time_limit=total_time_limit)
                # status, obj_best, elapsed_time, lb_bits_pred_reset, times_pred_rest, objs_pred_rest = lb_model3.search_localbranch(is_symmetric=self.is_symmetric,
                #                                                               reset_k_at_2nditeration=reset_k_at_2nditeration)
                #
                # print("Instance:", MIP_model_copy3.getProbName())
                # print("Status of LB: ", status)
                # print("Best obj of LB: ", obj_best)
                # print("Solving time: ", elapsed_time)
                # print('\n')
                #
                # MIP_model_copy3.freeProb()
                # del sol_MIP_copy3
                # del MIP_model_copy3
                #
                # # execute local branching with 1. first k predicted by GNN; 2. from 2nd iteration of lb, continue lb algorithm with no further injection
                # lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=k_pred,
                #                            node_time_limit=node_time_limit,
                #                            total_time_limit=total_time_limit)
                # status, obj_best, elapsed_time, lb_bits_pred, times_pred, objs_pred = lb_model2.search_localbranch(is_symmetric=self.is_symmetric,
                #                                                               reset_k_at_2nditeration=False)
                #
                # print("Instance:", MIP_model_copy2.getProbName())
                # print("Status of LB: ", status)
                # print("Best obj of LB: ", obj_best)
                # print("Solving time: ", elapsed_time)
                # print('\n')
                #
                # MIP_model_copy2.freeProb()
                # del sol_MIP_copy2
                # del MIP_model_copy2
                #
                # data = [objs, times, objs_pred, times_pred, objs_pred_rest, times_pred_rest]
                # filename = f'{self.directory_lb_test}lb-test-{instance_name}.pkl'  # instance 100-199
                # with gzip.open(filename, 'wb') as f:
                #     pickle.dump(data, f)
                # del data
                # del objs
                # del times
                # del objs_pred
                # del times_pred
                # del objs_pred_rest
                # del times_pred_rest
                # del lb_model
                # del lb_model2
                # del lb_model3

            index_instance += 1
        del instance
        return index_instance

    def evaluate_localbranching(self, test_instance_size='-small', train_instance_size='-small', total_time_limit=60, node_time_limit=30, reset_k_at_2nditeration=False):

        self.train_dataset = self.instance_type + train_instance_size
        self.evaluation_dataset = self.instance_type + test_instance_size

        self.generator = generator_switcher(self.evaluation_dataset)
        self.generator.seed(self.seed)

        direc = './data/generated_instances/' + self.instance_type + '/' + test_instance_size + '/'
        self.directory_transformedmodel = direc + 'transformedmodel' + '/'
        self.directory_sol = direc + self.incumbent_mode + '/'

        self.k_baseline = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_gnn_directory = './result/saved_models/'

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        self.directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        index_instance = 0
        while index_instance < 200:
            index_instance = self.evaluate_lb_per_instance(node_time_limit=node_time_limit, total_time_limit=total_time_limit, index_instance=index_instance, reset_k_at_2nditeration=reset_k_at_2nditeration)

    def solve2opt_evaluation(self, test_instance_size='-small'):

        self.evaluation_dataset = self.instance_type + test_instance_size
        directory_opt = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + 'opt_solution' + '/'
        pathlib.Path(directory_opt).mkdir(parents=True, exist_ok=True)

        self.generator = generator_switcher(self.evaluation_dataset)
        self.generator.seed(self.seed)

        index_instance = 0
        while index_instance < 200:

            instance = next(self.generator)
            MIP_model = instance.as_pyscipopt()
            MIP_model.setProbName(self.instance_type + test_instance_size + '-' + str(index_instance))
            instance_name = MIP_model.getProbName()
            print(' \n')
            print(instance_name)

            n_vars = MIP_model.getNVars()
            n_binvars = MIP_model.getNBinVars()
            print("N of variables: {}".format(n_vars))
            print("N of binary vars: {}".format(n_binvars))
            print("N of constraints: {}".format(MIP_model.getNConss()))

            valid, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)

            if valid:
                if index_instance > 99:
                    MIP_model.resetParams()
                    MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
                        problemName='Baseline', origcopy=False)

                    MIP_model_copy.setParam('presolving/maxrounds', 0)
                    MIP_model_copy.setParam('presolving/maxrestarts', 0)
                    MIP_model_copy.setParam("display/verblevel", 0)
                    MIP_model_copy.optimize()
                    status = MIP_model_copy.getStatus()
                    if status == 'optimal':
                        obj = MIP_model_copy.getObjVal()
                        time = MIP_model_copy.getSolvingTime()
                        data = [obj, time]

                        filename = f'{directory_opt}{instance_name}-optimal-obj-time.pkl'
                        with gzip.open(filename, 'wb') as f:
                            pickle.dump(data, f)
                        del data
                    else:
                        print('Warning: solved problem ' + instance_name + ' is not optimal!')

                    print("instance:", MIP_model_copy.getProbName(),
                          "status:", MIP_model_copy.getStatus(),
                          "best obj: ", MIP_model_copy.getObjVal(),
                          "solving time: ", MIP_model_copy.getSolvingTime())

                    MIP_model_copy.freeProb()
                    del MIP_copy_vars
                    del MIP_model_copy

                index_instance += 1

            else:
                print('This instance is not valid for evaluation')

            MIP_model.freeProb()
            del MIP_model
            del incumbent_solution
            del instance

    def primal_integral(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/'
            directory_lb_test_2 = directory_2 + 'lb-from-' +  'rootsol' + '-t_node' + str(30) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/'
            directory_lb_test_2 = directory_2 + 'lb-from-' + 'firstsol' + '-t_node' + str(30) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        primal_int_baselines = []
        primal_int_preds = []
        primal_int_preds_reset = []
        primal_gap_final_baselines = []
        primal_gap_final_preds = []
        primal_gap_final_preds_reset = []
        steplines_baseline = []
        steplines_pred = []
        steplines_pred_reset = []

        for i in range(100, 200):

            instance_name = self.instance_type + '-' + str(i)  # instance 100-199

            filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'

            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs, times, objs_pred, times_pred, objs_pred_reset, times_pred_reset = data  # objs contains objs of a single instance of a lb test

            # filename_2 = f'{directory_lb_test_2}lb-test-{instance_name}.pkl'
            #
            # with gzip.open(filename_2, 'rb') as f:
            #     data = pickle.load(f)
            # objs_2, times_2, objs_pred_2, times_pred_2, objs_pred_reset_2, times_pred_reset_2 = data  # objs contains objs of a single instance of a lb test

            a = [objs.min(), objs_pred.min(), objs_pred_reset.min()] # objs_2.min(), objs_pred_2.min(), objs_pred_reset_2.min()
            # a = [objs.min(), objs_pred.min(), objs_pred_reset.min()]
            obj_opt = np.amin(a)

            # compute primal gap for baseline localbranching run
            # if times[-1] < total_time_limit:
            times = np.append(times, total_time_limit)
            objs = np.append(objs, objs[-1])

            gamma_baseline = np.zeros(len(objs))
            for j in range(len(objs)):
                if objs[j] == 0 and obj_opt == 0:
                    gamma_baseline[j] = 0
                elif objs[j] * obj_opt < 0:
                    gamma_baseline[j] = 1
                else:
                    gamma_baseline[j] = np.abs(objs[j] - obj_opt) / np.maximum(np.abs(objs[j]), np.abs(obj_opt)) #

            # compute the primal gap of last objective
            primal_gap_final_baseline = np.abs(objs[-1] - obj_opt) / np.abs(obj_opt)
            primal_gap_final_baselines.append(primal_gap_final_baseline)

            # create step line
            stepline_baseline = interp1d(times, gamma_baseline, 'previous')
            steplines_baseline.append(stepline_baseline)

            # compute primal integral
            primal_int_baseline = 0
            for j in range(len(objs) - 1):
                primal_int_baseline += gamma_baseline[j] * (times[j + 1] - times[j])
            primal_int_baselines.append(primal_int_baseline)



            # lb-gnn
            # if times_pred[-1] < total_time_limit:
            times_pred = np.append(times_pred, total_time_limit)
            objs_pred = np.append(objs_pred, objs_pred[-1])

            gamma_pred = np.zeros(len(objs_pred))
            for j in range(len(objs_pred)):
                if objs_pred[j] == 0 and obj_opt == 0:
                    gamma_pred[j] = 0
                elif objs_pred[j] * obj_opt < 0:
                    gamma_pred[j] = 1
                else:
                    gamma_pred[j] = np.abs(objs_pred[j] - obj_opt) / np.maximum(np.abs(objs_pred[j]), np.abs(obj_opt)) #

            primal_gap_final_pred = np.abs(objs_pred[-1] - obj_opt) / np.abs(obj_opt)
            primal_gap_final_preds.append(primal_gap_final_pred)

            stepline_pred = interp1d(times_pred, gamma_pred, 'previous')
            steplines_pred.append(stepline_pred)

            #
            # t = np.linspace(start=0.0, stop=total_time_limit, num=1001)
            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of primal gap")
            # fig.subplots_adjust(top=0.5)
            # # ax.set_title(instance_name, loc='right')
            # ax.plot(t, stepline_baseline(t), label='lb baseline')
            # ax.plot(t, stepline_pred(t), label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()

            # compute primal interal
            primal_int_pred = 0
            for j in range(len(objs_pred) - 1):
                primal_int_pred += gamma_pred[j] * (times_pred[j + 1] - times_pred[j])
            primal_int_preds.append(primal_int_pred)

            # lb-gnn-reset
            times_pred_reset = np.append(times_pred_reset, total_time_limit)
            objs_pred_reset = np.append(objs_pred_reset, objs_pred_reset[-1])

            gamma_pred_reset = np.zeros(len(objs_pred_reset))
            for j in range(len(objs_pred_reset)):
                if objs_pred_reset[j] == 0 and obj_opt == 0:
                    gamma_pred_reset[j] = 0
                elif objs_pred_reset[j] * obj_opt < 0:
                    gamma_pred_reset[j] = 1
                else:
                    gamma_pred_reset[j] = np.abs(objs_pred_reset[j] - obj_opt) / np.maximum(np.abs(objs_pred_reset[j]), np.abs(obj_opt)) #

            primal_gap_final_pred_reset = np.abs(objs_pred_reset[-1] - obj_opt) / np.abs(obj_opt)
            primal_gap_final_preds_reset.append(primal_gap_final_pred_reset)

            stepline_pred_reset = interp1d(times_pred_reset, gamma_pred_reset, 'previous')
            steplines_pred_reset.append(stepline_pred_reset)

            # compute primal interal
            primal_int_pred_reset = 0
            for j in range(len(objs_pred_reset) - 1):
                primal_int_pred_reset += gamma_pred_reset[j] * (times_pred_reset[j + 1] - times_pred_reset[j])
            primal_int_preds_reset.append(primal_int_pred_reset)

            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of objective")
            # fig.subplots_adjust(top=0.5)
            # ax.set_title(instance_name, loc='right')
            # ax.plot(times, objs, label='lb baseline')
            # ax.plot(times_pred, objs_pred, label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()
            #
            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of primal gap")
            # fig.subplots_adjust(top=0.5)
            # ax.set_title(instance_name, loc='right')
            # ax.plot(times, gamma_baseline, label='lb baseline')
            # ax.plot(times_pred, gamma_pred, label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()


        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_preds = np.array(primal_int_preds).reshape(-1)
        primal_int_preds_reset = np.array(primal_int_preds_reset).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_preds = np.array(primal_gap_final_preds).reshape(-1)
        primal_gap_final_preds_reset = np.array(primal_gap_final_preds_reset).reshape(-1)

        # avarage primal integral over test dataset
        primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_pred_ave = primal_int_preds.sum() / len(primal_int_preds)
        primal_int_pred_ave_reset = primal_int_preds_reset.sum() / len(primal_int_preds_reset)

        primal_gap_final_baselines = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_preds = primal_gap_final_preds.sum() / len(primal_gap_final_preds)
        primal_gap_final_preds_reset = primal_gap_final_preds_reset.sum() / len(primal_gap_final_preds_reset)

        print(self.instance_type + self.instance_size)
        print(self.incumbent_mode + 'Solution')
        print('baseline primal integral: ', primal_int_base_ave)
        print('k_pred primal integral: ', primal_int_pred_ave)
        print('k_pred_reset primal integral: ', primal_int_pred_ave_reset)
        print('\n')
        print('baseline primal gap: ',primal_gap_final_baselines)
        print('k_pred primal gap: ', primal_gap_final_preds)
        print('k_pred_reset primal gap: ', primal_gap_final_preds_reset)

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)
        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n==0:
                primalgaps_baseline = primal_gap
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)

        primalgaps_pred = None
        for n, stepline_pred in enumerate(steplines_pred):
            primal_gap = stepline_pred(t)
            if n == 0:
                primalgaps_pred = primal_gap
            else:
                primalgaps_pred = np.vstack((primalgaps_pred, primal_gap))
        primalgap_pred_ave = np.average(primalgaps_pred, axis=0)

        primalgaps_pred_reset = None
        for n, stepline_pred_reset in enumerate(steplines_pred_reset):
            primal_gap = stepline_pred_reset(t)
            if n == 0:
                primalgaps_pred_reset = primal_gap
            else:
                primalgaps_pred_reset = np.vstack((primalgaps_pred_reset, primal_gap))
        primalgap_pred_ave_reset = np.average(primalgaps_pred_reset, axis=0)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle("Normalized primal gap")
        # fig.subplots_adjust(top=0.5)
        ax.set_title(self.instance_type + '-' + self.incumbent_mode, loc='right')
        ax.plot(t, primalgap_baseline_ave, label='lb-baseline')
        ax.plot(t, primalgap_pred_ave, label='lb-gnn')
        ax.plot(t, primalgap_pred_ave_reset, '--', label='lb-gnn-reset')
        ax.set_xlabel('time /s')
        ax.set_ylabel("normalized primal gap")
        ax.legend()
        plt.show()

class ImitationLocalbranch(MlLocalbranch):
    def __init__(self, instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=100):
        super().__init__(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed)

    def rltrain_per_instance(self, node_time_limit, total_time_limit, index_instance,
                             reset_k_at_2nditeration=False, policy=None, optimizer=None,
                             criterion=None, device=None):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        instance = next(self.generator)
        MIP_model = instance.as_pyscipopt()
        MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
        instance_name = MIP_model.getProbName()
        print('\n')
        print(instance_name)

        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        valid, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)
        # conti = -1
        # if self.incumbent_mode == 'rootsol' and self.instance_type == 'independentset':
        #     conti = 196

        loss_instance = 0
        accu_instance = 0
        if valid:
            if index_instance > -1:
                gc.collect()
                observation, _, _, done, _ = self.env.reset(instance)
                del observation
                # print(observation)

                if self.incumbent_mode == 'firstsol':
                    action = {'limits/solutions': 1}
                elif self.incumbent_mode == 'rootsol':
                    action = {'limits/nodes': 1}  #
                sample_observation, _, _, done, _ = self.env.step(action)

                # print(sample_observation)
                graph = BipartiteNodeData(sample_observation.constraint_features,
                                          sample_observation.edge_features.indices,
                                          sample_observation.edge_features.values,
                                          sample_observation.variable_features)

                # We must tell pytorch geometric how many nodes there are, for indexing purposes
                graph.num_nodes = sample_observation.constraint_features.shape[0] + \
                                  sample_observation.variable_features.shape[
                                      0]

                # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
                # MIP_model = instance

                # MIP_model.optimize()
                # print("Status:", MIP_model.getStatus())
                # print("best obj: ", MIP_model.getObjVal())
                # print("Solving time: ", MIP_model.getSolvingTime())

                initial_obj = MIP_model.getSolObjVal(incumbent_solution)
                print("Initial obj before LB: {}".format(initial_obj))

                binary_supports = binary_support(MIP_model, incumbent_solution)
                print('binary support: ', binary_supports)

                model_gnn = GNNPolicy()

                model_gnn.load_state_dict(torch.load(
                    self.saved_gnn_directory + 'trained_params_' + self.train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '.pth'))

                # model_gnn.load_state_dict(torch.load(
                #      'trained_params_' + self.instance_type + '.pth'))

                k_model = model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                    graph.variable_features)

                k_pred = k_model.item() * n_binvars
                print('GNN prediction: ', k_model.item())

                if self.is_symmetric == False:
                    k_pred = k_model.item() * binary_supports

                del k_model
                del graph
                del sample_observation
                del model_gnn

                # create a copy of MIP
                MIP_model.resetParams()

                MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
                    problemName='GNN_reset',
                    origcopy=False)

                print('MIP copies are created')

                MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
                                                          MIP_copy_vars3)

                print('incumbent solution is copied to MIP copies')
                MIP_model.freeProb()
                del MIP_model
                del incumbent_solution

                # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
                lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                                           node_time_limit=node_time_limit,
                                           total_time_limit=total_time_limit)
                status, obj_best, elapsed_time, lb_bits_pred_reset, times_pred_rest, objs_pred_rest, loss_instance, accu_instance = lb_model3.mdp_localbranch(
                    is_symmetric=self.is_symmetric,
                    reset_k_at_2nditeration=reset_k_at_2nditeration,
                    policy=policy,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    samples_dir=self.imitation_samples_directory)

                print("Instance:", MIP_model_copy3.getProbName())
                print("Status of LB: ", status)
                print("Best obj of LB: ", obj_best)
                print("Solving time: ", elapsed_time)
                print('\n')

                MIP_model_copy3.freeProb()
                del sol_MIP_copy3
                del MIP_model_copy3

                del objs_pred_rest
                del times_pred_rest

                del lb_model3

            index_instance += 1
        del instance
        return index_instance, loss_instance, accu_instance

    def execute_rl4localbranch(self, test_instance_size='-small', total_time_limit=60, node_time_limit=10,
                       reset_k_at_2nditeration=False, lr=0.001, n_epochs=20):

        self.train_dataset = self.instance_type + self.instance_size
        self.evaluation_dataset = self.instance_type + test_instance_size

        self.k_baseline = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_gnn_directory = './result/saved_models/'

        self.imitation_samples_directory = self.directory + 'imitation_samples' + '/'
        pathlib.Path(self.imitation_samples_directory).mkdir(parents=True, exist_ok=True)

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # self.directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        # pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        rl_policy = SimplePolicy(7, 4)
        optimizer = torch.optim.Adam(rl_policy.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        rl_policy = rl_policy.to(device)
        criterion = criterion.to(device)

        loss = []
        accu = []
        epochs = []

        for epoch in range(n_epochs):
            print(f"Epoch {epoch}")
            if epoch == 0:
                optimizer = None

            self.generator = generator_switcher(self.evaluation_dataset)
            self.generator.seed(self.seed)
            index_instance = 0
            loss_epoch = 0
            accu_epoch = 0
            size_trainset = 10
            while index_instance < size_trainset:
                # train_previous rl_policy
                index_instance, loss_instance, accu_instance = self.rltrain_per_instance(node_time_limit=node_time_limit,
                                                               total_time_limit=total_time_limit,
                                                               index_instance=index_instance,
                                                               reset_k_at_2nditeration=reset_k_at_2nditeration,
                                                               policy=None,
                                                               optimizer=optimizer,
                                                               criterion=criterion,
                                                               device=device
                                                           )
            #     loss_epoch += loss_instance
            #     accu_epoch += accu_instance
            #
            # loss_epoch /= size_trainset
            # accu_epoch /= size_trainset
            #
            # epochs.append(epoch)
            # loss.append(loss_epoch)
            # accu.append(accu_epoch)
            #
            # epochs_np = np.array(epochs).reshape(-1)
            # loss_np = np.array(loss).reshape(-1)
            # accu_np = np.array(accu).reshape(-1)
            #
            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(2, 1, figsize=(8, 6.4))
            # fig.suptitle("Train: loss and imitation accuracy")
            # fig.subplots_adjust(top=0.5)
            # ax[0].set_title('learning rate = ' + str(lr), loc='right')
            # ax[0].plot(epochs_np, loss_np, label='loss')
            # ax[0].set_xlabel('epoch')
            # ax[0].set_ylabel("loss")
            #
            # ax[1].plot(epochs_np, accu_np, label='accuracy')
            # ax[1].set_xlabel('epoch')
            # ax[1].set_ylabel("accuray")
            # ax[1].set_ylim([0, 1.1])
            # ax[1].legend()
            # plt.show()
            #
            # print(f"Train loss: {loss_epoch:0.6f}")
            # print(f"Train accu: {accu_epoch:0.6f}")

    def load_dataset(self, test_dataset_directory=None):

        if test_dataset_directory is not None:
            self.imitation_samples_directory = test_dataset_directory
        else:
            self.imitation_samples_directory = self.directory + 'imitation_samples' + '/'
            pathlib.Path(self.imitation_samples_directory).mkdir(parents=True, exist_ok=True)

        filename = 'imitation_*.pkl'
        # print(filename)
        sample_files = [str(path) for path in pathlib.Path(self.imitation_samples_directory).glob(filename)]
        train_files = sample_files[:int(0.7 * len(sample_files))]
        valid_files = sample_files[int(0.7 * len(sample_files)):int(0.8 * len(sample_files))]
        test_files =  sample_files[int(0.8 * len(sample_files)):]

        train_data = ImitationLbDataset(train_files)

        # state, lab = train_data.__getitem__(0)
        # print(state.shape)
        # print(lab.shape)

        train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
        valid_data = ImitationLbDataset(valid_files)
        valid_loader = DataLoader(valid_data, batch_size=4, shuffle=True)
        test_data = ImitationLbDataset(test_files)
        test_loader = DataLoader(test_data, batch_size=4, shuffle=True)

        return train_loader, valid_loader, test_loader

    def train(self, policy, data_loader, optimizer=None, criterion=None, device=None):
        """
        training function
        :param gnn_model:
        :param data_loader:
        :param optimizer:
        :return:
        """
        loss_epoch = 0
        accu_epoch = 0
        with torch.set_grad_enabled(optimizer is not None):
            for (state, label) in data_loader:

                state.to(device)
                label.to(device)
                label = label.view(-1)

                k_pred = policy(state)
                loss = criterion(k_pred, label)
                accu = imitation_accuracy(k_pred, label)

                if optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                loss_epoch += loss.item()
                accu_epoch += accu.item()
        loss_mean = loss_epoch / len(data_loader)
        accu_mean = accu_epoch / len(data_loader)

        return loss_mean, accu_mean

    def test(self, policy, data_loader, criterion, device):

        loss_epoch = 0
        accu_epoch = 0
        for (state, label) in data_loader:
            state.to(device)
            label.to(device)
            label = label.view(-1)

            k_pred = policy(state)
            loss = criterion(k_pred, label)
            accu = imitation_accuracy(k_pred, label)

            loss_epoch += loss.item()
            accu_epoch += accu.item()

        loss_mean = loss_epoch / len(data_loader)
        accu_mean = accu_epoch / len(data_loader)

        return loss_mean, accu_mean

    def execute_imitation(self, lr=0.01, n_epochs=20):

        saved_gnn_directory = './result/saved_models/'
        pathlib.Path(saved_gnn_directory).mkdir(parents=True, exist_ok=True)

        train_loaders = {}
        val_loaders = {}
        test_loaders = {}

        # load the small dataset
        small_dataset = self.instance_type + self.instance_size
        self.imitation_samples_directory = self.directory + 'imitation_samples' + '/'

        train_loader, valid_loader, test_loader = self.load_dataset(test_dataset_directory=self.imitation_samples_directory)
        train_loaders[small_dataset] = train_loader
        val_loaders[small_dataset] = valid_loader
        test_loaders[small_dataset] = test_loader

        rl_policy = SimplePolicy(7, 4)
        optimizer = torch.optim.Adam(rl_policy.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        rl_policy = rl_policy.to(device)
        criterion = criterion.to(device)

        loss = []
        accu = []
        epochs = []

        train_dataset = small_dataset
        valid_dataset = small_dataset
        test_dataset = small_dataset
        # LEARNING_RATE = 0.0000001  # setcovering:0.0000005 cap-loc: 0.00000005 independentset: 0.0000001

        for epoch in range(n_epochs):
            print(f"Epoch {epoch}")

            if epoch == 0:
                optim = None
            else:
                optim = optimizer

            train_loader = train_loaders[train_dataset]
            train_loss, train_accu = self.train(rl_policy, train_loader, optimizer=optim, criterion=criterion, device=device)
            print(f"Train loss: {train_loss:0.6f}")
            print(f"Train accu: {train_accu:0.6f}")

            # torch.save(model_gnn.state_dict(), 'trained_params_' + train_dataset + '.pth')
            # model_gnn2.load_state_dict(torch.load('trained_params_' + train_dataset + '.pth'))

            valid_loader = val_loaders[valid_dataset]
            valid_loss, valid_accu = self.train(rl_policy, valid_loader, optimizer=None, criterion=criterion, device=device)
            print(f"Valid loss: {valid_loss:0.6f}")
            print(f"Valid accu: {valid_accu:0.6f}")

            test_loader = test_loaders[test_dataset]
            test_loss, test_accu = self.test(policy=rl_policy, data_loader=test_loader, criterion=criterion, device=device)

            loss.append(test_loss)
            accu.append(test_accu)
            epochs.append(epoch)

        loss_np = np.array(loss).reshape(-1)
        accu_np = np.array(accu).reshape(-1)
        epochs_np = np.array(epochs).reshape(-1)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(2, 1, figsize=(8, 6.4))
        fig.suptitle("Test: loss and imitation accuracy")
        fig.subplots_adjust(top=0.5)
        ax[0].set_title('learning rate = ' + str(lr), loc='right')
        ax[0].plot(epochs_np, loss_np, label='loss')
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel("loss")

        ax[1].plot(epochs_np, accu_np, label='accuracy')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel("accuray")
        ax[1].set_ylim([0, 1.1])
        ax[1].legend()
        plt.show()

        torch.save(rl_policy.state_dict(),
                   saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_imitation.pth')

    def evaluate_lb_per_instance(self, node_time_limit, total_time_limit, index_instance, reset_k_at_2nditeration=False, policy=None,
                             criterion=None, device=None):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        instance = next(self.generator)
        MIP_model = instance.as_pyscipopt()
        MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
        instance_name = MIP_model.getProbName()
        print('\n')
        print(instance_name)

        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        valid, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)
        conti = 99
        # if self.incumbent_mode == 'rootsol' and self.instance_type == 'independentset':
        #     conti = 196

        if valid:
            if index_instance > 99 and index_instance > conti:
                gc.collect()
                observation, _, _, done, _ = self.env.reset(instance)
                del observation
                # print(observation)

                if self.incumbent_mode == 'firstsol':
                    action = {'limits/solutions': 1}
                elif self.incumbent_mode == 'rootsol':
                    action = {'limits/nodes': 1}  #
                sample_observation, _, _, done, _ = self.env.step(action)

                # print(sample_observation)
                graph = BipartiteNodeData(sample_observation.constraint_features,
                                          sample_observation.edge_features.indices,
                                          sample_observation.edge_features.values,
                                          sample_observation.variable_features)

                # We must tell pytorch geometric how many nodes there are, for indexing purposes
                graph.num_nodes = sample_observation.constraint_features.shape[0] + \
                                  sample_observation.variable_features.shape[
                                      0]

                # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
                # MIP_model = instance

                # MIP_model.optimize()
                # print("Status:", MIP_model.getStatus())
                # print("best obj: ", MIP_model.getObjVal())
                # print("Solving time: ", MIP_model.getSolvingTime())

                initial_obj = MIP_model.getSolObjVal(incumbent_solution)
                print("Initial obj before LB: {}".format(initial_obj))

                binary_supports = binary_support(MIP_model, incumbent_solution)
                print('binary support: ', binary_supports)

                model_gnn = GNNPolicy()

                model_gnn.load_state_dict(torch.load(
                    self.saved_gnn_directory + 'trained_params_' + self.train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '.pth'))

                # model_gnn.load_state_dict(torch.load(
                #      'trained_params_' + self.instance_type + '.pth'))

                k_model = model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                    graph.variable_features)

                k_pred = k_model.item() * n_binvars
                print('GNN prediction: ', k_model.item())

                if self.is_symmetric == False:
                    k_pred = k_model.item() * binary_supports

                k_pred = np.ceil(k_pred)

                del k_model
                del graph
                del sample_observation
                del model_gnn

                # create a copy of MIP
                MIP_model.resetParams()
                # MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
                #     problemName='Baseline', origcopy=False)
                MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
                    problemName='GNN',
                    origcopy=False)
                MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
                    problemName='GNN+reset',
                    origcopy=False)

                print('MIP copies are created')

                # MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent_solution,
                #                                         MIP_copy_vars)
                MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent_solution,
                                                          MIP_copy_vars2)
                MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
                                                          MIP_copy_vars3)

                print('incumbent solution is copied to MIP copies')
                MIP_model.freeProb()
                del MIP_model
                del incumbent_solution

                # sol = MIP_model_copy.getBestSol()
                # initial_obj = MIP_model_copy.getSolObjVal(sol)
                # print("Initial obj before LB: {}".format(initial_obj))

                # # execute local branching baseline heuristic by Fischetti and Lodi
                # lb_model = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy, k=self.k_baseline,
                #                           node_time_limit=node_time_limit,
                #                           total_time_limit=total_time_limit)
                # status, obj_best, elapsed_time, lb_bits, times, objs = lb_model.search_localbranch(is_symmeric=self.is_symmetric,
                #                                                              reset_k_at_2nditeration=False)
                # print("Instance:", MIP_model_copy.getProbName())
                # print("Status of LB: ", status)
                # print("Best obj of LB: ", obj_best)
                # print("Solving time: ", elapsed_time)
                # print('\n')
                #
                # MIP_model_copy.freeProb()
                # del sol_MIP_copy
                # del MIP_model_copy

                # sol = MIP_model_copy2.getBestSol()
                # initial_obj = MIP_model_copy2.getSolObjVal(sol)
                # print("Initial obj before LB: {}".format(initial_obj))

                # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
                lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                                           node_time_limit=node_time_limit,
                                           total_time_limit=total_time_limit)
                status, obj_best, elapsed_time, lb_bits_pred_reset, times_reset_imitation, objs_reset_imitation, loss_instance, accu_instance = lb_model3.mdp_localbranch(
                    is_symmetric=self.is_symmetric,
                    reset_k_at_2nditeration=reset_k_at_2nditeration,
                    policy=policy,
                    optimizer=None,
                    criterion=criterion,
                    device=device
                    )
                print("Instance:", MIP_model_copy3.getProbName())
                print("Status of LB: ", status)
                print("Best obj of LB: ", obj_best)
                print("Solving time: ", elapsed_time)
                print('\n')

                MIP_model_copy3.freeProb()
                del sol_MIP_copy3
                del MIP_model_copy3

                # execute local branching with 1. first k predicted by GNN; 2. for 2nd iteration of lb, continue lb algorithm with no further injection
                lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=k_pred,
                                           node_time_limit=node_time_limit,
                                           total_time_limit=total_time_limit)
                status, obj_best, elapsed_time, lb_bits_pred, times_reset_vanilla, objs_reset_vanilla, _, _ = lb_model2.mdp_localbranch(
                    is_symmetric=self.is_symmetric,
                    reset_k_at_2nditeration=True,
                    policy=None,
                    optimizer=None,
                    criterion=None,
                    device=None
                )

                print("Instance:", MIP_model_copy2.getProbName())
                print("Status of LB: ", status)
                print("Best obj of LB: ", obj_best)
                print("Solving time: ", elapsed_time)
                print('\n')

                MIP_model_copy2.freeProb()
                del sol_MIP_copy2
                del MIP_model_copy2

                data = [objs_reset_vanilla, times_reset_vanilla, objs_reset_imitation, times_reset_imitation]
                filename = f'{self.directory_lb_test}lb-test-{instance_name}.pkl'  # instance 100-199
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)

                del data
                del lb_model2
                del lb_model3

            index_instance += 1
        del instance
        return index_instance

    def evaluate_localbranching(self, test_instance_size='-small', total_time_limit=60, node_time_limit=30, reset_k_at_2nditeration=False, greedy=True):

        self.train_dataset = self.instance_type + self.instance_size
        self.evaluation_dataset = self.instance_type + test_instance_size

        self.generator = generator_switcher(self.evaluation_dataset)
        self.generator.seed(self.seed)

        self.k_baseline = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_gnn_directory = './result/saved_models/'

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/'
        self.directory_lb_test = directory + 'imitation4lb-from-' + self.incumbent_mode + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        rl_policy = SimplePolicy(7, 4)

        rl_policy.load_state_dict(torch.load(
            self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_imitation.pth'))

        rl_policy.eval()
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        greedy = greedy
        rl_policy = rl_policy.to(device)
        agent = AgentReinforce(rl_policy, device, greedy, None, 0.0)

        index_instance = 0
        while index_instance < 200:
            index_instance = self.evaluate_lb_per_instance(node_time_limit=node_time_limit, total_time_limit=total_time_limit, index_instance=index_instance, reset_k_at_2nditeration=reset_k_at_2nditeration,
                                                           policy=agent, criterion=criterion, device=device
                                                           )

    def primal_integral(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/'
        directory_lb_test = directory + 'imitation4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/' + 'rl/'
            directory_lb_test_2 = directory_2 + 'imitation4lb-from-' +  'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/' + 'rl/'
            directory_lb_test_2 = directory_2 + 'imitation4lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        # primal_int_baselines = []
        primal_int_reset_vanillas = []
        primal_in_reset_imitations = []
        # primal_gap_final_baselines = []
        primal_gap_final_reset_vanillas = []
        primal_gap_final_reset_imitations = []
        # steplines_baseline = []
        steplines_reset_vanillas = []
        steplines_reset_imitations = []

        for i in range(100,200):
            if not (i == 148 or i ==113 or i == 110 or i ==199 or i== 198 or i == 134 or i == 123 or i == 116):
                instance_name = self.instance_type + '-' + str(i)  # instance 100-199

                filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'

                with gzip.open(filename, 'rb') as f:
                    data = pickle.load(f)
                objs_reset_vanilla, times_reset_vanilla, objs_reset_imitation, times_reset_imitation = data  # objs contains objs of a single instance of a lb test

                filename_2 = f'{directory_lb_test_2}lb-test-{instance_name}.pkl'

                with gzip.open(filename_2, 'rb') as f:
                    data = pickle.load(f)
                objs_reset_vanilla_2, times_reset_vanilla_2, objs_reset_imitation_2, times_reset_imitation_2 = data  # objs contains objs of a single instance of a lb test

                a = [objs_reset_vanilla.min(), objs_reset_imitation.min(), objs_reset_vanilla_2.min(), objs_reset_imitation_2.min()]
                # a = [objs.min(), objs_reset_vanilla.min(), objs_reset_imitation.min()]
                obj_opt = np.amin(a)

                # # compute primal gap for baseline localbranching run
                # # if times[-1] < total_time_limit:
                # times = np.append(times, total_time_limit)
                # objs = np.append(objs, objs[-1])
                #
                # gamma_baseline = np.zeros(len(objs))
                # for j in range(len(objs)):
                #     if objs[j] == 0 and obj_opt == 0:
                #         gamma_baseline[j] = 0
                #     elif objs[j] * obj_opt < 0:
                #         gamma_baseline[j] = 1
                #     else:
                #         gamma_baseline[j] = np.abs(objs[j] - obj_opt) / np.maximum(np.abs(objs[j]), np.abs(obj_opt)) #
                #
                # # compute the primal gap of last objective
                # primal_gap_final_baseline = np.abs(objs[-1] - obj_opt) / np.abs(obj_opt)
                # primal_gap_final_baselines.append(primal_gap_final_baseline)
                #
                # # create step line
                # stepline_baseline = interp1d(times, gamma_baseline, 'previous')
                # steplines_baseline.append(stepline_baseline)
                #
                # # compute primal integral
                # primal_int_baseline = 0
                # for j in range(len(objs) - 1):
                #     primal_int_baseline += gamma_baseline[j] * (times[j + 1] - times[j])
                # primal_int_baselines.append(primal_int_baseline)
                #


                # lb-gnn
                # if times_reset_vanilla[-1] < total_time_limit:
                times_reset_vanilla = np.append(times_reset_vanilla, total_time_limit)
                objs_reset_vanilla = np.append(objs_reset_vanilla, objs_reset_vanilla[-1])

                gamma_reset_vanilla = np.zeros(len(objs_reset_vanilla))
                for j in range(len(objs_reset_vanilla)):
                    if objs_reset_vanilla[j] == 0 and obj_opt == 0:
                        gamma_reset_vanilla[j] = 0
                    elif objs_reset_vanilla[j] * obj_opt < 0:
                        gamma_reset_vanilla[j] = 1
                    else:
                        gamma_reset_vanilla[j] = np.abs(objs_reset_vanilla[j] - obj_opt) / np.maximum(np.abs(objs_reset_vanilla[j]), np.abs(obj_opt)) #

                primal_gap_final_vanilla = np.abs(objs_reset_vanilla[-1] - obj_opt) / np.abs(obj_opt)
                primal_gap_final_reset_vanillas.append(primal_gap_final_vanilla)

                stepline_reset_vanilla = interp1d(times_reset_vanilla, gamma_reset_vanilla, 'previous')
                steplines_reset_vanillas.append(stepline_reset_vanilla)

                #
                # t = np.linspace(start=0.0, stop=total_time_limit, num=1001)
                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(8, 6.4))
                # fig.suptitle("Test Result: comparison of primal gap")
                # fig.subplots_adjust(top=0.5)
                # # ax.set_title(instance_name, loc='right')
                # ax.plot(t, stepline_baseline(t), label='lb baseline')
                # ax.plot(t, stepline_reset_vanilla(t), label='lb with k predicted')
                # ax.set_xlabel('time /s')
                # ax.set_ylabel("objective")
                # ax.legend()
                # plt.show()

                # compute primal interal
                primal_int_reset_vanilla = 0
                for j in range(len(objs_reset_vanilla) - 1):
                    primal_int_reset_vanilla += gamma_reset_vanilla[j] * (times_reset_vanilla[j + 1] - times_reset_vanilla[j])
                primal_int_reset_vanillas.append(primal_int_reset_vanilla)

                # lb-gnn-reset
                times_reset_imitation = np.append(times_reset_imitation, total_time_limit)
                objs_reset_imitation = np.append(objs_reset_imitation, objs_reset_imitation[-1])

                gamma_reset_imitation = np.zeros(len(objs_reset_imitation))
                for j in range(len(objs_reset_imitation)):
                    if objs_reset_imitation[j] == 0 and obj_opt == 0:
                        gamma_reset_imitation[j] = 0
                    elif objs_reset_imitation[j] * obj_opt < 0:
                        gamma_reset_imitation[j] = 1
                    else:
                        gamma_reset_imitation[j] = np.abs(objs_reset_imitation[j] - obj_opt) / np.maximum(np.abs(objs_reset_imitation[j]), np.abs(obj_opt)) #

                primal_gap_final_imitation = np.abs(objs_reset_imitation[-1] - obj_opt) / np.abs(obj_opt)
                primal_gap_final_reset_imitations.append(primal_gap_final_imitation)

                stepline_reset_imitation = interp1d(times_reset_imitation, gamma_reset_imitation, 'previous')
                steplines_reset_imitations.append(stepline_reset_imitation)

                # compute primal interal
                primal_int_reset_imitation = 0
                for j in range(len(objs_reset_imitation) - 1):
                    primal_int_reset_imitation += gamma_reset_imitation[j] * (times_reset_imitation[j + 1] - times_reset_imitation[j])
                primal_in_reset_imitations.append(primal_int_reset_imitation)

                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(8, 6.4))
                # fig.suptitle("Test Result: comparison of objective")
                # fig.subplots_adjust(top=0.5)
                # ax.set_title(instance_name, loc='right')
                # ax.plot(times, objs, label='lb baseline')
                # ax.plot(times_reset_vanilla, objs_reset_vanilla, label='lb with k predicted')
                # ax.set_xlabel('time /s')
                # ax.set_ylabel("objective")
                # ax.legend()
                # plt.show()
                #
                # plt.close('all')
                # plt.clf()
                # fig, ax = plt.subplots(figsize=(8, 6.4))
                # fig.suptitle("Test Result: comparison of primal gap")
                # fig.subplots_adjust(top=0.5)
                # ax.set_title(instance_name, loc='right')
                # ax.plot(times, gamma_baseline, label='lb baseline')
                # ax.plot(times_reset_vanilla, gamma_reset_vanilla, label='lb with k predicted')
                # ax.set_xlabel('time /s')
                # ax.set_ylabel("objective")
                # ax.legend()
                # plt.show()


        # primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_reset_vanilla = np.array(primal_int_reset_vanillas).reshape(-1)
        primal_in_reset_imitation = np.array(primal_in_reset_imitations).reshape(-1)

        # primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_reset_vanilla = np.array(primal_gap_final_reset_vanillas).reshape(-1)
        primal_gap_final_reset_imitation = np.array(primal_gap_final_reset_imitations).reshape(-1)

        # avarage primal integral over test dataset
        # primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_reset_vanilla_ave = primal_int_reset_vanilla.sum() / len(primal_int_reset_vanilla)
        primal_int_reset_imitation_ave = primal_in_reset_imitation.sum() / len(primal_in_reset_imitation)

        # primal_gap_final_baselines = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_reset_vanilla = primal_gap_final_reset_vanilla.sum() / len(primal_gap_final_reset_vanilla)
        primal_gap_final_reset_imitation = primal_gap_final_reset_imitation.sum() / len(primal_gap_final_reset_imitation)

        print(self.instance_type + self.instance_size)
        print(self.incumbent_mode + 'Solution')
        # print('baseline primal integral: ', primal_int_base_ave)
        print('baseline primal integral: ', primal_int_reset_vanilla_ave)
        print('imitation primal integral: ', primal_int_reset_imitation_ave)
        print('\n')
        # print('baseline primal gap: ',primal_gap_final_baselines)
        print('baseline primal gap: ', primal_gap_final_reset_vanilla)
        print('imitation primal gap: ', primal_gap_final_reset_imitation)

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)

        # primalgaps_baseline = None
        # for n, stepline_baseline in enumerate(steplines_baseline):
        #     primal_gap = stepline_baseline(t)
        #     if n==0:
        #         primalgaps_baseline = primal_gap
        #     else:
        #         primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        # primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)

        primalgaps_reset_vanilla = None
        for n, stepline_reset_vanilla in enumerate(steplines_reset_vanillas):
            primal_gap = stepline_reset_vanilla(t)
            if n == 0:
                primalgaps_reset_vanilla = primal_gap
            else:
                primalgaps_reset_vanilla = np.vstack((primalgaps_reset_vanilla, primal_gap))
        primalgap_reset_vanilla_ave = np.average(primalgaps_reset_vanilla, axis=0)

        primalgaps_reset_imitation = None
        for n, stepline_reset_imitation in enumerate(steplines_reset_imitations):
            primal_gap = stepline_reset_imitation(t)
            if n == 0:
                primalgaps_reset_imitation = primal_gap
            else:
                primalgaps_reset_imitation = np.vstack((primalgaps_reset_imitation, primal_gap))
        primalgap_reset_imitation_ave = np.average(primalgaps_reset_imitation, axis=0)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle("Normalized primal gap")
        # fig.subplots_adjust(top=0.5)
        ax.set_title(self.instance_type + '-' + self.incumbent_mode, loc='right')
        # ax.plot(t, primalgap_baseline_ave, label='lb-baseline')
        ax.plot(t, primalgap_reset_vanilla_ave, label='lb-gnn-baseline')
        ax.plot(t, primalgap_reset_imitation_ave,'--', label='lb-gnn-imitation')
        ax.set_xlabel('time /s')
        ax.set_ylabel("normalized primal gap")
        ax.legend()
        plt.show()

class RlLocalbranch(MlLocalbranch):
    def __init__(self, instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed=100):
        super().__init__(instance_type, instance_size, lbconstraint_mode, incumbent_mode, seed)
        self.alpha = 0.01
        self.gamma = 0.99
        self.eps = np.finfo(np.float32).eps.item()

    def mdp_localbranch(self, localbranch=None, is_symmetric=True, reset_k_at_2nditeration=False, agent=None, optimizer=None, device=None):


        # self.total_time_limit = total_time_limit
        localbranch.total_time_available = localbranch.total_time_limit
        localbranch.first = False
        localbranch.diversify = False
        localbranch.t_node = localbranch.default_node_time_limit
        localbranch.div = 0
        localbranch.is_symmetric = is_symmetric
        localbranch.reset_k_at_2nditeration = reset_k_at_2nditeration
        lb_bits = 0
        t_list = []
        obj_list = []
        lb_bits_list = []

        lb_bits_list.append(lb_bits)
        t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
        obj_list.append(localbranch.MIP_obj_best)

        k_action = localbranch.actions['unchange']
        t_action = localbranch.actions['unchange']

        # initialize the env to state_0
        lb_bits += 1
        state, reward, done, _ = localbranch.step_localbranch(k_action=k_action, t_action=t_action, lb_bits=lb_bits)
        localbranch.MIP_obj_init = localbranch.MIP_obj_best
        lb_bits_list.append(lb_bits)
        t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
        obj_list.append(localbranch.MIP_obj_best)


        if (not done) and reset_k_at_2nditeration:
            lb_bits += 1
            localbranch.default_k = 20
            if not localbranch.is_symmetric:
                localbranch.default_k = 10
            localbranch.k = localbranch.default_k
            localbranch.diversify = False
            localbranch.first = False

            state, reward, done, _ = localbranch.step_localbranch(k_action=k_action, t_action=t_action,
                                                                   lb_bits=lb_bits)
            localbranch.MIP_obj_init = localbranch.MIP_obj_best
            lb_bits_list.append(lb_bits)
            t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
            obj_list.append(localbranch.MIP_obj_best)

        while not done:  # and localbranch.div < localbranch.div_max
            lb_bits += 1

            k_vanilla, t_action = localbranch.policy_vanilla(state)

            # data_sample = [state, k_vanilla]
            #
            # filename = f'{samples_dir}imitation_{localbranch.MIP_model.getProbName()}_{lb_bits}.pkl'
            #
            # with gzip.open(filename, 'wb') as f:
            #     pickle.dump(data_sample, f)

            k_action = k_vanilla
            if agent is not None:
                k_action = agent.select_action(state)

                # # for online learning, update policy
                # if optimizer is not None:
                #     optimizer.zero_grad()
                #     loss.backward()
                #     optimizer.step()

            # execute one iteration of LB, get the state and rewards

            state, reward, done, _ = localbranch.step_localbranch(k_action=k_action, t_action=t_action, lb_bits=lb_bits)

            if agent is not None:
                agent.rewards.append(reward)

            lb_bits_list.append(lb_bits)
            t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
            obj_list.append(localbranch.MIP_obj_best)

        print(
            'K_final: {:.0f}'.format(localbranch.k),
            'div_final: {:.0f}'.format(localbranch.div)
        )

        localbranch.solve_rightbranch()
        t_list.append(localbranch.total_time_limit - localbranch.total_time_available)
        obj_list.append(localbranch.MIP_obj_best)

        status = localbranch.MIP_model.getStatus()
        # if status == "optimal" or status == "bestsollimit":
        #     localbranch.MIP_obj_best = localbranch.MIP_model.getObjVal()

        elapsed_time = localbranch.total_time_limit - localbranch.total_time_available

        lb_bits_list = np.array(lb_bits_list).reshape(-1)
        times_list = np.array(t_list).reshape(-1)
        objs_list = np.array(obj_list).reshape(-1)

        del localbranch.subMIP_sol_best
        del localbranch.MIP_sol_bar
        del localbranch.MIP_sol_best


        return status, localbranch.MIP_obj_best, elapsed_time, lb_bits_list, times_list, objs_list, agent

    def train_agent_per_instance(self, node_time_limit, total_time_limit, index_instance,
                                 reset_k_at_2nditeration=False, agent=None, optimizer=None,
                                 device=None):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        instance = next(self.generator)
        MIP_model = instance.as_pyscipopt()
        MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
        instance_name = MIP_model.getProbName()
        print('\n')
        print(instance_name)

        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        valid, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)
        # conti = -1
        # if self.incumbent_mode == 'rootsol' and self.instance_type == 'independentset':
        #     conti = 196

        if valid:
            if index_instance > -1:
                gc.collect()
                observation, _, _, done, _ = self.env.reset(instance)
                del observation
                # print(observation)

                if self.incumbent_mode == 'firstsol':
                    action = {'limits/solutions': 1}
                elif self.incumbent_mode == 'rootsol':
                    action = {'limits/nodes': 1}  #
                sample_observation, _, _, done, _ = self.env.step(action)

                # print(sample_observation)
                graph = BipartiteNodeData(sample_observation.constraint_features,
                                          sample_observation.edge_features.indices,
                                          sample_observation.edge_features.values,
                                          sample_observation.variable_features)

                # We must tell pytorch geometric how many nodes there are, for indexing purposes
                graph.num_nodes = sample_observation.constraint_features.shape[0] + \
                                  sample_observation.variable_features.shape[
                                      0]

                # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
                # MIP_model = instance

                # MIP_model.optimize()
                # print("Status:", MIP_model.getStatus())
                # print("best obj: ", MIP_model.getObjVal())
                # print("Solving time: ", MIP_model.getSolvingTime())

                initial_obj = MIP_model.getSolObjVal(incumbent_solution)
                print("Initial obj before LB: {}".format(initial_obj))

                binary_supports = binary_support(MIP_model, incumbent_solution)
                print('binary support: ', binary_supports)

                model_gnn = GNNPolicy()

                model_gnn.load_state_dict(torch.load(
                    self.saved_gnn_directory + 'trained_params_' + self.train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '.pth'))

                # model_gnn.load_state_dict(torch.load(
                #      'trained_params_' + self.instance_type + '.pth'))

                k_model = model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                    graph.variable_features)

                k_pred = k_model.item() * n_binvars
                print('GNN prediction: ', k_model.item())

                if self.is_symmetric == False:
                    k_pred = k_model.item() * binary_supports

                del k_model
                del graph
                del sample_observation
                del model_gnn

                # create a copy of MIP
                MIP_model.resetParams()

                MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
                    problemName='GNN_reset',
                    origcopy=False)

                print('MIP copies are created')

                MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
                                                          MIP_copy_vars3)

                print('incumbent solution is copied to MIP copies')
                MIP_model.freeProb()
                del MIP_model
                del incumbent_solution

                # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
                lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=self.k_baseline, # k_pred
                                           node_time_limit=node_time_limit,
                                           total_time_limit=total_time_limit,
                                           is_symmetric=self.is_symmetric)
                status, obj_best, elapsed_time, lb_bits_pred_reset, times_pred_reset, objs_pred_reset, agent = self.mdp_localbranch(
                    localbranch=lb_model3,
                    is_symmetric=self.is_symmetric,
                    reset_k_at_2nditeration=reset_k_at_2nditeration,
                    agent=agent,
                    optimizer=optimizer,
                    device=device)

                print("Instance:", MIP_model_copy3.getProbName())
                print("Status of LB: ", status)
                print("Best obj of LB: ", obj_best)
                print("Solving time: ", elapsed_time)
                print('\n')

                data = [objs_pred_reset, times_pred_reset]
                primal_integral, primal_gap_final, stepline = self.compute_primal_integral(times_pred_reset, objs_pred_reset, total_time_limit)

                MIP_model_copy3.freeProb()
                del sol_MIP_copy3
                del MIP_model_copy3

                del objs_pred_reset
                del times_pred_reset

                del lb_model3
                del stepline

            index_instance += 1
        else:
            primal_integral = 0.0
            primal_gap_final = 0.0
        del instance
        return index_instance, agent, primal_integral, primal_gap_final

    def update_agent(self, agent, optimizer):

        R = 0
        policy_losses = []
        returns = []
        # calculate the return
        for r in agent.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0,R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        # calculate loss
        with torch.set_grad_enabled(optimizer is not None):
            for log_prob, Return in zip(agent.log_probs, returns):
                policy_losses.append(-log_prob * Return)

            # optimize policy network
            if optimizer is not None:
                optimizer.zero_grad()
                policy_losses = torch.cat(policy_losses).sum()
                policy_losses.backward()
                optimizer.step()

        del agent.rewards[:]
        del agent.log_probs[:]
        return agent, optimizer, R


    def train_agent(self, test_instance_size='-small', total_time_limit=60, node_time_limit=10,
                    reset_k_at_2nditeration=False, lr=0.001, n_epochs=20, epsilon=0, use_checkpoint=False):

        self.train_dataset = self.instance_type + self.instance_size
        self.evaluation_dataset = self.instance_type + test_instance_size

        self.k_baseline = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_gnn_directory = './result/saved_models/'
        self.saved_rlmodels_directory = self.saved_gnn_directory + 'rl_noimitation/'
        pathlib.Path( self.saved_rlmodels_directory).mkdir(parents=True, exist_ok=True)

        self.reinforce_train_directory = self.directory + 'rl/'+'reinforce/train/noimitation/'
        pathlib.Path(self.reinforce_train_directory).mkdir(parents=True, exist_ok=True)

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        # self.directory_lb_test = directory + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
        #     node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        # pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        rl_policy = SimplePolicy(7, 4)
        # rl_policy.load_state_dict(torch.load(
        #     # self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_reinforce_lr0.1_epsilon0.0_pre.pth'
        #     self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_imitation.pth'
        # ))
        rl_policy.train()

        optim = torch.optim.Adam(rl_policy.parameters(), lr=lr)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        greedy = False
        rl_policy = rl_policy.to(device)
        agent = AgentReinforce(rl_policy, device, greedy, optim, epsilon)

        returns = []
        epochs = []
        primal_integrals = []
        primal_gaps = []
        data = None
        epochs_np = None
        returns_np = None
        primal_integrals_np = None
        primal_gaps_np = None
        epoch_init = 0
        epoch_start = epoch_init  # 50
        epoch_end = epoch_start+n_epochs+1

        if use_checkpoint:
            checkpoint = torch.load(
                # self.saved_gnn_directory + 'checkpoint_simplepolicy_rl4lb_reinforce_lr' + str(lr) + '_epsilon' + str(epsilon) + '.pth'
                self.saved_rlmodels_directory + 'checkpoint_noregression_noimitation_reward3_simplepolicy_rl4lb_reinforce_lr' + str(
                    lr) + '_epsilon' + str(epsilon) + '.pth'
            )
            rl_policy.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            data = checkpoint['loss_data']
            epochs, returns, primal_integrals, primal_gaps = data
            rl_policy.train()

            epoch_start = checkpoint['epoch'] + 1
            epoch_end = epoch_start + n_epochs
            optimizer = optim

        for epoch in range(epoch_start,epoch_end):
            del data
            print(f"Epoch {epoch}")
            if epoch == epoch_init:
                optimizer = None
            elif epoch == epoch_init + 1:
                optimizer = optim

            self.generator = generator_switcher(self.evaluation_dataset)
            self.generator.seed(self.seed)
            index_instance = 0
            size_trainset = 5
            return_epoch = 0
            primal_integral_epoch = 0
            primal_gap_epoch = 0

            while index_instance < size_trainset:

                # train_previous rl_policy
                index_instance, agent, primal_integral, primal_gap_final = self.train_agent_per_instance(node_time_limit=node_time_limit,
                                                                                                    total_time_limit=total_time_limit,
                                                                                                    index_instance=index_instance,
                                                                                                    reset_k_at_2nditeration=reset_k_at_2nditeration,
                                                                                                    agent=agent,
                                                                                                    optimizer=optimizer,
                                                                                                    device=device
                                                                                                    )
                agent, optimizer, R = self.update_agent(agent, optimizer)
                return_epoch += R

                primal_integral_epoch += primal_integral
                primal_gap_epoch += primal_gap_final

            return_epoch = return_epoch/size_trainset
            primal_integral_epoch = primal_integral_epoch/size_trainset
            primal_gap_epoch = primal_gap_epoch/size_trainset

            returns.append(return_epoch)
            epochs.append(epoch)
            primal_integrals.append(primal_integral_epoch)
            primal_gaps.append(primal_gap_epoch)

            print(f"Return: {return_epoch:0.6f}")
            print(f"Primal ingtegral: {primal_integral_epoch:0.6f}")

            data = [epochs, returns, primal_integrals, primal_gaps]

            if epoch > 0 and epoch % 10 == 0:
                # filename = f'{self.reinforce_train_directory}lb-rl-train-checkpoint50-lr{str(lr)}-epsilon{str(epsilon)}.pkl'  # instance 100-199
                filename = f'{self.reinforce_train_directory}lb-rl-noregression-noimitation-reward3-train-lr{str(lr)}-epsilon{str(epsilon)}.pkl'  # instance 100-199
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)

                # save checkpoint
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': rl_policy.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss_data':data,
                            },
                           self.saved_rlmodels_directory + 'checkpoint_noregression_noimitation_reward3_simplepolicy_rl4lb_reinforce_lr' + str(lr) + '_epsilon' + str(epsilon) + '.pth'
                           # self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_reinforce_lr' + str(lr) + '_epsilon' + str(epsilon) + '.pth'
                           )
                # torch.save(rl_policy.state_dict(),
                #            # self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_reinforce-checkpoint50_lr' + str(lr) + '_epsilon' + str(epsilon) + '.pth'
                #            self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_reinforce_lr' + str(lr) +'_epsilon' + str(epsilon) + '.pth'
                #            )

        epochs_np = np.array(epochs).reshape(-1)
        returns_np = np.array(returns).reshape(-1)
        primal_integrals_np = np.array(primal_integrals).reshape(-1)
        primal_gaps_np = np.array(primal_gaps).reshape(-1)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(3, 1, figsize=(8, 6.4))
        fig.suptitle(self.train_dataset)
        fig.subplots_adjust(top=0.5)
        ax[0].set_title('lr= ' + str(lr) + ', epsilon=' + str(epsilon), loc='right')
        ax[0].plot(epochs_np, returns_np, label='loss')
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel("return")

        ax[1].plot(epochs_np, primal_integrals_np, label='primal ingegral')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel("primal integral")
        # ax[1].set_ylim([0, 1.1])
        ax[1].legend()

        ax[2].plot(epochs_np, primal_gaps_np, label='primal gap')
        ax[2].set_xlabel('epoch')
        ax[2].set_ylabel("primal gap")
        ax[2].legend()
        plt.show()


    def evaluate_lb_per_instance(self, node_time_limit, total_time_limit, index_instance, reset_k_at_2nditeration=False, policy=None,
                             criterion=None, device=None):
        """
        evaluate a single MIP instance by two algorithms: lb-baseline and lb-pred_k
        :param node_time_limit:
        :param total_time_limit:
        :param index_instance:
        :return:
        """
        instance = next(self.generator)
        MIP_model = instance.as_pyscipopt()
        MIP_model.setProbName(self.instance_type + '-' + str(index_instance))
        instance_name = MIP_model.getProbName()
        print('\n')
        print(instance_name)

        n_vars = MIP_model.getNVars()
        n_binvars = MIP_model.getNBinVars()
        print("N of variables: {}".format(n_vars))
        print("N of binary vars: {}".format(n_binvars))
        print("N of constraints: {}".format(MIP_model.getNConss()))

        valid, MIP_model, incumbent_solution = self.initialize_MIP(MIP_model)
        conti = 99
        # if self.incumbent_mode == 'rootsol' and self.instance_type == 'independentset':
        #     conti = 196

        if valid:
            if index_instance > 99 and index_instance > conti:
                gc.collect()
                observation, _, _, done, _ = self.env.reset(instance)
                del observation
                # print(observation)

                if self.incumbent_mode == 'firstsol':
                    action = {'limits/solutions': 1}
                elif self.incumbent_mode == 'rootsol':
                    action = {'limits/nodes': 1}  #
                sample_observation, _, _, done, _ = self.env.step(action)

                # print(sample_observation)
                graph = BipartiteNodeData(sample_observation.constraint_features,
                                          sample_observation.edge_features.indices,
                                          sample_observation.edge_features.values,
                                          sample_observation.variable_features)

                # We must tell pytorch geometric how many nodes there are, for indexing purposes
                graph.num_nodes = sample_observation.constraint_features.shape[0] + \
                                  sample_observation.variable_features.shape[
                                      0]

                # instance = Loader().load_instance('b1c1s1' + '.mps.gz')
                # MIP_model = instance

                # MIP_model.optimize()
                # print("Status:", MIP_model.getStatus())
                # print("best obj: ", MIP_model.getObjVal())
                # print("Solving time: ", MIP_model.getSolvingTime())

                initial_obj = MIP_model.getSolObjVal(incumbent_solution)
                print("Initial obj before LB: {}".format(initial_obj))

                binary_supports = binary_support(MIP_model, incumbent_solution)
                print('binary support: ', binary_supports)

                model_gnn = GNNPolicy()

                model_gnn.load_state_dict(torch.load(
                    self.saved_gnn_directory + 'trained_params_' + self.train_dataset + '_' + self.lbconstraint_mode + '_' + self.incumbent_mode + '.pth'))

                # model_gnn.load_state_dict(torch.load(
                #      'trained_params_' + self.instance_type + '.pth'))

                k_model = model_gnn(graph.constraint_features, graph.edge_index, graph.edge_attr,
                                    graph.variable_features)

                k_pred = k_model.item() * n_binvars
                print('GNN prediction: ', k_model.item())

                if self.is_symmetric == False:
                    k_pred = k_model.item() * binary_supports

                k_pred = np.ceil(k_pred)

                del k_model
                del graph
                del sample_observation
                del model_gnn

                # create a copy of MIP
                MIP_model.resetParams()
                # MIP_model_copy, MIP_copy_vars, success = MIP_model.createCopy(
                #     problemName='Baseline', origcopy=False)
                MIP_model_copy2, MIP_copy_vars2, success2 = MIP_model.createCopy(
                    problemName='GNN',
                    origcopy=False)
                MIP_model_copy3, MIP_copy_vars3, success3 = MIP_model.createCopy(
                    problemName='GNN+reset',
                    origcopy=False)

                print('MIP copies are created')

                # MIP_model_copy, sol_MIP_copy = copy_sol(MIP_model, MIP_model_copy, incumbent_solution,
                #                                         MIP_copy_vars)
                MIP_model_copy2, sol_MIP_copy2 = copy_sol(MIP_model, MIP_model_copy2, incumbent_solution,
                                                          MIP_copy_vars2)
                MIP_model_copy3, sol_MIP_copy3 = copy_sol(MIP_model, MIP_model_copy3, incumbent_solution,
                                                          MIP_copy_vars3)

                print('incumbent solution is copied to MIP copies')
                MIP_model.freeProb()
                del MIP_model
                del incumbent_solution

                # sol = MIP_model_copy.getBestSol()
                # initial_obj = MIP_model_copy.getSolObjVal(sol)
                # print("Initial obj before LB: {}".format(initial_obj))

                # # execute local branching baseline heuristic by Fischetti and Lodi
                # lb_model = LocalBranching(MIP_model=MIP_model_copy, MIP_sol_bar=sol_MIP_copy, k=self.k_baseline,
                #                           node_time_limit=node_time_limit,
                #                           total_time_limit=total_time_limit)
                # status, obj_best, elapsed_time, lb_bits, times, objs = lb_model.search_localbranch(is_symmeric=self.is_symmetric,
                #                                                              reset_k_at_2nditeration=False)
                # print("Instance:", MIP_model_copy.getProbName())
                # print("Status of LB: ", status)
                # print("Best obj of LB: ", obj_best)
                # print("Solving time: ", elapsed_time)
                # print('\n')
                #
                # MIP_model_copy.freeProb()
                # del sol_MIP_copy
                # del MIP_model_copy

                # sol = MIP_model_copy2.getBestSol()
                # initial_obj = MIP_model_copy2.getSolObjVal(sol)
                # print("Initial obj before LB: {}".format(initial_obj))

                # execute local branching with 1. first k predicted by GNN, 2. for 2nd iteration of lb, reset k to default value of baseline
                lb_model3 = LocalBranching(MIP_model=MIP_model_copy3, MIP_sol_bar=sol_MIP_copy3, k=k_pred,
                                           node_time_limit=node_time_limit,
                                           total_time_limit=total_time_limit)
                status, obj_best, elapsed_time, lb_bits_pred_reset, times_reset_imitation, objs_reset_imitation, loss_instance, accu_instance = lb_model3.mdp_localbranch(
                    is_symmetric=self.is_symmetric,
                    reset_k_at_2nditeration=reset_k_at_2nditeration,
                    policy=policy,
                    optimizer=None,
                    criterion=criterion,
                    device=device
                    )
                print("Instance:", MIP_model_copy3.getProbName())
                print("Status of LB: ", status)
                print("Best obj of LB: ", obj_best)
                print("Solving time: ", elapsed_time)
                print('\n')

                MIP_model_copy3.freeProb()
                del sol_MIP_copy3
                del MIP_model_copy3

                # execute local branching with 1. first k predicted by GNN; 2. for 2nd iteration of lb, continue lb algorithm with no further injection
                lb_model2 = LocalBranching(MIP_model=MIP_model_copy2, MIP_sol_bar=sol_MIP_copy2, k=k_pred,
                                           node_time_limit=node_time_limit,
                                           total_time_limit=total_time_limit)
                status, obj_best, elapsed_time, lb_bits_pred, times_reset_vanilla, objs_reset_vanilla, _, _ = lb_model2.mdp_localbranch(
                    is_symmetric=self.is_symmetric,
                    reset_k_at_2nditeration=True,
                    policy=None,
                    optimizer=None,
                    criterion=None,
                    device=None
                )

                print("Instance:", MIP_model_copy2.getProbName())
                print("Status of LB: ", status)
                print("Best obj of LB: ", obj_best)
                print("Solving time: ", elapsed_time)
                print('\n')

                MIP_model_copy2.freeProb()
                del sol_MIP_copy2
                del MIP_model_copy2

                data = [objs_reset_vanilla, times_reset_vanilla, objs_reset_imitation, times_reset_imitation]
                filename = f'{self.directory_lb_test}lb-test-{instance_name}.pkl'  # instance 100-199
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(data, f)

                del data
                del lb_model2
                del lb_model3

            index_instance += 1
        del instance
        return index_instance

    def evaluate_localbranching(self, test_instance_size='-small', total_time_limit=60, node_time_limit=30, reset_k_at_2nditeration=False, greedy=False):

        self.train_dataset = self.instance_type + self.instance_size
        self.evaluation_dataset = self.instance_type + test_instance_size

        self.generator = generator_switcher(self.evaluation_dataset)
        self.generator.seed(self.seed)

        self.k_baseline = 20

        self.is_symmetric = True
        if self.lbconstraint_mode == 'asymmetric':
            self.is_symmetric = False
            self.k_baseline = self.k_baseline / 2
        total_time_limit = total_time_limit
        node_time_limit = node_time_limit

        self.saved_gnn_directory = './result/saved_models/'

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/'
        self.directory_lb_test = directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        pathlib.Path(self.directory_lb_test).mkdir(parents=True, exist_ok=True)

        rl_policy = SimplePolicy(7, 4)

        self.saved_rlmodels_directory = self.saved_gnn_directory + 'rl_noimitation/'
        checkpoint = torch.load(
            self.saved_rlmodels_directory + 'checkpoint_noimitation_reward2_simplepolicy_rl4lb_reinforce_lr0.05_epsilon0.0.pth')
        rl_policy.load_state_dict(checkpoint['model_state_dict'])

        # rl_policy.load_state_dict(torch.load(
        #     self.saved_gnn_directory + 'trained_params_simplepolicy_rl4lb_reinforce_lr0.1_epsilon0.0_pre.pth'))

        rl_policy.eval()
        criterion = nn.CrossEntropyLoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        greedy = greedy
        rl_policy = rl_policy.to(device)
        agent = AgentReinforce(rl_policy, device, greedy, None, 0.0)

        index_instance = 0
        while index_instance < 200:
            index_instance = self.evaluate_lb_per_instance(node_time_limit=node_time_limit, total_time_limit=total_time_limit, index_instance=index_instance, reset_k_at_2nditeration=reset_k_at_2nditeration,
                                                           policy=agent, criterion=criterion, device=device
                                                           )

    def primal_integral(self, test_instance_size, total_time_limit=60, node_time_limit=30):

        directory = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/' + 'rl/reinforce/'
        directory_lb_test = directory + 'evaluation-reinforce4lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        if self.incumbent_mode == 'firstsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'rootsol' + '/' + 'rl/'
            directory_lb_test_2 = directory_2 + 'imitation4lb-from-' +  'rootsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'
        elif self.incumbent_mode == 'rootsol':
            directory_2 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + 'firstsol' + '/' + 'rl/'
            directory_lb_test_2 = directory_2 + 'imitation4lb-from-' + 'firstsol' + '-t_node' + str(node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        directory_3 = './result/generated_instances/' + self.instance_type + '/' + test_instance_size + '/' + self.lbconstraint_mode + '/' + self.incumbent_mode + '/'
        directory_lb_test_3 = directory_3 + 'lb-from-' + self.incumbent_mode + '-t_node' + str(
            node_time_limit) + 's' + '-t_total' + str(total_time_limit) + 's' + test_instance_size + '/'

        primal_int_baselines = []
        primal_int_reset_vanillas = []
        primal_in_reset_reinforces = []
        primal_gap_final_baselines = []
        primal_gap_final_reset_vanillas = []
        primal_gap_final_reset_reinforces = []
        steplines_baseline = []
        steplines_reset_vanillas = []
        steplines_reset_imitations = []

        for i in range(100,200):
            instance_name = self.instance_type + '-' + str(i)  # instance 100-199

            filename = f'{directory_lb_test}lb-test-{instance_name}.pkl'

            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            objs_reset_vanilla, times_reset_vanilla, objs_reset_reinforce, times_reset_reinforce = data  # objs contains objs of a single instance of a lb test

            filename_3 = f'{directory_lb_test_3}lb-test-{instance_name}.pkl'

            with gzip.open(filename_3, 'rb') as f:
                data = pickle.load(f)
            objs, times, objs_pred_2, times_pred_2, objs_pred_reset_2, times_pred_reset_2 = data  # objs contains objs of a single instance of a lb test


            # a = [objs_reset_vanilla.min(), objs_reset_reinforce.min(), objs_reset_vanilla_2.min(), objs_reset_imitation_2.min()]
            a = [objs_reset_vanilla.min(), objs_reset_reinforce.min(), objs.min()]
            obj_opt = np.amin(a)

            # compute primal gap for baseline localbranching run
            # if times[-1] < total_time_limit:
            times = np.append(times, total_time_limit)
            objs = np.append(objs, objs[-1])

            gamma_baseline = np.zeros(len(objs))
            for j in range(len(objs)):
                if objs[j] == 0 and obj_opt == 0:
                    gamma_baseline[j] = 0
                elif objs[j] * obj_opt < 0:
                    gamma_baseline[j] = 1
                else:
                    gamma_baseline[j] = np.abs(objs[j] - obj_opt) / np.maximum(np.abs(objs[j]), np.abs(obj_opt)) #

            # compute the primal gap of last objective
            primal_gap_final_baseline = np.abs(objs[-1] - obj_opt) / np.abs(obj_opt)
            primal_gap_final_baselines.append(primal_gap_final_baseline)

            # create step line
            stepline_baseline = interp1d(times, gamma_baseline, 'previous')
            steplines_baseline.append(stepline_baseline)

            # compute primal integral
            primal_int_baseline = 0
            for j in range(len(objs) - 1):
                primal_int_baseline += gamma_baseline[j] * (times[j + 1] - times[j])
            primal_int_baselines.append(primal_int_baseline)



            # lb-regression
            # if times_reset_vanilla[-1] < total_time_limit:
            times_reset_vanilla = np.append(times_reset_vanilla, total_time_limit)
            objs_reset_vanilla = np.append(objs_reset_vanilla, objs_reset_vanilla[-1])

            gamma_reset_vanilla = np.zeros(len(objs_reset_vanilla))
            for j in range(len(objs_reset_vanilla)):
                if objs_reset_vanilla[j] == 0 and obj_opt == 0:
                    gamma_reset_vanilla[j] = 0
                elif objs_reset_vanilla[j] * obj_opt < 0:
                    gamma_reset_vanilla[j] = 1
                else:
                    gamma_reset_vanilla[j] = np.abs(objs_reset_vanilla[j] - obj_opt) / np.maximum(np.abs(objs_reset_vanilla[j]), np.abs(obj_opt)) #

            primal_gap_final_vanilla = np.abs(objs_reset_vanilla[-1] - obj_opt) / np.abs(obj_opt)
            primal_gap_final_reset_vanillas.append(primal_gap_final_vanilla)

            stepline_reset_vanilla = interp1d(times_reset_vanilla, gamma_reset_vanilla, 'previous')
            steplines_reset_vanillas.append(stepline_reset_vanilla)

            # compute primal interal
            primal_int_reset_vanilla = 0
            for j in range(len(objs_reset_vanilla) - 1):
                primal_int_reset_vanilla += gamma_reset_vanilla[j] * (
                            times_reset_vanilla[j + 1] - times_reset_vanilla[j])
            primal_int_reset_vanillas.append(primal_int_reset_vanilla)

            #
            # t = np.linspace(start=0.0, stop=total_time_limit, num=1001)
            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of primal gap")
            # fig.subplots_adjust(top=0.5)
            # # ax.set_title(instance_name, loc='right')
            # ax.plot(t, stepline_baseline(t), label='lb baseline')
            # ax.plot(t, stepline_reset_vanilla(t), label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()

            # lb-regression-reinforce
            times_reset_reinforce = np.append(times_reset_reinforce, total_time_limit)
            objs_reset_reinforce = np.append(objs_reset_reinforce, objs_reset_reinforce[-1])

            gamma_reset_reinforce = np.zeros(len(objs_reset_reinforce))
            for j in range(len(objs_reset_reinforce)):
                if objs_reset_reinforce[j] == 0 and obj_opt == 0:
                    gamma_reset_reinforce[j] = 0
                elif objs_reset_reinforce[j] * obj_opt < 0:
                    gamma_reset_reinforce[j] = 1
                else:
                    gamma_reset_reinforce[j] = np.abs(objs_reset_reinforce[j] - obj_opt) / np.maximum(np.abs(objs_reset_reinforce[j]), np.abs(obj_opt)) #

            primal_gap_final_reinforce = np.abs(objs_reset_reinforce[-1] - obj_opt) / np.abs(obj_opt)
            primal_gap_final_reset_reinforces.append(primal_gap_final_reinforce)

            stepline_reset_imitation = interp1d(times_reset_reinforce, gamma_reset_reinforce, 'previous')
            steplines_reset_imitations.append(stepline_reset_imitation)

            # compute primal interal
            primal_int_reset_reinforce = 0
            for j in range(len(objs_reset_reinforce) - 1):
                primal_int_reset_reinforce += gamma_reset_reinforce[j] * (times_reset_reinforce[j + 1] - times_reset_reinforce[j])
            primal_in_reset_reinforces.append(primal_int_reset_reinforce)

            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of objective")
            # fig.subplots_adjust(top=0.5)
            # ax.set_title(instance_name, loc='right')
            # ax.plot(times, objs, label='lb baseline')
            # ax.plot(times_reset_vanilla, objs_reset_vanilla, label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()
            #
            # plt.close('all')
            # plt.clf()
            # fig, ax = plt.subplots(figsize=(8, 6.4))
            # fig.suptitle("Test Result: comparison of primal gap")
            # fig.subplots_adjust(top=0.5)
            # ax.set_title(instance_name, loc='right')
            # ax.plot(times, gamma_baseline, label='lb baseline')
            # ax.plot(times_reset_vanilla, gamma_reset_vanilla, label='lb with k predicted')
            # ax.set_xlabel('time /s')
            # ax.set_ylabel("objective")
            # ax.legend()
            # plt.show()


        primal_int_baselines = np.array(primal_int_baselines).reshape(-1)
        primal_int_reset_vanilla = np.array(primal_int_reset_vanillas).reshape(-1)
        primal_in_reset_imitation = np.array(primal_in_reset_reinforces).reshape(-1)

        primal_gap_final_baselines = np.array(primal_gap_final_baselines).reshape(-1)
        primal_gap_final_reset_vanilla = np.array(primal_gap_final_reset_vanillas).reshape(-1)
        primal_gap_final_reset_imitation = np.array(primal_gap_final_reset_reinforces).reshape(-1)

        # avarage primal integral over test dataset
        primal_int_base_ave = primal_int_baselines.sum() / len(primal_int_baselines)
        primal_int_reset_vanilla_ave = primal_int_reset_vanilla.sum() / len(primal_int_reset_vanilla)
        primal_int_reset_imitation_ave = primal_in_reset_imitation.sum() / len(primal_in_reset_imitation)

        primal_gap_final_baselines = primal_gap_final_baselines.sum() / len(primal_gap_final_baselines)
        primal_gap_final_reset_vanilla = primal_gap_final_reset_vanilla.sum() / len(primal_gap_final_reset_vanilla)
        primal_gap_final_reset_imitation = primal_gap_final_reset_imitation.sum() / len(primal_gap_final_reset_imitation)

        print(self.instance_type + self.instance_size)
        print(self.incumbent_mode + 'Solution')
        print('baseline primal integral: ', primal_int_base_ave)
        print('regression primal integral: ', primal_int_reset_vanilla_ave)
        print('rl primal integral: ', primal_int_reset_imitation_ave)
        print('\n')
        print('baseline primal gap: ',primal_gap_final_baselines)
        print('regression primal gap: ', primal_gap_final_reset_vanilla)
        print('rl primal gap: ', primal_gap_final_reset_imitation)

        t = np.linspace(start=0.0, stop=total_time_limit, num=1001)

        primalgaps_baseline = None
        for n, stepline_baseline in enumerate(steplines_baseline):
            primal_gap = stepline_baseline(t)
            if n==0:
                primalgaps_baseline = primal_gap
            else:
                primalgaps_baseline = np.vstack((primalgaps_baseline, primal_gap))
        primalgap_baseline_ave = np.average(primalgaps_baseline, axis=0)

        primalgaps_reset_vanilla = None
        for n, stepline_reset_vanilla in enumerate(steplines_reset_vanillas):
            primal_gap = stepline_reset_vanilla(t)
            if n == 0:
                primalgaps_reset_vanilla = primal_gap
            else:
                primalgaps_reset_vanilla = np.vstack((primalgaps_reset_vanilla, primal_gap))
        primalgap_reset_vanilla_ave = np.average(primalgaps_reset_vanilla, axis=0)

        primalgaps_reset_imitation = None
        for n, stepline_reset_imitation in enumerate(steplines_reset_imitations):
            primal_gap = stepline_reset_imitation(t)
            if n == 0:
                primalgaps_reset_imitation = primal_gap
            else:
                primalgaps_reset_imitation = np.vstack((primalgaps_reset_imitation, primal_gap))
        primalgap_reset_imitation_ave = np.average(primalgaps_reset_imitation, axis=0)

        plt.close('all')
        plt.clf()
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.suptitle("Normalized primal gap")
        # fig.subplots_adjust(top=0.5)
        ax.set_title(self.instance_type + '-' + self.incumbent_mode, loc='right')
        ax.plot(t, primalgap_baseline_ave, label='lb-baseline')
        ax.plot(t, primalgap_reset_vanilla_ave, label='lb-regression')
        ax.plot(t, primalgap_reset_imitation_ave,'--', label='lb-regression-rl')
        ax.set_xlabel('time /s')
        ax.set_ylabel("normalized primal gap")
        ax.legend()
        plt.show()













