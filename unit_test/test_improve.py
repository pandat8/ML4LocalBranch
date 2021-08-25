import pyscipopt
from pyscipopt import Model
import ecole
import numpy
import matplotlib.pyplot as plt
import pathlib
from localbranching import addLBConstraint
from geco.mips.loading.miplib import Loader
from improvelp import improvelp

modes = ['improve-supportbinvars', 'improve-binvars']
mode = modes[1]

directory = './result/miplib2017/' + mode + '/'
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

# directory = './result/miplib2017/improve/'
data = numpy.load('./result/miplib2017/miplib2017_binary39.npz')
miplib2017_binary39 = data['miplib2017_binary39']

for p in range(27, len(miplib2017_binary39)):
    instance = Loader().load_instance(miplib2017_binary39[p] + '.mps.gz')
    MIP_model = instance
    print(MIP_model.getProbName())
    improvelp(MIP_model, directory, mode)
