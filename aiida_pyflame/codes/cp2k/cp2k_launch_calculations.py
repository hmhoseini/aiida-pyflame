from aiida.orm import Str
from aiida_submission_controller import FromGroupSubmissionController
from aiida_pyflame.codes.cp2k.cp2k_workchains import (
    RefGeOptWorkChain,
    Scheme1GeOptWorkChain,
    Scheme2GeOptWorkChain,
    ClusterGeOptWorkChain,
    SinglePointtWorkChain)

class CP2KSubmissionController(FromGroupSubmissionController):
    """A SubmissionController
    """
    def __init__(self,
              GTHorSIRIUS,
              *args,
              **kwargs):
        super().__init__(*args, **kwargs)
        self.GTHorSIRIUS = GTHorSIRIUS

    def get_extra_unique_keys(self):
        """Return a tuple of the keys of the unique extras that
           will be used to uniquely identify your workchains
        """
        return ('job', )

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """Return the inputs and the process class for the process to run,
           associated a given tuple of extras values.
           Param: extras_values: a tuple of values of the extras,
           in same order as the keys returned by get_extra_unique_keys().
        """
        parent_node = self.get_parent_node_from_extras(extras_values)
        structure = parent_node
        inputs = {'structure': structure, 'GTHorSIRIUS': Str(self.GTHorSIRIUS)}
        if 'reference' in parent_node.label:
            process_class = RefGeOptWorkChain
        if 'scheme1' in parent_node.label:
            process_class = Scheme1GeOptWorkChain
        if 'scheme2' in parent_node.label:
            process_class = Scheme2GeOptWorkChain
        if 'cluster' in parent_node.label:
            process_class = ClusterGeOptWorkChain
        return inputs, process_class

class CP2KSPSubmissionController(FromGroupSubmissionController):
    """A SubmissionController
    """
    def __init__(self,
              GTHorSIRIUS,
              *args,
              **kwargs):
        super().__init__(*args, **kwargs)
        self._process_class = SinglePointtWorkChain
        self.GTHorSIRIUS = GTHorSIRIUS

    def get_extra_unique_keys(self):
        """Return a tuple of the keys of the unique extras that
           will be used to uniquely identify your workchains
        """
        return ('job', )

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """Return the inputs and the process class for the process to run,
           associated a given tuple of extras values.
           Param: extras_values: a tuple of values of the extras,
           in same order as the keys returned by get_extra_unique_keys().
        """
        parent_node = self.get_parent_node_from_extras(extras_values)
        structure = parent_node
        if 'cluster' in parent_node.label:
            bc = 'free'
        else:
            bc = 'bulk'
        inputs = {'structure': structure, 'bc': Str(bc), 'GTHorSIRIUS': Str(self.GTHorSIRIUS)}
        return inputs, self._process_class
