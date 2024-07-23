from aiida.plugins import DataFactory
from aiida.orm import Str, Int
from aiida_submission_controller import BaseSubmissionController, FromGroupSubmissionController
from aiida_pyflame.codes.flame.flame_workchains import (
    GenSymCrysWorkChain,
    AverDistWorkChain,
    FLAMETrainWorkChain,
    FLAMEMinhocaoWorkChain,
    FLAMEMinhoppWorkChain,
    FLAMEDivCheckWorkChain,
    QBCWorkChain)

StructureData = DataFactory('structure')

class GenSymCrysSubmissionController(BaseSubmissionController):
    """ SubmissionController
    """
    def __init__(self,
            data_dict,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dict = data_dict

    def get_extra_unique_keys(self):
        """ Return a tuple of the keys of the unique extras that
            will be used to uniquely identify your workchains
        """
        return ('a_comp', 'a_n_a', 'atmpts')

    def get_all_extras_to_submit(self):
        """ Return a *set* of the values of all extras uniquely
            identifying all simulations that you want to submit.
            Each entry of the set must be a tuple, in same order
            as the keys returned by get_extra_unique_keys().
            Note: for each item, pass extra values as tuples
        """
        data_dict = self.data_dict
        all_extras = set()
        for a_comp in data_dict.keys():
            for a_n_a in data_dict[a_comp][0]:
                for atmpts in range(data_dict[a_comp][1]):
                    all_extras.add((a_comp, a_n_a, atmpts))
        return all_extras

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """ Return the inputs and the process class for the process
            to run, associated a given tuple of extras values.
            Param: extras_values: a tuple of values of the extras,
            in same order as the keys returned by get_extra_unique_keys().
        """
        inputs = {'composition' : Str(extras_values[0]), 'n_atom' : Int(extras_values[1])}
        return inputs, GenSymCrysWorkChain

class AverDistSubmissionController(BaseSubmissionController):
    """ SubmissionController
    """
    def __init__(self,
            nats,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.nats = nats

    def get_extra_unique_keys(self):
        """ Return a tuple of the keys of the unique extras
            that will be used to uniquely identify your workchains
        """
        return ('nat', )

    def get_all_extras_to_submit(self):
        """ Return a *set* of the values of all extras uniquely
            identifying all simulations that you want to submit.
            Each entry of the set must be a tuple, in same order as
            the keys returned by get_extra_unique_keys().
            Note: for each item, pass extra values as tuples
        """
        all_extras = set()
        for nat in self.nats:
            all_extras.add((nat, ))
        return all_extras

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """ Return the inputs and the process class for the process to run,
            associated a given tuple of extras values.
            Param: extras_values: a tuple of values of the extras,
            in same order as the keys returned by get_extra_unique_keys().
        """
        nat = Int(extras_values[0])
        inputs = {'nat' : nat}
        return inputs, AverDistWorkChain

class TrainSubmissionController(BaseSubmissionController):
    """ SubmissionController
    """
    def __init__(self,
            cycle_number,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.cycle_number = cycle_number

    def get_extra_unique_keys(self):
        """ Return a tuple of the keys of the unique extras
            that will be used to uniquely identify your workchains
        """
        return ('cycle_number', )

    def get_all_extras_to_submit(self):
        """ Return a *set* of the values of all extras uniquely
            identifying all simulations that you want to submit.
            Each entry of the set must be a tuple, in same order as
            the keys returned by get_extra_unique_keys().
            Note: for each item, pass extra values as tuples
        """
        all_extras = set()
        all_extras.add((self.cycle_number,))
        return all_extras

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """ Return the inputs and the process class for the process to run,
            associated a given tuple of extras values.
            Param: extras_values: a tuple of values of the extras,
            in same order as the keys returned by get_extra_unique_keys().
        """
        cycle_number = Str(extras_values[0])
        inputs = {'cycle_number': cycle_number}
        return inputs, FLAMETrainWorkChain

class MinimaHoppingSubmissionController(FromGroupSubmissionController):
    """ SubmissionController
    """
    def __init__(self,
            cycle_number,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.cycle_number = cycle_number

    def get_extra_unique_keys(self):
        """ Return a tuple of the keys of the unique extras that
            will be used to uniquely identify your workchains
        """
        return ['job', '_aiida_hash']

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """ Return the inputs and the process class for the process to run,
            associated a given tuple of extras values.
            Param: extras_values: a tuple of values of the extras,
            in same order as the keys returned by get_extra_unique_keys().
        """
        parent_node = self.get_parent_node_from_extras(extras_values)
        structure = parent_node
        if 'minhocao' in parent_node.label:
            process_class = FLAMEMinhocaoWorkChain
            inputs = {'cycle_number': Str(self.cycle_number), 'structure': structure}
        if 'minhopp' in parent_node.label:
            if 'cluster' in parent_node.label:
                bc = 'free'
            if 'bulk' in parent_node.label:
                bc = 'bulk'
            process_class = FLAMEMinhoppWorkChain
            inputs = {'cycle_number': Str(self.cycle_number), 'structure': structure, 'bc': Str(bc)}
        return inputs, process_class

class DivCheckSubmissionController(BaseSubmissionController):
    """ SubmissionController
    """
    def __init__(self,
            cycle_number,
            nats,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.cycle_number = cycle_number
        self.nats = nats

    def get_extra_unique_keys(self):
        """ Return a tuple of the keys of the unique extras
            that will be used to uniquely identify your workchains
        """
        return ('cycle-number', 'nat', 'bc')

    def get_all_extras_to_submit(self):
        """ Return a *set* of the values of all extras uniquely
            identifying all simulations that you want to submit.
            Each entry of the set must be a tuple, in same order as
            the keys returned by get_extra_unique_keys().
            Note: for each item, pass extra values as tuples
        """
        all_extras = set()
        for nat in self.nats['bulk']:
            all_extras.add((self.cycle_number, nat, 'bulk'))
        for nat in self.nats['cluster']:
            all_extras.add((self.cycle_number, nat, 'free'))
        return all_extras

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """ Return the inputs and the process class for the process to run,
            associated a given tuple of extras values.
            Param: extras_values: a tuple of values of the extras,
            in same order as the keys returned by get_extra_unique_keys().
        """
        cycle_number = Str(extras_values[0])
        nat = Int(extras_values[1])
        bc = Str(extras_values[2])
        inputs = {'cycle_number': cycle_number, 'nat': nat, 'bc': bc}
        return inputs, FLAMEDivCheckWorkChain

class QBCSubmissionController(BaseSubmissionController):
    """ SubmissionController
    """
    def __init__(self,
            cycle_number,
            nats,
            ref,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.cycle_number = cycle_number
        self.nats = nats
        self.ref = ref

    def get_extra_unique_keys(self):
        """ Return a tuple of the keys of the unique extras
            that will be used to uniquely identify your workchains
        """
        return ('cycle-number', 'nat', 'job_type')

    def get_all_extras_to_submit(self):
        """ Return a *set* of the values of all extras uniquely
            identifying all simulations that you want to submit.
            Each entry of the set must be a tuple, in same order as
            the keys returned by get_extra_unique_keys().
            Note: for each item, pass extra values as tuples
        """
        all_extras = set()
        if self.ref:
            all_extras.add((self.cycle_number, 0, 'ref'))
        else:
            for nat in self.nats['bulk']:
                all_extras.add((self.cycle_number, nat, 'bulk'))
            for nat in self.nats['cluster']:
                all_extras.add((self.cycle_number, nat, 'free'))
        return all_extras

    def get_inputs_and_processclass_from_extras(self, extras_values):
        """ Return the inputs and the process class for the process to run,
            associated a given tuple of extras values.
            Param: extras_values: a tuple of values of the extras,
            in same order as the keys returned by get_extra_unique_keys().
        """
        cycle_number = Str(extras_values[0])
        nat = Int(extras_values[1])
        job_type = Str(extras_values[2])
        inputs = {'cycle_number': cycle_number, 'nat': nat, 'job_type': job_type}
        return inputs, QBCWorkChain
