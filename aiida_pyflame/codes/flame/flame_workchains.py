import os
import json
import collections
from random import randint
from itertools import combinations_with_replacement
import yaml
from aiida.orm import Group, Dict, Str, Int, Code, SinglefileData
from aiida.plugins import CalculationFactory, DataFactory
from aiida.engine import WorkChain
from pymatgen.core.composition import Composition
from pymatgen.core.periodic_table import Element
from aiida_pyflame.codes.utils import get_element_list, get_allowed_n_atom_for_compositions
import aiida_pyflame.workflows.settings as settings

def dict_merge(dct, merge_dct):
    """ Taken from https://gist.github.com/angstwad/
    """
    for k in merge_dct.keys():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], collections.abc.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

def get_options(job_type, max_wallclock = False):
    job_script = settings.job_script
    if 'averdist' in job_type:
        resources = {
            'num_machines': 1,
            'num_mpiprocs_per_machine': job_script['QBC']['ntasks']}
        job_type = 'QBC'
    else:
        resources = {
            'num_machines': job_script[job_type]['nodes'],
            'num_mpiprocs_per_machine': job_script[job_type]['ntasks']}
    options = {'resources': resources}
    if job_script[job_type]['exclusive']:
        options.update({'custom_scheduler_commands' : '#SBATCH --exclusive'})
    if max_wallclock:
        options['max_wallclock_seconds'] = max_wallclock
    else:
        options['max_wallclock_seconds'] = job_script[job_type]['time']
    return options

class GenSymCrysWorkChain(WorkChain):
    """ FLAME calculation for generating crystal structures
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('composition', valid_type=Str)
        spec.input('n_atom', valid_type=Int)
        spec.outline(
            cls.initialize,
            cls.run_gensymcrys,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message="The calculation did not finish successfully")

    def initialize(self):
        """ Initialize
        """
        composition = self.inputs.composition.value
        with open(os.path.join(settings.FLAME_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.flame_in = yaml.safe_load(fhandle)
        self.ctx.elements = []
        self.ctx.n_elmnt = []
        for elmnt, nelmnt in Composition(composition).items():
            self.ctx.elements.append(str(elmnt))
            self.ctx.n_elmnt.append(int(nelmnt))

    def run_gensymcrys(self):
        """ Run
        """
        n_atom = self.inputs.n_atom.value
        # input parameters
        parameters = self.ctx.flame_in['gensymcrys']
        additional_parameters = self._get_additional_parameters(self.ctx.elements, self.ctx.n_elmnt, n_atom)
        dict_merge(parameters, additional_parameters)
        # builder
        builder = self._construct_builder(parameters)
        # submit
        future = self.submit(builder)
        self.to_context(**{'gensymcrys': future})

    def inspect_calculation(self):
        """ Inspect
        """
        if not self.ctx['gensymcrys'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    @staticmethod
    def _construct_builder(parameters):
        Workflow = CalculationFactory('flame')
        builder = Workflow.get_builder()
        builder.parameters = Dict(dict=parameters)
        builder.job_type_info = Dict(dict={'gensymcrys':{}})
        builder.code = Code.get_from_string(settings.configs['aiida_settings']['FLAME_code_string'])
        builder.settings = Dict(dict={
            'additional_retrieve_list': ['posout.yaml'],
            'retrieve_temporary_list':[]})
        builder.metadata['label'] = 'gensymcrys'
        builder.metadata.options = get_options('gensymcrys')
        builder.metadata.options.parser_name = 'pyflame_gensymcrys_parser'
        return builder

    @staticmethod
    def _get_additional_parameters(elements, n_elmnt, n_atom):
        vpas = []
        dimers = {}
        known_structures_group = Group.collection.get(label='known_structures')
        for a_node in known_structures_group.nodes:
            if 'vpas' in a_node.label:
                vpas = a_node.get_list()
            if 'dimers' in a_node.label:
                dimers = a_node.get_dict()
        pairs = {}
        for a_pair in combinations_with_replacement(elements,2):
            pairs[''.join(a_pair)] = dimers['-'.join(a_pair)] * 0.90

        additional_parameters = {}
        additional_parameters['main'] = {}
        additional_parameters['main']['types'] = ' '.join(elements)
        additional_parameters['main']['seed'] = randint(1,10**randint(1,6))
        additional_parameters['genconf'] = {}
        additional_parameters['genconf']['volperatom_bounds'] = [vpas[0], vpas[1]]
        additional_parameters['genconf']['nat_types_fu'] = n_elmnt
        additional_parameters['genconf']['list_fu'] = [int(n_atom/sum(n_elmnt))]
        additional_parameters['genconf']['nconf'] = 230
        additional_parameters['genconf']['rmin_pairs'] = pairs
        return additional_parameters

class AverDistWorkChain(WorkChain):
    """ Average distance calculations
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('nat', valid_type=Int)
        spec.outline(
            cls.initialize,
            cls.run_divcheck,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message="The calculation did not finish successfully")

    def initialize(self):
        """ Initialize
        """
        self.ctx.nat = self.inputs.nat.value
        self.ctx.element_list = get_element_list()
        with open(os.path.join(settings.FLAME_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.flame_in = yaml.safe_load(fhandle)

    def run_divcheck(self):
        """ Run
        """
        # input parameters
        parameters = self.ctx.flame_in['averdist']
        parameters['main']['types'] = ' '.join(self.ctx.element_list)
        # builder
        builder = self._construct_builder(parameters, self.ctx.nat)
        # submit
        future = self.submit(builder)
        self.to_context(**{'divcheck': future})

    def inspect_calculation(self):
        """ Inspect
        """
        if not self.ctx['divcheck'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    @staticmethod
    def _construct_builder(parameters, nat):
        Workflow = CalculationFactory('flame')
        builder = Workflow.get_builder()
        builder.parameters = Dict(dict=parameters)
        builder.job_type_info = Dict({'averdist': {'nat':nat}})
        builder.code = Code.get_from_string(settings.configs['aiida_settings']['FLAME_code_string'])
        builder.settings = Dict(dict={
            'additional_retrieve_list': ['nat.dat'],
            'retrieve_temporary_list':['distall']})
        builder.metadata['label'] = 'averdist'
        builder.metadata.options = get_options('averdist')
        builder.metadata.options.parser_name = 'pyflame_averdist_parser'
        return builder


class FLAMETrainWorkChain(WorkChain):
    """ FLAME calculation for model training
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('cycle_number', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_train,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message="The calculation did not finish successfully")

    def initialize(self):
        """ Initialize
        """
        self.ctx.cycle_number = self.inputs.cycle_number.value
        with open(os.path.join(settings.FLAME_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.flame_in = yaml.safe_load(fhandle)

    def run_train(self):
        """ Run
        """
        # input parameters
        parameters = self.ctx.flame_in['train']
        for train_n in ['1', '2', '3']:
            additional_parameters = self._get_additional_parameters(self.ctx.cycle_number)
            dict_merge(parameters, additional_parameters)
            # builder
            builder = self._construct_builder(parameters, self.ctx.cycle_number)
            # submit
            future = self.submit(builder)
            key = 'train_'+ train_n
            self.to_context(**{key: future})

    def inspect_calculation(self):
        """ Inspect
        """
        for train_n in ['1', '2', '3']:
            key = 'train_'+ train_n
            if not self.ctx[key].is_finished_ok:
                self.report('The calculation did not finish successfully')
                return self.exit_codes.ERROR_CALCULATION_FAILED

    @staticmethod
    def _construct_builder(parameters, cycle_number):
        Workflow = CalculationFactory('flame')
        builder = Workflow.get_builder()
        builder.parameters = Dict(dict=parameters)
        builder.job_type_info = Dict(dict={'train':{'cycle_number':cycle_number}})
        builder.code = Code.get_from_string(settings.configs['aiida_settings']['FLAME_code_string'])
        builder.settings = Dict(dict={
            'additional_retrieve_list': ['train_output.yaml', 'flame_in.yaml'],
            'retrieve_temporary_list':[]})
        builder.metadata['label'] = 'train_'+cycle_number
        builder.metadata.options = get_options('train')
        builder.metadata.options.parser_name = 'pyflame_train_parser'
        return builder

    @staticmethod
    def _get_additional_parameters(cycle_number):
        element_list = get_element_list()
        additional_parameters = {}
        additional_parameters['main'] = {}
        additional_parameters['main']['types'] = ' '.join(element_list)
        additional_parameters['main']['seed'] = randint(1,10**randint(1,6))
        additional_parameters['ann'] = {}
        c_no = int(cycle_number.split('-')[-1])
        additional_parameters['ann']['nstep_opt'] = settings.inputs['number_of_epoch'][c_no-1]
        return additional_parameters

class FLAMEMinhocaoWorkChain(WorkChain):
    """ FLAME calculation for minhocao
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('cycle_number', valid_type=Str)
        spec.input('structure', valid_type=DataFactory('structure'))
        spec.outline(
            cls.initialize,
            cls.run_minhocao,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message="The calculation did not finish successfully")

    def initialize(self):
        """ Initialize
        """
        self.ctx.cycle_number = self.inputs.cycle_number.value
        self.ctx.pymatgen_structure = self.inputs.structure.get_pymatgen()
        with open(os.path.join(settings.FLAME_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.flame_in = yaml.safe_load(fhandle)

    def run_minhocao(self):
        """ Run
        """
        # input parameters
        parameters = self.ctx.flame_in['minhocao']
        additional_parameters = self._get_additional_parameters(self.ctx.pymatgen_structure, self.ctx.cycle_number)
        dict_merge(parameters, additional_parameters)
        # builder
        builder = self._construct_builder(parameters, self.ctx.pymatgen_structure, self.ctx.cycle_number)
        # submit
        future = self.submit(builder)
        self.to_context(**{'minhocao': future})

    def inspect_calculation(self):
        """ Inspect
        """
        if not self.ctx['minhocao'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    @staticmethod
    def _construct_builder(parameters, pymatgen_structure, cycle_number):
        Workflow = CalculationFactory('flame')
        builder = Workflow.get_builder()
        builder.parameters = Dict(dict=parameters)
        builder.job_type_info = Dict(dict={'minhocao':{'cycle_number':cycle_number, 'structure':pymatgen_structure.as_dict()}})
        ann_pots = {}
        element_list = get_element_list()
        for elmnt in element_list:
            with open(os.path.join(settings.output_dir,cycle_number,'train',elmnt+'.ann.param.yaml'), 'rb') as fhandle:
                ann_pots[elmnt] = SinglefileData(file=fhandle)
        builder.file = ann_pots
        builder.code = Code.get_from_string(settings.configs['aiida_settings']['FLAME_code_string'])
        builder.settings = Dict(dict={
            'additional_retrieve_list': ['global.mon'],
            'retrieve_temporary_list':['poslow*.ascii', ('data_hop_*/posmd.*.ascii', '.', 2)]})
        builder.metadata['label'] = 'minhocao_'+cycle_number
        builder.metadata.options = get_options('minimahopping', int((parameters['main']['time_limit'] + 1) * 3600))
        builder.metadata.options.parser_name = 'pyflame_minhocao_parser'
        return builder

    @staticmethod
    def _get_additional_parameters(pymatgen_structure, cycle_number):
        element_list = get_element_list()
        c_no = int(cycle_number.split('-')[-1])
        allowed_n_atom = get_allowed_n_atom_for_compositions(settings.inputs['Chemical_formula'])
        min_n = min(allowed_n_atom)
        max_n = max(allowed_n_atom)
        additional_parameters = {}
        additional_parameters['main'] = {}
        additional_parameters['main']['types'] = ' '.join(element_list)
        additional_parameters['main']['seed'] = randint(1,10**randint(1,6))
        site_symbols = []
        for site in pymatgen_structure:
            site_symbols.append(site.specie.symbol)
        additional_parameters['main']['nat'] = len(pymatgen_structure.sites)
        dummy_typat = []
        for i in range(len(element_list)):
            dummy_typat.append('{}*{}'.format(site_symbols.count(element_list[i]),i+1))
        additional_parameters['main']['typat'] = ' '.join(dummy_typat)
        znucl = []
        for i in range(len(element_list)):
            znucl.append(Element(element_list[i]).Z)
        additional_parameters['main']['znucl'] = znucl
        amass = []
        for i in range(len(element_list)):
            eam = str(Element(element_list[i]).atomic_mass)
            amass.append(float(eam.split()[0]))
        additional_parameters['main']['amass'] = amass
        min_t = settings.inputs['minimahopping_time'][0]
        max_t = settings.inputs['minimahopping_time'][1]
        if max_n == min_n:
            minhocao_time = max_t
        else:
            minhocao_time = min_t + (max_t - min_t)/(max_n - min_n) * (len(pymatgen_structure.sites) - min_n)
        additional_parameters['main']['time_limit'] = minhocao_time - 1
        additional_parameters['minhopp'] = {}
        additional_parameters['minhopp']['nstep'] = settings.inputs['minhocao_steps'][c_no-1]
        return additional_parameters

class FLAMEMinhoppWorkChain(WorkChain):
    """ FLAME calculation for minhocao
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('cycle_number', valid_type=Str)
        spec.input('structure', valid_type=DataFactory('structure'))
        spec.input('bc', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_minhopp,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message="The calculation did not finish successfully")

    def initialize(self):
        """ Initialize
        """
        self.ctx.cycle_number = self.inputs.cycle_number.value
        self.ctx.bc = self.inputs.bc.value
        self.ctx.pymatgen_structure = self.inputs.structure.get_pymatgen()
        with open(os.path.join(settings.FLAME_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.flame_in = yaml.safe_load(fhandle)

    def run_minhopp(self):
        """ Run
        """
        # input parameters
        parameters = self.ctx.flame_in['minhopp']
        additional_parameters = self._get_additional_parameters(self.ctx.pymatgen_structure, self.ctx.cycle_number)
        dict_merge(parameters, additional_parameters)
        # builder
        builder = self._construct_builder(parameters, self.ctx.pymatgen_structure, self.ctx.cycle_number, self.ctx.bc)
        # submit
        future = self.submit(builder)
        self.to_context(**{'minhopp': future})

    def inspect_calculation(self):
        """ Inspect
        """
        if not self.ctx['minhopp'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    @staticmethod
    def _construct_builder(parameters, pymatgen_structure, cycle_number, bc):
        Workflow = CalculationFactory('flame')
        builder = Workflow.get_builder()
        builder.parameters = Dict(dict=parameters)
        builder.job_type_info = Dict(dict={'minhopp':{'cycle_number':cycle_number, 'structure':pymatgen_structure.as_dict(), 'bc':bc}})
        ann_pots = {}
        element_list = get_element_list()
        for elmnt in element_list:
            with open(os.path.join(settings.output_dir,cycle_number,'train',elmnt+'.ann.param.yaml'), 'rb') as fhandle:
                ann_pots[elmnt] = SinglefileData(file=fhandle)
        builder.file = ann_pots
        builder.code = Code.get_from_string(settings.configs['aiida_settings']['FLAME_code_string'])
        builder.settings = Dict(dict={
            'additional_retrieve_list': ['monminhopp/monitoring.000'],
            'retrieve_temporary_list':['poslow.yaml', 'traj_*_mde.bin']})
        if 'free' in bc:
            builder.metadata['label'] = 'minhopp_cluster_'+cycle_number
        else:
            builder.metadata['label'] = 'minhopp_bulk_'+cycle_number
        builder.metadata.options = get_options('minimahopping', int((parameters['main']['time_limit'] + 1) * 3600))
        builder.metadata.options.parser_name = 'pyflame_minhopp_parser'
        return builder

    @staticmethod
    def _get_additional_parameters(pymatgen_structure, cycle_number):
        element_list = get_element_list()
        c_no = int(cycle_number.split('-')[-1])
        allowed_n_atom = get_allowed_n_atom_for_compositions(settings.inputs['Chemical_formula'])
        min_n = min(allowed_n_atom)
        max_n = max(allowed_n_atom)
        additional_parameters = {}
        additional_parameters['main'] = {}
        additional_parameters['main']['types'] = ' '.join(element_list)
        additional_parameters['main']['seed'] = randint(1,10**randint(1,6))
        min_t = settings.inputs['minimahopping_time'][0]
        max_t = settings.inputs['minimahopping_time'][1]
        if max_n == min_n:
            minhocao_time = max_t
        else:
            minhocao_time = min_t + (max_t - min_t)/(max_n - min_n) * (len(pymatgen_structure.sites) - min_n)
        additional_parameters['main']['time_limit'] = minhocao_time - 1
        additional_parameters['minhopp'] = {}
        additional_parameters['minhopp']['nstep'] = settings.inputs['minhopp_steps'][c_no-1]
        return additional_parameters

class FLAMEDivCheckWorkChain(WorkChain):
    """ Diversity check calculations
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('nat', valid_type=Int)
        spec.input('cycle_number', valid_type=Str)
        spec.input('bc', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_divcheck,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message="The calculation did not finish successfully")

    def initialize(self):
        """ Initialize
        """
        self.ctx.nat = self.inputs.nat.value
        self.ctx.cycle_number = self.inputs.cycle_number.value
        self.ctx.bc = self.inputs.bc.value
        with open(os.path.join(settings.FLAME_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.flame_in = yaml.safe_load(fhandle)

    def run_divcheck(self):
        """ Run
        """
        # input parameters
        parameters = self.ctx.flame_in['divcheck']
        additional_parameters = self._get_additional_parameters(self.ctx.nat, self.ctx.bc)
        dict_merge(parameters, additional_parameters)
        # builder
        builder = self._construct_builder(parameters, self.ctx.nat, self.ctx.cycle_number, self.ctx.bc)
        # submit
        future = self.submit(builder)
        self.to_context(**{'divcheck': future})

    def inspect_calculation(self):
        """ Inspect
        """
        if not self.ctx['divcheck'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    @staticmethod
    def _construct_builder(parameters, nat, cycle_number, bc):
        Workflow = CalculationFactory('flame')
        builder = Workflow.get_builder()
        builder.parameters = Dict(dict=parameters)
        builder.job_type_info = Dict({'divcheck': {'cycle_number':cycle_number, 'nat':nat, 'bc':bc}})
        builder.code = Code.get_from_string(settings.configs['aiida_settings']['FLAME_code_string'])
        builder.settings = Dict(dict={
            'additional_retrieve_list': ['data.dat'],
            'retrieve_temporary_list':['posout.yaml']})
        if 'free' in bc:
            builder.metadata['label'] = 'divcheck_cluster'
        if 'bulk' in bc:
            builder.metadata['label'] = 'divcheck_bulk'
        builder.metadata.options = get_options('QBC')
        builder.metadata.options.parser_name = 'pyflame_divcheck_parser'
        return builder

    @staticmethod
    def _get_additional_parameters(nat, bc):
        element_list = get_element_list()
        additional_parameters = {}
        additional_parameters['main'] = {}
        additional_parameters['main']['types'] = ' '.join(element_list)
        with open(os.path.join(settings.Flame_dir,'aver_dist.json'), 'r', encoding='utf8') as fhandle:
            aver_dist_dict = json.loads(fhandle.read())
        dtol = float(aver_dist_dict[str(nat)]) * float(settings.inputs['dtol_prefactor'])
        if 'free' in bc:
            dtol = dtol * float(settings.inputs['prefactor_cluster'])
        additional_parameters['ann'] = {}
        additional_parameters['ann']['dtol'] = dtol
        return additional_parameters

class QBCWorkChain(WorkChain):
    """ single point calculations
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('nat', valid_type=Int)
        spec.input('cycle_number', valid_type=Str)
        spec.input('job_type', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_SP,
            cls.inspect_calculation)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message="The calculation did not finish successfully")

    def initialize(self):
        """ Initialize
        """
        self.ctx.nat = self.inputs.nat.value
        self.ctx.cycle_number = self.inputs.cycle_number.value
        self.ctx.job_type = self.inputs.job_type.value
        with open(os.path.join(settings.FLAME_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.flame_in = yaml.safe_load(fhandle)

    def run_SP(self):
        """ Run
        """
        # input parameters
        parameters = self.ctx.flame_in['single_point']

        for SP_n in ['1', '2', '3']:
            additional_parameters = self._get_additional_parameters()
            dict_merge(parameters, additional_parameters)
            # builder
            builder = self._construct_builder(SP_n, parameters, self.ctx.nat, self.ctx.cycle_number, self.ctx.job_type)
            # submit
            future = self.submit(builder)
            key = 'single_point_'+ SP_n
            self.to_context(**{key: future})

    def inspect_calculation(self):
        """ Inspect
        """
        for SP_n in ['1', '2', '3']:
            key = 'single_point_'+ SP_n
            if not self.ctx[key].is_finished_ok:
                self.report('The calculation did not finish successfully')
                return self.exit_codes.ERROR_CALCULATION_FAILED

    @staticmethod
    def _construct_builder(SP_n, parameters, nat, cycle_number, job_type):
        Workflow = CalculationFactory('flame')
        builder = Workflow.get_builder()
        builder.parameters = Dict(dict=parameters)
        builder.job_type_info = Dict({'qbc': {'cycle_number':cycle_number, 'nat':nat, 'job_type':job_type, 'SP_n':SP_n}})
        builder.code = Code.get_from_string(settings.configs['aiida_settings']['FLAME_code_string'])
        builder.settings = Dict(dict={
            'retrieve_temporary_list':['posout.yaml']})
        if 'ref' in job_type:
            builder.metadata['label'] = 'SP_ref'
        if 'free' in job_type:
            builder.metadata['label'] = 'SP_cluster'
        if 'bulk' in job_type:
            builder.metadata['label'] = 'SP_bulk'
        builder.metadata.options = get_options('QBC')
        builder.metadata.options.parser_name = 'pyflame_SP_parser'
        return builder

    @staticmethod
    def _get_additional_parameters():
        element_list = get_element_list()
        additional_parameters = {}
        additional_parameters['main'] = {}
        additional_parameters['main']['types'] = ' '.join(element_list)
        return additional_parameters
