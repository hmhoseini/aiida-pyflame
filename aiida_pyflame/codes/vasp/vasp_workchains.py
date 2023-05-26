import os
import yaml
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.engine import WorkChain
from aiida.orm import Int, Str, Code, Dict, Bool, load_group
import aiida_pyflame.workflows.settings as settings

results_step3_group = load_group(label='results_step3')
results_singlepoint_group = load_group(label='results_singlepoint')
StructureData = DataFactory('structure')

def get_options():
    job_script = settings.job_script
    resources = {
        'num_machines': job_script['geopt']['nodes'],
        'num_mpiprocs_per_machine': job_script['geopt']['ntasks']}
    if job_script['geopt']['ncpu']:
        resources['num_cores_per_mpiproc'] = job_script['geopt']['ncpu']
    options = {'resources': resources,
               'max_wallclock_seconds': job_script['geopt']['time']}
    return options

def construct_builder(structure, protocol, potential_mapping):
    Workflow = WorkflowFactory('vasp.vasp')
    builder = Workflow.get_builder()
    builder.structure = structure
    builder.parameters = Dict(dict={'incar':protocol['incar']})
    builder.potential_family = Str(settings.configs['aiida_settings']['VASP_potential_family'])
    builder.potential_mapping = Dict(dict=potential_mapping['potential_mapping'])
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    if protocol['name'] in ['opt1c', 'opt2_cluster', 'single_point_cluster']:
        kpoints.set_kpoints_mesh_from_density(5)
    else:
        kpoints.set_kpoints_mesh_from_density(protocol['kpoint_distance'])
    builder.kpoints = kpoints
    builder.code = Code.get_from_string(settings.configs['aiida_settings']['DFT_code_string'])
    if protocol['name'] in ['opt1vc', 'opt1c']:
        parser_settings = {'add_structure': True}
    else:
        parser_settings = {'add_structure': True,
                           'add_trajectory': True,
                           'add_energies': True,
                           }
    builder.settings = Dict(dict={'parser_settings': parser_settings})
#    builder.settings = Dict(dict={'CHECK_IONIC_CONVERGENCE': False})
    builder.options = Dict(dict=get_options())
    builder.clean_workdir = Bool(False)
    builder.max_iterations = Int(2)
    builder.verbose = Bool(True)
    builder.metadata['label'] = protocol['name']
    return builder

def get_scaled_structure(scale_factor, structure):
    pymatgen_structure = structure.get_pymatgen()
    pymatgen_structure.scale_lattice(pymatgen_structure.volume*scale_factor)
    return StructureData(pymatgen=pymatgen_structure)

class RefGeOptWorkChain(WorkChain):
    """ VASP calculations
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=(DataFactory('structure')))
        spec.outline(
            cls.initialize,
            cls.run_opt1vc,
            cls.inspect_calculation_1,
            cls.run_bulk,
            cls.inspect_calculation_2)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        with open(os.path.join(settings.VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.vasp_protocol = yaml.safe_load(fhandle)
        with open(os.path.join(settings.VASP_input_files_path,'potential_mapping.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.potential_mapping = yaml.safe_load(fhandle)

    def run_opt1vc(self):
        structure = self.inputs.structure
        protocol = self.ctx.vasp_protocol['opt1vc']
        # builder
        builder = construct_builder(structure, protocol, self.ctx.potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt1vc': future})

    def inspect_calculation_1(self):
        if not self.ctx['opt1vc'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_bulk(self):
        structure = self.ctx['opt1vc'].outputs['structure']
        protocol = self.ctx.vasp_protocol['bulk']
        # builder
        builder = construct_builder(structure, protocol, self.ctx.potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'bulk': future})

    def inspect_calculation_2(self):
        if not self.ctx['bulk'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['bulk'].called[0])

class Scheme1GeOptWorkChain(WorkChain):
    """ VASP calculations
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.outline(
            cls.initialize,
            cls.run_opt1vc,
            cls.inspect_calculation_1,
            cls.run_opt2,
            cls.inspect_calculation_2,
            cls.run_scaled_1,
            cls.inspect_calculation_3,
            cls.run_scaled_2,
            cls.inspect_calculation_4)

    def initialize(self):
        with open(os.path.join(settings.VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.vasp_protocol = yaml.safe_load(fhandle)
        with open(os.path.join(settings.VASP_input_files_path,'potential_mapping.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.potential_mapping = yaml.safe_load(fhandle)

    def run_opt1vc(self):
        structure = self.inputs.structure
        protocol = self.ctx.vasp_protocol['opt1vc']
        # builder
        builder = construct_builder(structure, protocol, self.ctx.potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt1vc': future})

    def inspect_calculation_1(self):
        if not self.ctx['opt1vc'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_opt2(self):
        structure = self.ctx['opt1vc'].outputs['structure']
        protocol = self.ctx.vasp_protocol['opt2']
        protocol['name'] = 'opt2_bulk'
        # builder
        builder = construct_builder(structure, protocol, self.ctx.potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt2_bulk': future})

    def inspect_calculation_2(self):
        if not self.ctx['opt2_bulk'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['opt2_bulk'].called[0])

    def run_scaled_1(self):
        structure = self.ctx['opt2_bulk'].outputs['structure']
        scaled_structure = get_scaled_structure(0.90, structure)
        protocol = self.ctx.vasp_protocol['opt2']
        protocol['name'] = 'scaled_bulk'
        # builder
        builder = construct_builder(scaled_structure, protocol, self.ctx.potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'scaled_bulk_1': future})

    def inspect_calculation_3(self):
        if not self.ctx['scaled_bulk_1'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['scaled_bulk_1'].called[0])

    def run_scaled_2(self):
        structure = self.ctx['opt2_bulk'].outputs['structure']
        scaled_structure = get_scaled_structure(0.95, structure)
        protocol = self.ctx.vasp_protocol['opt2']
        protocol['name'] = 'scaled_bulk'
        # builder
        builder = construct_builder(scaled_structure, protocol, self.ctx.potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'scaled_bulk_2': future})

    def inspect_calculation_4(self):
        if not self.ctx['scaled_bulk_2'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['scaled_bulk_2'].called[0])

class Scheme2GeOptWorkChain(WorkChain):
    """ VASP calculations
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.outline(
            cls.initialize,
            cls.run_opt1vc,
            cls.inspect_calculation_1,
            cls.run_single_point,
            cls.inspect_calculation_2,
            cls.run_scaled_1,
            cls.inspect_calculation_3,
            cls.run_scaled_2,
            cls.inspect_calculation_4)

    def initialize(self):
        with open(os.path.join(settings.VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.vasp_protocol = yaml.safe_load(fhandle)
        with open(os.path.join(settings.VASP_input_files_path,'potential_mapping.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.potential_mapping = yaml.safe_load(fhandle)

    def run_opt1vc(self):
        structure = self.inputs.structure
        protocol = self.ctx.vasp_protocol['opt1vc']
        # builder
        builder = construct_builder(structure, protocol, self.ctx.potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt1vc': future})

    def inspect_calculation_1(self):
        if not self.ctx['opt1vc'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_single_point(self):
        structure = self.ctx['opt1vc'].outputs['structure']
        protocol = self.ctx.vasp_protocol['single_point']
        protocol['name'] = 'single_point_bulk'
        # builder
        builder = construct_builder(structure, protocol, self.ctx.potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'single_point_bulk': future})

    def inspect_calculation_2(self):
        if not self.ctx['single_point_bulk'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['single_point_bulk'].called[0])

    def run_scaled_1(self):
        structure = self.ctx['single_point_bulk'].outputs['structure']
        scaled_structure = get_scaled_structure(0.90, structure)
        protocol = self.ctx.vasp_protocol['opt2']
        protocol['name'] = 'scaled_bulk'
        # builder
        builder = construct_builder(scaled_structure, protocol, self.ctx.potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'scaled_bulk_1': future})

    def inspect_calculation_3(self):
        if not self.ctx['scaled_bulk_1'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['scaled_bulk_1'].called[0])

    def run_scaled_2(self):
        structure = self.ctx['single_point_bulk'].outputs['structure']
        scaled_structure = get_scaled_structure(0.95, structure)
        protocol = self.ctx.vasp_protocol['opt2']
        protocol['name'] = 'scaled_bulk'
        # builder
        builder = construct_builder(scaled_structure, protocol, self.ctx.potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'scaled_bulk_2': future})

    def inspect_calculation_4(self):
        if not self.ctx['scaled_bulk_2'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['scaled_bulk_2'].called[0])

class ClusterGeOptWorkChain(WorkChain):
    """ VASP calculations
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.outline(
            cls.initialize,
            cls.run_opt1c,
            cls.inspect_calculation_1,
            cls.run_opt2c,
            cls.inspect_calculation_2)

    def initialize(self):
        with open(os.path.join(settings.VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.vasp_protocol = yaml.safe_load(fhandle)
        with open(os.path.join(settings.VASP_input_files_path,'potential_mapping.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.potential_mapping = yaml.safe_load(fhandle)

    def run_opt1c(self):
        structure = self.inputs.structure
        protocol = self.ctx.vasp_protocol['opt1']
        protocol['name'] = 'opt1c'
        # builder
        builder = construct_builder(structure, protocol, self.ctx.potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt1c': future})

    def inspect_calculation_1(self):
        if not self.ctx['opt1c'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_opt2c(self):
        structure = self.ctx['opt1c'].outputs['structure']
        protocol = self.ctx.vasp_protocol['opt2']
        protocol['name'] = 'opt2_cluster'
        # builder
        builder = construct_builder(structure, protocol, self.ctx.potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt2_cluster': future})

    def inspect_calculation_2(self):
        if not self.ctx['opt2_cluster'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['opt2_cluster'].called[0])

class SinglePointtWorkChain(WorkChain):
    """ VASP calculations
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('bc', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_single_point,
            cls.inspect_calculation_1)

    def initialize(self):
        with open(os.path.join(settings.VASP_input_files_path,'protocol.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.vasp_protocol = yaml.safe_load(fhandle)
        with open(os.path.join(settings.VASP_input_files_path,'potential_mapping.yaml'), 'r', encoding='utf8') as fhandle:
            self.ctx.potential_mapping = yaml.safe_load(fhandle)

    def run_single_point(self):
        structure = self.inputs.structure
        bc = self.inputs.bc.value
        protocol = self.ctx.vasp_protocol['single_point']
        if 'free' in bc:
            protocol['name'] = 'single_point_cluster'
        else:
            protocol['name'] = 'single_point_bulk'
        # builder
        builder = construct_builder(structure, protocol, self.ctx.potential_mapping)
        # submit
        future = self.submit(builder)
        self.to_context(**{'single_point': future})

    def inspect_calculation_1(self):
        if not self.ctx['single_point'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_singlepoint_group.add_nodes(self.ctx['single_point'].called[0])
