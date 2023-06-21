import os
import yaml
import copy
import collections
from aiida.plugins import WorkflowFactory, DataFactory
from aiida.engine import WorkChain
from aiida.orm import Int, Str, Dict, Code, SinglefileData, load_group
import aiida_pyflame.workflows.settings as settings

results_step3_group = load_group(label='results_step3')
results_singlepoint_group = load_group(label='results_singlepoint')
StructureData = DataFactory('structure')

def dict_merge(dct, merge_dct):
    """Taken from https://gist.github.com/angstwad/
    """
    for k in merge_dct.keys():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], collections.abc.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

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

def get_kinds_section(structure, basis_pseudo, GTHorSIRIUS, magnetization_tags=None):
    """ Write the &KIND section
    Taken from aiida-commonworkflow
    Modfied for SIRIUS/GTH
    """
    kinds = []
    with open(os.path.join(settings.CP2K_input_files_path, basis_pseudo), 'rb') as fhandle:
        atom_data = yaml.safe_load(fhandle)
    ase_structure = structure.get_ase()
    symbol_tag = {
    (symbol, str(tag)) for symbol, tag in zip(ase_structure.get_chemical_symbols(), ase_structure.get_tags())
    }
    for symbol, tag in symbol_tag:

        if 'GTH' in GTHorSIRIUS:
            new_atom = {
                '_': symbol if tag == '0' else symbol + tag,
                'BASIS_SET': atom_data['basis_set'][symbol],
                'POTENTIAL': atom_data['pseudopotential'][symbol],
            }
        if 'SIRIUS' in GTHorSIRIUS:
            new_atom = {
                '_': symbol if tag == '0' else symbol + tag,
                'POTENTIAL': 'UPF ' + atom_data['UPF_pseudopotential'][symbol],
            }
        if magnetization_tags:
            new_atom['MAGNETIZATION'] = magnetization_tags[tag]
        kinds.append(new_atom)
    return {'FORCE_EVAL': {'SUBSYS': {'KIND': kinds, 'CELL': {}}}}

def get_file_section(structure, GTHorSIRIUS):
    """ Potential files for SIRIUS/GTH
    """
    files_dict =  {}
    if 'GTH' in GTHorSIRIUS:
        with open(os.path.join(settings.CP2K_input_files_path,'GTH_POTENTIALS'), 'rb') as handler:
            potential = SinglefileData(file=handler)
        files_dict['potential'] = potential
        with open(os.path.join(settings.CP2K_input_files_path,'GTH_BASIS_SETS'), 'rb') as handle:
            basis_gth = SinglefileData(file=handle)
        files_dict['basis_gth'] = basis_gth
        with open(os.path.join(settings.CP2K_input_files_path,'BASIS_MOLOPT'), 'rb') as handle:
            basis_molopt = SinglefileData(file=handle)
        files_dict['basis_molopt'] = basis_molopt
        with open(os.path.join(settings.CP2K_input_files_path,'BASIS_MOLOPT_UCL'), 'rb') as handle:
            basis_molopt_ucl = SinglefileData(file=handle)
        files_dict['basis_molopt_ucl'] = basis_molopt_ucl
    else:
        with open(os.path.join(settings.CP2K_input_files_path,'pseudopotentials.yml'), 'rb') as fhandle:
            atom_data = yaml.safe_load(fhandle)

            ase_structure = structure.get_ase()
            symbol_tag = {
            (symbol, str(tag)) for symbol, tag in zip(ase_structure.get_chemical_symbols(), ase_structure.get_tags())
            }
            for symbol, tag in symbol_tag:
                with open(os.path.join(settings.CP2K_input_files_path,'pseudopotentials',atom_data['UPF_pseudopotential'][symbol]), 'rb') as fhandle:
                    files_dict[symbol] = SinglefileData(file=fhandle)
    return files_dict

def get_kpoints(kpoints_distance, structure):
    """ kpoints for SIRIUS/GTH
    """
    KpointsData = DataFactory('array.kpoints')
    if kpoints_distance:
        kpoints_mesh = KpointsData()
        kpoints_mesh.set_cell_from_structure(structure)
        kpoints_mesh.set_kpoints_mesh_from_density(distance=kpoints_distance)
        return kpoints_mesh
    return None

def construct_builder(structure, parameters, job_type, GTHorSIRIUS):
    Workflow = WorkflowFactory('cp2k.base')
    builder = Workflow.get_builder()
    builder.cp2k.structure = structure
    kpoints_distance = parameters.pop('kpoints_distance', 10)
    kpoints = get_kpoints(kpoints_distance, structure)
    mesh, _ = kpoints.get_kpoints_mesh()
    if 'GTH' in GTHorSIRIUS:
        if mesh != [1, 1, 1]:
            builder.cp2k.kpoints = kpoints
        basis_pseudo = parameters.pop('basis_pseudo')
    if 'SIRIUS' in GTHorSIRIUS:
        if mesh != [1, 1, 1]:
            parameters['FORCE_EVAL']['PW_DFT']['PARAMETERS']['NGRIDK'] = '{} {} {}'.format(mesh[0], mesh[1], mesh[2])
        cell = parameters['FORCE_EVAL']['SUBSYS']['CELL']
        for i, keys in enumerate(cell.keys()):
            cell[keys] = '{} {:<15} {:<15} {:<15}'.format(cell[keys],
                    round(structure.cell[i][0],14),
                    round(structure.cell[i][1],14),
                    round(structure.cell[i][2],14))
        basis_pseudo = 'pseudopotentials.yml'
    dict_merge(parameters, get_kinds_section(structure, basis_pseudo, GTHorSIRIUS))
    if job_type in ['opt1c', 'opt2_cluster', 'single_point_cluster']:
        periodic = None
    else:
        periodic = 'XYZ'
    parameters['FORCE_EVAL']['SUBSYS']['CELL']['PERIODIC'] = periodic

    builder.cp2k.parameters = Dict(dict=parameters)
    builder.cp2k.file = get_file_section(structure, GTHorSIRIUS)
    builder.cp2k.code = Code.get_from_string(settings.configs['aiida_settings']['DFT_code_string'])
    builder.cp2k.settings = Dict(dict={
        'additional_retrieve_list': ['aiida.inp',
                                     'aiida-pos-1.xyz',
                                     'aiida-frc-1.xyz',
                                     'aiida-1.cell',
                                     'aiida-s_p_forces-1_0.xyz',
                                     'aiida.coords.xyz']})
    builder.cp2k.metadata.options = get_options()
    if job_type in ['opt1vc', 'opt1c']:
        builder.cp2k.metadata.options['parser_name'] = 'cp2k_base_parser'
    else:
        builder.cp2k.metadata.options['parser_name'] = 'cp2k_efs_parser'
    builder.handler_overrides = Dict(dict={'restart_incomplete_calculation': True})
    builder.max_iterations = Int(2)
    builder.cp2k.metadata['label'] = job_type
    return builder

def get_scaled_structure(scale_factor, structure):
    pymatgen_structure = structure.get_pymatgen()
    pymatgen_structure.scale_lattice(pymatgen_structure.volume*scale_factor)
    return StructureData(pymatgen=pymatgen_structure)

class RefGeOptWorkChain(WorkChain):
    """ CP2K calculations
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('GTHorSIRIUS', valid_type=Str)
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
        if 'GTH' in self.inputs.GTHorSIRIUS.value:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_GTH.yml')
        else:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_SIRIUS.yml')
        with open(protocol, 'r', encoding='utf8') as fhandle:
            self.ctx.protocol = yaml.safe_load(fhandle)

    def run_opt1vc(self):
        structure = self.inputs.structure
        GTHorSIRIUS = self.inputs.GTHorSIRIUS.value
        # input parameters
        parameters_opt1vc = self.ctx.protocol['opt1vc']
        parameters_opt1vc['GLOBAL']['RUN_TYPE'] = 'CELL_OPT'
        parameters_opt1vc['### JOB_TYPE'] = 'opt1vc'
        # builder
        builder = construct_builder(structure, parameters_opt1vc, 'opt1vc', GTHorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt1vc': future})

    def inspect_calculation_1(self):
        if not self.ctx['opt1vc'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_bulk(self):
        structure = self.ctx['opt1vc'].outputs['output_structure']
        GTHorSIRIUS = self.inputs.GTHorSIRIUS.value
        # parameters
        parameters_bulk = self.ctx.protocol['bulk']
        parameters_bulk['GLOBAL']['RUN_TYPE'] = 'CELL_OPT'
        parameters_bulk['### JOB_TYPE'] = 'bulk'
        parameters_bulk['FORCE_EVAL'].pop('PRINT')
        # builder
        builder = construct_builder(structure, parameters_bulk, 'bulk', GTHorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'bulk': future})

    def inspect_calculation_2(self):
        if not self.ctx['bulk'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['bulk'])

class Scheme1GeOptWorkChain(WorkChain):
    """ CP2K calculations
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('GTHorSIRIUS', valid_type=Str)
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
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        if 'GTH' in self.inputs.GTHorSIRIUS.value:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_GTH.yml')
        else:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_SIRIUS.yml')
        with open(protocol, 'r', encoding='utf8') as fhandle:
            self.ctx.protocol = yaml.safe_load(fhandle)

    def run_opt1vc(self):
        structure = self.inputs.structure
        GTHorSIRIUS = self.inputs.GTHorSIRIUS.value
        # parameters
        parameters_opt1vc = self.ctx.protocol['opt1vc']
        parameters_opt1vc['GLOBAL']['RUN_TYPE'] = 'CELL_OPT'
        parameters_opt1vc['### JOB_TYPE'] = 'opt1vc'
        # builder
        builder = construct_builder(structure, parameters_opt1vc, 'opt1vc', GTHorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt1vc': future})

    def inspect_calculation_1(self):
        if not self.ctx['opt1vc'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_opt2(self):
        structure = self.ctx['opt1vc'].outputs['output_structure']
        GTHorSIRIUS = self.inputs.GTHorSIRIUS.value
        # parameters
        parameters_opt2 = self.ctx.protocol['opt2']
        parameters_opt2['GLOBAL']['RUN_TYPE'] = 'GEO_OPT'
        parameters_opt2['### JOB_TYPE'] = 'opt2_bulk'
        # builder
        builder = construct_builder(structure, parameters_opt2, 'opt2_bulk', GTHorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt2_bulk': future})

    def inspect_calculation_2(self):
        if not self.ctx['opt2_bulk'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['opt2_bulk'])

    def run_scaled_1(self):
        structure = self.ctx['opt2_bulk'].outputs['output_structure']
        GTHorSIRIUS = self.inputs.GTHorSIRIUS.value
        scaled_structure = get_scaled_structure(0.90, structure)
        # parameters
        parameters_optscaled = copy.deepcopy(self.ctx.protocol['scaled_bulk'])
        parameters_optscaled['GLOBAL']['RUN_TYPE'] = 'GEO_OPT'
        parameters_optscaled['### JOB_TYPE'] = 'scaled_bulk_1'
        # builder
        builder = construct_builder(scaled_structure, parameters_optscaled, 'scaled_bulk', GTHorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'scaled_bulk_1': future})

    def inspect_calculation_3(self):
        if not self.ctx['scaled_bulk_1'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['scaled_bulk_1'])

    def run_scaled_2(self):
        structure = self.ctx['opt2_bulk'].outputs['output_structure']
        GTHorSIRIUS = self.inputs.GTHorSIRIUS.value
        scaled_structure = get_scaled_structure(0.90, structure)
        # parameters
        parameters_optscaled = copy.deepcopy(self.ctx.protocol['scaled_bulk'])
        parameters_optscaled['GLOBAL']['RUN_TYPE'] = 'GEO_OPT'
        parameters_optscaled['### JOB_TYPE'] = 'scaled_bulk_2'
        # builder
        builder = construct_builder(scaled_structure, parameters_optscaled, 'scaled_bulk', GTHorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'scaled_bulk_2': future})

    def inspect_calculation_4(self):
        if not self.ctx['scaled_bulk_2'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['scaled_bulk_2'])

class Scheme2GeOptWorkChain(WorkChain):
    """ CP2K calculations
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('GTHorSIRIUS', valid_type=Str)
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
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        if 'GTH' in self.inputs.GTHorSIRIUS.value:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_GTH.yml')
        else:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_SIRIUS.yml')
        with open(protocol, 'r', encoding='utf8') as fhandle:
            self.ctx.protocol = yaml.safe_load(fhandle)

    def run_opt1vc(self):
        structure = self.inputs.structure
        GTHorSIRIUS = self.inputs.GTHorSIRIUS.value
        # parameters
        parameters_opt1vc = self.ctx.protocol['opt1vc']
        parameters_opt1vc['GLOBAL']['RUN_TYPE'] = 'CELL_OPT'
        parameters_opt1vc['### JOB_TYPE'] = 'opt1vc'
        # builder
        builder = construct_builder(structure, parameters_opt1vc, 'opt1vc', GTHorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt1vc': future})

    def inspect_calculation_1(self):
        if not self.ctx['opt1vc'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_single_point(self):
        structure = self.ctx['opt1vc'].outputs['output_structure']
        GTHorSIRIUS = self.inputs.GTHorSIRIUS.value
        # parameters
        parameters_s_p = self.ctx.protocol['bulk']
        parameters_s_p['GLOBAL']['RUN_TYPE'] = 'ENERGY_FORCE'
        parameters_s_p['### JOB_TYPE'] = 'single_point_bulk'
        parameters_s_p.pop('MOTION')
        # builder
        builder = construct_builder(structure, parameters_s_p, 'single_point_bulk', GTHorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'single_point_bulk': future})

    def inspect_calculation_2(self):
        if not self.ctx['single_point_bulk'].is_finished_ok:
            self.report('the calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['single_point_bulk'])

    def run_scaled_1(self):
        structure = self.ctx['opt1vc'].outputs['output_structure']
        GTHorSIRIUS = self.inputs.GTHorSIRIUS.value
        scaled_structure = get_scaled_structure(0.90, structure)
        # parameters
        parameters_optscaled = copy.deepcopy(self.ctx.protocol['scaled_bulk'])
        parameters_optscaled['GLOBAL']['RUN_TYPE'] = 'GEO_OPT'
        parameters_optscaled['### JOB_TYPE'] = 'scaled_bulk_1'
        # builder
        builder = construct_builder(scaled_structure, parameters_optscaled, 'scaled_bulk', GTHorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'scaled_bulk_1': future})

    def inspect_calculation_3(self):
        if not self.ctx['scaled_bulk_1'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['scaled_bulk_1'])

    def run_scaled_2(self):
        structure = self.ctx['opt1vc'].outputs['output_structure']
        GTHorSIRIUS = self.inputs.GTHorSIRIUS.value
        scaled_structure = get_scaled_structure(0.95, structure)
        # parameters
        parameters_optscaled = copy.deepcopy(self.ctx.protocol['scaled_bulk'])
        parameters_optscaled['GLOBAL']['RUN_TYPE'] = 'GEO_OPT'
        parameters_optscaled['### JOB_TYPE'] = 'scaled_bulk_2'
        # builder
        builder = construct_builder(scaled_structure, parameters_optscaled, 'scaled_bulk', GTHorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'scaled_bulk_2': future})

    def inspect_calculation_4(self):
        if not self.ctx['scaled_bulk_2'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['scaled_bulk_2'])

class ClusterGeOptWorkChain(WorkChain):
    """ CP2K calculations
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('GTHorSIRIUS', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_opt1c,
            cls.inspect_calculation_1,
            cls.run_opt2c,
            cls.inspect_calculation_2)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        if 'GTH' in self.inputs.GTHorSIRIUS.value:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_GTH.yml')
        else:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_SIRIUS.yml')
        with open(protocol, 'r', encoding='utf8') as fhandle:
            self.ctx.protocol = yaml.safe_load(fhandle)

    def run_opt1c(self):
        structure = self.inputs.structure
        GTHorSIRIUS = self.inputs.GTHorSIRIUS.value
        # input parameters
        parameters_opt1c = self.ctx.protocol['opt1']
        parameters_opt1c['GLOBAL']['RUN_TYPE'] = 'GEO_OPT'
        parameters_opt1c['### JOB_TYPE'] = 'opt1c'
        parameters_opt1c['kpoints_distance'] = 10
        # builder
        builder = construct_builder(structure, parameters_opt1c, 'opt1c', GTHorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt1c': future})

    def inspect_calculation_1(self):
        if not self.ctx['opt1c'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED

    def run_opt2c(self):
        structure = self.ctx['opt1c'].outputs['output_structure']
        GTHorSIRIUS = self.inputs.GTHorSIRIUS.value
        # input parameters
        parameters_opt2c = self.ctx.protocol['opt2']
        parameters_opt2c['GLOBAL']['RUN_TYPE'] = 'GEO_OPT'
        parameters_opt2c['### JOB_TYPE'] = 'opt2_cluster'
        parameters_opt2c['kpoints_distance'] = 10
        # builder
        builder = construct_builder(structure, parameters_opt2c, 'opt2_cluster', GTHorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'opt2_cluster': future})

    def inspect_calculation_2(self):
        if not self.ctx['opt2_cluster'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_step3_group.add_nodes(self.ctx['opt2_cluster'])

class SinglePointtWorkChain(WorkChain):
    """ CP2K calculations
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input('structure', valid_type=StructureData)
        spec.input('GTHorSIRIUS', valid_type=Str)
        spec.input('bc', valid_type=Str)
        spec.outline(
            cls.initialize,
            cls.run_single_point,
            cls.inspect_calculation_1)
        spec.exit_code(
            300, 'ERROR_CALCULATION_FAILED',
            message='The calculation did not finish successfully')

    def initialize(self):
        if 'GTH' in self.inputs.GTHorSIRIUS.value:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_GTH.yml')
        else:
            protocol = os.path.join(settings.CP2K_input_files_path,'protocol_SIRIUS.yml')
        with open(protocol, 'r', encoding='utf8') as fhandle:
            self.ctx.protocol = yaml.safe_load(fhandle)

    def run_single_point(self):
        structure = self.inputs.structure
        GTHorSIRIUS = self.inputs.GTHorSIRIUS.value
        # parameters
        parameters_s_p = self.ctx.protocol['bulk']
        parameters_s_p['GLOBAL']['RUN_TYPE'] = 'ENERGY_FORCE'
        parameters_s_p['### JOB_TYPE'] = 'single_point_bulk'
        parameters_s_p.pop('MOTION')

        bc = self.inputs.bc.value
        if 'free' in bc:
            job_type = 'single_point_cluster'
            parameters_s_p['kpoints_distance'] = 10
        else:
            job_type = 'single_point_bulk'
        # builder
        builder = construct_builder(structure, parameters_s_p, job_type, GTHorSIRIUS)
        # submit
        future = self.submit(builder)
        self.to_context(**{'single_point': future})

    def inspect_calculation_1(self):
        if not self.ctx['single_point'].is_finished_ok:
            self.report('The calculation did not finish successfully')
            return self.exit_codes.ERROR_CALCULATION_FAILED
        results_singlepoint_group.add_nodes(self.ctx['single_point'])
