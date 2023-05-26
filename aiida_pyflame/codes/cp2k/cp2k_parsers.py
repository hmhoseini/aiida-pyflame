from aiida_cp2k.parsers import Cp2kBaseParser
from aiida.engine import ExitCode
from aiida.common import exceptions
from aiida.orm import Dict
from aiida.plugins import DataFactory
from pymatgen.core.structure import Structure
from aiida_pyflame.codes.cp2k.parsers import (
    parse_cp2k_output_simple,
    read_coordinates,
    read_positions,
    read_forces,
    read_s_p_forces,
    read_cell_parameters,
    read_lattice_parameters)

class Cp2kEFSParser(Cp2kBaseParser):
    """AiiDA parser class for the output of CP2K
    Modified for SIRIUS
    """
    def parse(self, **kwargs):
        try:
            _ = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        exit_code = self._parse_stdout()
        if exit_code is not None:
            return exit_code
        return ExitCode(0)
    
    def _parse_stdout(self):
        exit_code, output_string = self._read_stdout()
        if exit_code:
            return exit_code
        # check the standard output for errors
        exit_code = self._check_stdout_for_errors(output_string)
        if exit_code:
            return exit_code
        result_dict = parse_cp2k_output_simple(output_string)
        exit_code = self._parse_efs(result_dict)
        if exit_code:
            return exit_code            
        return None

    def _parse_efs(self, result_dict):
        if result_dict["run_type"] in ["GEO_OPT", "CELL_OPT"] and\
           'aiida-pos-1.xyz' in self.retrieved.list_object_names() and\
           'aiida-frc-1.xyz' in self.retrieved.list_object_names() and\
           'aiida-1.cell' in self.retrieved.list_object_names():
            positions = read_positions(self.retrieved.get_object_content('aiida-pos-1.xyz'))
            forces = read_forces(self.retrieved.get_object_content('aiida-frc-1.xyz'))
            cells = read_cell_parameters(self.retrieved.get_object_content('aiida-1.cell'))
            symbols, _ = read_coordinates(self.retrieved.get_object_content('aiida.coords.xyz'))
        elif result_dict["run_type"] in ["ENERGY_FORCE"] and\
             'aiida-s_p_forces-1_0.xyz' in self.retrieved.list_object_names():
            symbols, positions = read_coordinates(self.retrieved.get_object_content('aiida.coords.xyz'))
            forces = read_s_p_forces(self.retrieved.get_object_content('aiida-s_p_forces-1_0.xyz'))
            if result_dict["SIRIUS"]:
                cells = [result_dict["lattice_vectors"]]
            else:
                cells = [read_lattice_parameters(self.retrieved.get_object_content('aiida.inp'))]
        else:
            return self.exit_codes.ERROR_OUTPUT_MISSING

        StructureData = DataFactory("structure")
        result_dict['motion_step_info'].update({'symbols': symbols, 'positions': positions, 'forces': forces, 'cells': cells})
        output_structure = StructureData(pymatgen=Structure(cells[-1], symbols, positions[-1], coords_are_cartesian=True))

        self.out("output_parameters", Dict(dict=result_dict))
        self.out("output_structure", output_structure)
        return None
