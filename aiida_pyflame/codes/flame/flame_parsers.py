import os
import re
import yaml
from random import sample
from aiida.orm import Dict
from aiida.parsers import Parser
from aiida.common import exceptions
from aiida_pyflame.codes.flame.flame_functions.io_yaml import read_yaml, atoms2dict
from aiida_pyflame.codes.flame.flame_functions.ascii import ascii_read
from aiida_pyflame.codes.flame.flame_functions.io_bin import bin_read

class GenSymCrysParser(Parser):
    """ AiiDA parser class for the output of GenSymCrys
    """
    def parse(self, **kwargs):
        """Parse
        """
        try:
            retrieved_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            with retrieved_folder.open('posout.yaml', 'r') as fhandle:
                status = self._read_stdout(fhandle)
        except (OSError, IOError):
            return self.exit_codes.ERROR_OUTPUT_PARSE
        if not status:
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE
        return None

    @staticmethod
    def _read_stdout(stdout):
        """ Parse stdout
        """
        if len(list(yaml.safe_load_all(stdout))) == 0:
            return False
        return True

class AverDistParser(Parser):
    """ AiiDA parser class
    """
    def parse(self, **kwargs):
        """ Parse
        """
        aver_dist = None
        try:
            retrieved_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            retrieved_temporary_folder = kwargs['retrieved_temporary_folder']
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            with open(os.path.join(retrieved_temporary_folder,'distall'), 'r', encoding='utf8') as fhandle:
                lines = fhandle.read().splitlines()
        except (OSError, IOError):
            return self.exit_codes.ERROR_OUTPUT_PARSE
        aver_dist = self._aver_dist(lines)
        if not aver_dist:
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE
        with retrieved_folder.open('nat.dat', 'r') as fhandle:
            nat = fhandle.readline().strip()
        result_dict = {str(nat): aver_dist}
        self.out('output_parameters', Dict(dict=result_dict))
        return None

    @staticmethod
    def _aver_dist(lines):
        """ Calculate average distance
        """
        dist_list = []
        for a_line in lines:
            dist_list.append(float(re.split('\s+', a_line)[4]))
        if len(dist_list) == 0:
            return False
        return sum(dist_list)/len(dist_list)

class TrainParser(Parser):
    """ AiiDA parser class for training output
    """
    def parse(self, **kwargs):
        """Parse
        """
        try:
            retrieved_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            with retrieved_folder.open('flame_in.yaml', 'r') as fhandle:
                flame_in = yaml.safe_load(fhandle)
            n_step = flame_in['ann']['nstep_opt']
            with retrieved_folder.open('train_output.yaml', 'r') as fhandle:
                status = self._read_stdout(fhandle, n_step)
        except (OSError, IOError):
            return self.exit_codes.ERROR_OUTPUT_PARSE
        if not status:
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE
        return None

    @staticmethod
    def _read_stdout(stdout, n_step):
        """ Parse stdout
        """
        content = yaml.safe_load(stdout)
        if len(list(content)) == 0:
            return False
        if not content['training iterations']:
            return False
        if len(content['training iterations']) < int(n_step)+1:
            return False
        return True

class MinhocaoParser(Parser):
    """ AiiDA parser class for minhocao output
    """
    def parse(self, **kwargs):
        """ Parse
        """
        poslows = []
        posmds = []
        try:
            retrieved_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            retrieved_temporary_folder = kwargs['retrieved_temporary_folder']
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            with retrieved_folder.open('global.mon', 'r') as fhandle:
                status = self._read_stdout(fhandle)
        except (OSError, IOError):
            return self.exit_codes.ERROR_OUTPUT_PARSE
        if not status:
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE
        for a_file in os.listdir(retrieved_temporary_folder):
            if 'poslow' in a_file:
                try:
                    atoms = ascii_read(os.path.join(retrieved_temporary_folder,a_file))
                    poslows.append(atoms2dict(atoms))
                except:
                    pass
        for a_dir in os.listdir(retrieved_temporary_folder):
            posmd_files = []
            if 'data_hop_' in a_dir:
                for a_file in os.listdir(os.path.join(retrieved_temporary_folder, a_dir)):
                    posmd_files.append(os.path.join(retrieved_temporary_folder, a_dir, a_file))
                if len(posmd_files) > 2:
                    posmd_paths = sample(list(posmd_files), 2)
                    for a_posmd_path in posmd_paths:
                        atoms = ascii_read(a_posmd_path)
                        posmds.append(atoms2dict(atoms))
        result_dict = {'poslows': poslows, 'posmds': posmds}
        self.out("output_parameters", Dict(dict=result_dict))
        return None

    @staticmethod
    def _read_stdout(stdout):
        """ Parse stdout
        """
        if len(stdout.read().splitlines()) == 0:
            return False
        return True

class MinhoppParser(Parser):
    """ AiiDA parser class for minhopp output
    """
    def parse(self, **kwargs):
        """ Parse
        """
        bohr2ang=0.52917721
        poslows = []
        trajs = []
        try:
            retrieved_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            retrieved_temporary_folder = kwargs['retrieved_temporary_folder']
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            with retrieved_folder.open('monitoring.000', 'r') as fhandle:
                status = self._read_stdout(fhandle)
        except (OSError, IOError):
            return self.exit_codes.ERROR_OUTPUT_PARSE
        if not status:
            return self.exit_codes.ERROR_OUTPUT_INCOMPLETE
        try:
            atoms_yaml_all = read_yaml(os.path.join(retrieved_temporary_folder, 'poslow.yaml'))
            for atoms in atoms_yaml_all:
                poslows.append(atoms2dict(atoms))
        except (OSError, IOError):
            return self.exit_codes.ERROR_OUTPUT_PARSE
        for a_file in os.listdir(retrieved_temporary_folder):
            if 'traj' in a_file:
                atoms_bin_all = bin_read(os.path.join(retrieved_temporary_folder, a_file))
                if len(atoms_bin_all) > 2:
                    atoms_bin_selected = sample(atoms_bin_all, 2)
                    for atoms in atoms_bin_selected:
                        atoms.cellvec[0][0]=round(atoms.cellvec[0][0]*bohr2ang, 10)
                        atoms.cellvec[0][1]=round(atoms.cellvec[0][1]*bohr2ang, 10)
                        atoms.cellvec[0][2]=round(atoms.cellvec[0][2]*bohr2ang, 10)
                        atoms.cellvec[1][0]=round(atoms.cellvec[1][0]*bohr2ang, 10)
                        atoms.cellvec[1][1]=round(atoms.cellvec[1][1]*bohr2ang, 10)
                        atoms.cellvec[1][2]=round(atoms.cellvec[1][2]*bohr2ang, 10)
                        atoms.cellvec[2][0]=round(atoms.cellvec[2][0]*bohr2ang, 10)
                        atoms.cellvec[2][1]=round(atoms.cellvec[2][1]*bohr2ang, 10)
                        atoms.cellvec[2][2]=round(atoms.cellvec[2][2]*bohr2ang, 10)
                        for iat in range(atoms.nat):
                            atoms.rat[iat][0]=atoms.rat[iat][0]*bohr2ang
                            atoms.rat[iat][1]=atoms.rat[iat][1]*bohr2ang
                            atoms.rat[iat][2]=atoms.rat[iat][2]*bohr2ang
                        trajs.append(atoms2dict(atoms))
        result_dict = {'poslows': poslows, 'trajs': trajs}
        self.out("output_parameters", Dict(dict=result_dict))
        return None

    @staticmethod
    def _read_stdout(stdout):
        """ Parse stdout
        """
        if len(stdout.read().splitlines()) == 0:
            return False
        return True

class DivCheckParser(Parser):
    """ AiiDA parser class
    """
    def parse(self, **kwargs):
        """ Parse
        """
        posouts = []
        try:
            retrieved_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            retrieved_temporary_folder = kwargs['retrieved_temporary_folder']
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            with open(os.path.join(retrieved_temporary_folder,'posout.yaml'), 'r', encoding='utf8') as fhandle:
                confs = list(yaml.safe_load_all(fhandle))
        except (OSError, IOError):
            return self.exit_codes.ERROR_OUTPUT_PARSE
        try:
            with retrieved_folder.open('data.dat' , 'r') as fhandle:
                lines = fhandle.readlines()
            nat = lines[0].strip()
            nposin = lines[1]
        except (OSError, IOError):
            return self.exit_codes.ERROR_OUTPUT_PARSE
        for a_conf in confs:
            if a_conf['conf']['epot'] > 0:
                posouts.append(a_conf)
        nposout = len(posouts)
        result_dict = {'nat': int(nat), 'nposin': int(nposin), 'nposout': nposout, 'confs': posouts}
        self.out('output_parameters', Dict(dict=result_dict))
        return None

class SPParser(Parser):
    """ AiiDA parser class
    """
    def parse(self, **kwargs):
        """ Parse
        """
        posouts = []
        try:
            retrieved_folder = self.retrieved
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            retrieved_temporary_folder = kwargs['retrieved_temporary_folder']
        except exceptions.NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER
        try:
            with open(os.path.join(retrieved_temporary_folder,'posout.yaml'), 'r', encoding='utf8') as fhandle:
                confs = list(yaml.safe_load_all(fhandle))
        except (OSError, IOError):
            return self.exit_codes.ERROR_OUTPUT_PARSE
        result_dict = {'confs': confs}
        self.out('output_parameters', Dict(dict=result_dict))
        return None
