import yaml
from aiida.engine import CalcJob
from aiida.orm import Dict, SinglefileData
from aiida.common import CalcInfo, CodeInfo

def write_extra_files(job_type_info, folder):
    """ Write extra files for FLAME calculation
    """
    provenance_exclude_list = []
    if 'gensymcrys' in job_type_info.keys():
        pass
    elif 'averdist' in job_type_info.keys():
        from aiida_pyflame.codes.flame.averdist import write_averdist_files
        provenance_exclude_list = write_averdist_files(folder, job_type_info['averdist']['nat'])
    elif 'train' in job_type_info.keys():
        from aiida_pyflame.codes.flame.train import write_train_files
        provenance_exclude_list = write_train_files(folder, job_type_info['train']['cycle_number'])
    elif 'minhocao' in job_type_info.keys():
        from aiida_pyflame.codes.flame.minimahopping import write_minhocao_files
        provenance_exclude_list = write_minhocao_files(folder, job_type_info['minhocao']['cycle_number'], job_type_info['minhocao']['structure'])
    elif 'minhopp' in job_type_info.keys():
        from aiida_pyflame.codes.flame.minimahopping import write_minhopp_files
        provenance_exclude_list = write_minhopp_files(folder, job_type_info['minhopp']['structure'], job_type_info['minhopp']['bc'])
    elif 'divcheck' in job_type_info.keys():
        from aiida_pyflame.codes.flame.divcheck import write_divcheck_files
        provenance_exclude_list = write_divcheck_files(folder, job_type_info['divcheck']['cycle_number'], job_type_info['divcheck']['nat'], job_type_info['divcheck']['bc'])
    elif 'qbc' in job_type_info.keys():
        from aiida_pyflame.codes.flame.qbc import write_SP_files
        provenance_exclude_list = write_SP_files(folder, job_type_info['qbc']['cycle_number'], job_type_info['qbc']['nat'], job_type_info['qbc']['job_type'], job_type_info['qbc']['SP_n'])
    return provenance_exclude_list

class FlameCalculation(CalcJob):
    """ A subclass of JobCalculation, to prepare input for FLAME calculations
    """
    _INPUT_FILE = "flame_in.yaml"

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("parameters", valid_type=Dict)
        spec.input("job_type_info", valid_type=Dict)
        spec.input_namespace("file", valid_type=(SinglefileData), required=False, dynamic=True)
        spec.input("settings", valid_type=Dict, required=False)
        spec.input("metadata.options.withmpi", valid_type=bool, default=True)

        spec.output("output_parameters", valid_type=Dict, required=False)

        spec.exit_code(
            200,
            "ERROR_NO_RETRIEVED_FOLDER",
            message="The retrieved folder data node can not be accessed"
        )
        spec.exit_code(
            302, "ERROR_OUTPUT_PARSE",
            message="The output file can not be parsed."
        )
        spec.exit_code(
            303, "ERROR_OUTPUT_INCOMPLETE",
            message="The output file is incomplete."
        )

    def prepare_for_submission(self, folder):
        """ Create input files
        """
        inp = self.inputs.parameters.get_dict()
        job_type_info = self.inputs.job_type_info.get_dict()

        with folder.open(self._INPUT_FILE, 'w', encoding='utf-8') as fhandle:
            yaml.dump(inp, fhandle, default_flow_style=False)

        provenance_exclude_list = write_extra_files(job_type_info, folder)

        settings = self.inputs.settings.get_dict() if "settings" in self.inputs else {}
        # Code info.
        codeinfo = CodeInfo()
        codeinfo.cmdline_params = []
        codeinfo.withmpi = self.metadata.options.withmpi
        codeinfo.code_uuid = self.inputs.code.uuid
        # Calc info.
        calcinfo = CalcInfo()
        calcinfo.uuid = self.uuid
        calcinfo.cmdline_params = codeinfo.cmdline_params
        calcinfo.retrieve_list = []
        calcinfo.retrieve_list += settings.pop("additional_retrieve_list", [])
        calcinfo.retrieve_temporary_list = settings.pop("retrieve_temporary_list", [])
        calcinfo.codes_info = [codeinfo]
        if "file" in self.inputs:
            calcinfo.local_copy_list = []
            for _, obj in self.inputs.file.items():
                if isinstance(obj, SinglefileData):
                    calcinfo.local_copy_list.append((obj.uuid, obj.filename, obj.filename))
        calcinfo.provenance_exclude_list = provenance_exclude_list
        return calcinfo
