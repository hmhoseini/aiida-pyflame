from aiida_pyflame.workflows.settings import restart

def main():
    stpnmbr = restart['re-start_from_step']
    if stpnmbr == 1:
        from aiida_pyflame.workflows.step1 import step_1
        if step_1():
            stpnmbr = 2
    if stpnmbr == 2:
        from aiida_pyflame.workflows.step2 import step_2
        if step_2():
            stpnmbr = 3
    if stpnmbr == 3:
        from aiida_pyflame.workflows.step3 import step_3
        if step_3():
            stpnmbr = 4
    if stpnmbr == 4:
        from aiida_pyflame.workflows.step4 import step_4
        step_4()
    if stpnmbr == -1:
        from aiida_pyflame.workflows.rerun import rerun
        rerun()
