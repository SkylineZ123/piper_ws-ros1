"""Import and numerical checks only; never start nodes or open robot devices."""

import importlib
import sys
import warnings
from pathlib import Path


def check_data():
    report = {'python': sys.executable, 'checks': {}, 'errors': [], 'warnings': []}
    if sys.version_info < (3, 12):
        report['errors'].append('Data collection requires Python >= 3.12')
    modules = (
        'numpy',
        'scipy.spatial.transform',
        'yaml',
        'rospy',
        'rospkg',
        'cv2',
        'av',
        'matplotlib',
        'pyarrow',
        'pandas',
        'lerobot.datasets.lerobot_dataset',
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        for name in modules:
            try:
                loaded = importlib.import_module(name)
                report['checks'][name] = {
                    'path': loaded.__file__,
                    'version': getattr(loaded, '__version__', None),
                }
            except Exception as error:
                report['errors'].append(name + ': ' + str(error))
        report['warnings'] = list(dict.fromkeys(str(w.message) for w in caught))
    try:
        import av

        av.codec.Codec('libx264', 'w')
        report['checks']['h264_encoder'] = 'libx264 available'
    except Exception as error:
        report['errors'].append('h264_encoder: ' + str(error))
    try:
        import cv2

        report['checks']['opencv_gui_build'] = next(
            line.strip()
            for line in cv2.getBuildInformation().splitlines()
            if line.strip().startswith('GUI:')
        )
    except Exception as error:
        report['warnings'].append('GUI information: ' + str(error))
    report['ok'] = not report['errors']
    return report


def check_control():
    report = {
        'python': sys.executable,
        'python_version': sys.version.split()[0],
        'checks': {},
        'errors': [],
    }

    def check(name, function):
        try:
            report['checks'][name] = function()
        except Exception as error:
            report['errors'].append(name + ': ' + type(error).__name__ + ': ' + str(error))

    for name in (
        'numpy',
        'scipy',
        'yaml',
        'rospkg',
        'rospy',
        'piper_sdk',
        'can',
        'meshcat',
        'pika_locator_runtime',
        'utils.core',
    ):

        def load(module=name):
            imported = importlib.import_module(module)
            return {
                'version': getattr(imported, '__version__', 'import OK'),
                'path': getattr(imported, '__file__', ''),
            }

        check(name, load)

    def symbolic_fk():
        import casadi as ca
        import numpy as np
        import pinocchio as pin
        import pinocchio.casadi as cpin

        model = pin.buildSampleModelManipulator()
        symbolic = cpin.Model(model)
        data = symbolic.createData()
        q = ca.SX.sym('q', model.nq)
        cpin.forwardKinematics(symbolic, data, q)
        fk = ca.Function('pika_fk_smoke', [q], [data.oMi[-1].translation])
        result = np.asarray(fk(np.zeros(model.nq)))
        assert np.isfinite(result).all()
        return {
            'pinocchio': pin.__version__,
            'casadi': ca.__version__,
            'binding': cpin.__file__,
            'position': result.ravel().tolist(),
        }

    check('pinocchio_casadi_fk', symbolic_fk)

    def ipopt():
        import casadi as ca

        x = ca.SX.sym('x')
        solver = ca.nlpsol(
            'pika_ipopt_smoke',
            'ipopt',
            {'x': x, 'f': (x - 1) ** 2},
            {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': False},
        )
        result = solver(x0=0)
        assert solver.stats()['success']
        assert abs(float(result['x']) - 1) < 1e-6
        return {'solution': float(result['x']), 'status': solver.stats()['return_status']}

    check('ipopt', ipopt)

    def controller_import():
        import importlib.util

        script = Path(__file__).resolve().parents[1] / 'control_node.py'
        spec = importlib.util.spec_from_file_location('pika_controller_env_check', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Do not instantiate Controller: no ROS registration, services or timers.
        return 'Controller module imported; no node or driver instantiated'

    check('controller_import', controller_import)

    def model_resources():
        from .configuration import package_file

        return {
            package: str(package_file(relative, package=package))
            for package, relative in (
                ('piper_description', 'urdf/piper_description.urdf'),
                ('piper_with_gripper_moveit', 'config/piper_description.srdf'),
            )
        }

    check('model_resources', model_resources)

    def sdk_protocol():
        import piper_sdk

        if not hasattr(piper_sdk, 'C_PiperInterface_V3'):
            raise ImportError('sdk_protocol=v3 requires piper-sdk==0.7.0a1')
        return piper_sdk.__file__

    check('piper_sdk_v3', sdk_protocol)

    report['ok'] = not report['errors']
    return report
