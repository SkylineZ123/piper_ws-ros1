#!/usr/bin/env python3
"""Verify the Pika bringup environment without opening any hardware devices."""

import importlib
import json
from pathlib import Path
import sys
import tempfile


def main():
    report = {'python': sys.executable, 'python_version': sys.version.split()[0],
              'checks': {}, 'errors': []}

    def check(name, function):
        try:
            report['checks'][name] = function()
        except Exception as error:
            report['errors'].append(name + ': ' + type(error).__name__ + ': ' + str(error))

    for name in ('numpy', 'scipy', 'yaml', 'rospkg', 'rospy', 'rosbag', 'tf.transformations',
                 'piper_sdk', 'can', 'meshcat', 'pika_locator_runtime', 'pika_piper.core'):
        def load(module=name):
            imported = importlib.import_module(module)
            return {'version': getattr(imported, '__version__', 'import OK'),
                    'path': getattr(imported, '__file__', '')}
        check(name, load)

    def symbolic_fk():
        import numpy as np
        import casadi as ca
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
        return {'pinocchio': pin.__version__, 'casadi': ca.__version__,
                'binding': cpin.__file__, 'position': result.ravel().tolist()}
    check('pinocchio_casadi_fk', symbolic_fk)

    def ipopt():
        import casadi as ca
        x = ca.SX.sym('x')
        solver = ca.nlpsol('pika_ipopt_smoke', 'ipopt', {'x': x, 'f': (x - 1)**2},
                          {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': False})
        result = solver(x0=0)
        assert solver.stats()['success']
        assert abs(float(result['x']) - 1) < 1e-6
        return {'solution': float(result['x']), 'status': solver.stats()['return_status']}
    check('ipopt', ipopt)

    def image_roundtrip():
        import numpy as np
        from cv_bridge import CvBridge
        bridge = CvBridge()
        pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        msg = bridge.cv2_to_imgmsg(pixels, encoding='rgb8')
        assert np.array_equal(bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8'), pixels)
        return 'RGB conversion roundtrip passed'
    check('cv_bridge', image_roundtrip)

    def bag_roundtrip():
        import rosbag
        from std_msgs.msg import String
        with tempfile.TemporaryDirectory(prefix='pika-env-bag-') as directory:
            path = str(Path(directory) / 'check.bag')
            with rosbag.Bag(path, 'w', compression='lz4') as bag:
                bag.write('/environment_check', String(data='pika'))
            with rosbag.Bag(path, 'r') as bag:
                messages = list(bag.read_messages())
            assert len(messages) == 1 and messages[0].message.data == 'pika'
        return 'LZ4 rosbag write/read passed'
    check('rosbag_lz4', bag_roundtrip)

    def controller_import():
        import importlib.util
        script = Path(__file__).with_name('control_node.py')
        spec = importlib.util.spec_from_file_location('pika_controller_env_check', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Do not instantiate Controller: no ROS registration, services or timers.
        return 'Controller module imported; no node or driver instantiated'
    check('controller_import', controller_import)

    report['ok'] = not report['errors']
    print(json.dumps(report, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
