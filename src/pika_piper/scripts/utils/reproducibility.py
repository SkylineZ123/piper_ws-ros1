"""Record the actual interpreter, dependency versions and source fingerprints."""

import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from urllib.parse import unquote, urlparse

DEPENDENCIES = (
    'lerobot',
    'numpy',
    'scipy',
    'torch',
    'torchvision',
    'av',
    'datasets',
    'pyarrow',
    'pandas',
    'matplotlib',
    'opencv-python',
    'opencv-python-headless',
    'rospkg',
    'catkin-pkg',
    'PyYAML',
    'piper-sdk',
    'casadi',
    'pin',
    'pinocchio',
)


def git_revision(directory):
    """Best effort only: installed distributions need not contain .git."""
    try:
        result = subprocess.run(
            ['git', '-C', str(directory), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def runtime_snapshot():
    scripts = Path(__file__).resolve().parents[1]
    packages = {}
    for name in DEPENDENCIES:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        info = {'version': dist.version}
        direct = dist.read_text('direct_url.json')
        if direct:
            info['source'] = json.loads(direct)
            url = urlparse(info['source'].get('url', ''))
            if url.scheme == 'file':
                info['source_revision'] = git_revision(unquote(url.path))
        packages[name] = info
    return {
        'python': sys.executable,
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'packages': packages,
        'project_revision': git_revision(scripts),
        'loaded_modules': {
            name: {'path': str(module.__file__), 'version': getattr(module, '__version__', None)}
            for name in ('numpy', 'scipy', 'cv2', 'rospy', 'casadi', 'pinocchio')
            for module in [sys.modules.get(name)]
            if getattr(module, '__file__', None)
        },
        # Hash actual files as well: a commit alone misses uncommitted changes.
        'source_sha256': {
            str(path.relative_to(scripts)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(scripts.rglob('*.py'))
        },
    }
