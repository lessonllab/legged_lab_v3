"""Explicit archived actor selection; never silently play only the base MLP."""
import hashlib
import importlib.util
from pathlib import Path
import sys


def configure_actor(settings,contract,checkpoint):
    method=contract.get('support_transition')
    if method is None:return settings
    version=method.get('version')
    if version not in (1,2):raise ValueError('Unsupported support-transition actor version')
    archive=Path(checkpoint).parent/'params'
    if version==2:
        dependency=archive/'beamdojo_support_transition_v1.py'
        if not dependency.is_file():raise FileNotFoundError(str(dependency))
        expected=method.get('base_actor_source_sha256')
        if not expected or hashlib.sha256(dependency.read_bytes()).hexdigest()!=expected:
            raise ValueError('Archived base actor source digest mismatch')
        spec=importlib.util.spec_from_file_location('beamdojo_support_transition_v1',dependency)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    path=archive/f'beamdojo_support_transition_v{version}.py'
    if not path.is_file():raise FileNotFoundError('Missing archived support-transition actor: '+str(path))
    expected=method.get('actor_source_sha256')
    if expected and hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError('Archived support-transition source digest mismatch')
    name=f'beamdojo_support_transition_v{version}'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    settings['actor']['class_name']=module.ACTOR_CLASS
    settings['actor']['transition_joint_names']=contract['joint_names']
    return settings
