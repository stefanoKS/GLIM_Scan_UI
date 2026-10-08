import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ui/backend'))
import pytest, shutil
from factory_mapping import vdbfusion_jobs


@pytest.fixture(scope='session')
def vdbfusion_python():
    """Isolated VDBFusion interpreter, or skip the test when it is not installed.

    The native library lives only in that environment, so tests that need it must
    skip rather than silently pass.
    """
    python = vdbfusion_jobs.interpreter(ROOT)
    if not python.is_file():
        pytest.skip('VDBFusion is not installed; run scripts/setup_vdbfusion.sh')
    return python


@pytest.fixture
def root(tmp_path):
    shutil.copytree(ROOT/'config',tmp_path/'config')
    # Most fixtures exercise LiDAR-only operation; production defaults include RGB.
    import yaml
    system=yaml.safe_load((tmp_path/'config/system.yaml').read_text())
    system['camera']['enabled']=False
    system['camera']['profile']='dfk33ux287' # Existing DFK/ChArUco regression fixtures.
    (tmp_path/'config/system.yaml').write_text(yaml.safe_dump(system))
    (tmp_path/'config/calibration/camera_intrinsics.yaml').write_text('calibrated: false\n')
    (tmp_path/'ui/frontend').mkdir(parents=True);(tmp_path/'ui/frontend/index.html').write_text('test')
    (tmp_path/'.state').mkdir()
    (tmp_path/'.state/deployment.json').write_text('{"mode":"workstation"}')
    return tmp_path
