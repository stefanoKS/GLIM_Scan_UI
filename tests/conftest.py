import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ui/backend'))
import pytest, shutil
@pytest.fixture
def root(tmp_path):
    shutil.copytree(ROOT/'config',tmp_path/'config')
    # Most fixtures exercise LiDAR-only operation; production defaults include RGB.
    import yaml
    system=yaml.safe_load((tmp_path/'config/system.yaml').read_text())
    system['camera']['enabled']=False
    (tmp_path/'config/system.yaml').write_text(yaml.safe_dump(system))
    (tmp_path/'config/calibration/camera_intrinsics.yaml').write_text('calibrated: false\n')
    (tmp_path/'ui/frontend').mkdir(parents=True);(tmp_path/'ui/frontend/index.html').write_text('test')
    (tmp_path/'.state').mkdir()
    return tmp_path
