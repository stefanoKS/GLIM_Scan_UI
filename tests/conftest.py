import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ui/backend'))
import pytest, shutil
@pytest.fixture
def root(tmp_path):
    shutil.copytree(ROOT/'config',tmp_path/'config')
    (tmp_path/'ui/frontend').mkdir(parents=True);(tmp_path/'ui/frontend/index.html').write_text('test')
    (tmp_path/'.state').mkdir()
    return tmp_path
