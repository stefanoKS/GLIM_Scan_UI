"""The browser payload contract for preparation and mesh reconstruction.

Both request models validate the nested ``vdbfusion`` object with ``extra='forbid'``, so a
control the browser nests there but the schema does not define rejects the whole request
with 422 - which is what happened to ``mesh_output_mode``, a top-level execution-only field
of the mesh request. The key sets below are read from ``ui/frontend/app.js`` and replayed
against the API, so a drifted control fails here instead of in the browser, and a plain
unit test cannot drift from the real payload builder.
"""
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from factory_mapping import nksr_jobs
from factory_mapping import vdbfusion as V
from factory_mapping import vdbfusion_jobs as jobs
from factory_mapping.api import MeshRequest, ReconstructionRequest, VDBFusionSettings, make_app
from factory_mapping.reconstruction_jobs import validate_vdbfusion_settings
from factory_mapping.service import Service
from factory_mapping.storage import atomic_json

ROOT = Path(__file__).resolve().parents[1]
APP_JS = ROOT/'ui/frontend/app.js'
TRAJECTORY = 'processing/run_001/glim_dump/traj_lidar.txt'


def object_keys(literal):
    """Top-level ``key:`` names of one JS object literal."""
    depth, token, keys = 0, '', []
    for char in literal:
        if char in '([{':
            if depth == 0:
                token = ''
            depth += 1
        elif char in ')]}':
            if depth == 0:
                break
            depth -= 1
        elif depth:
            continue
        elif char == ':':
            if token.strip().isidentifier():
                keys.append(token.strip())
            token = ''
        elif char == ',':
            token = ''
        else:
            token += char
    return keys


def builder_body(function):
    """The inside of the object literal one frontend builder returns.

    Plain text scanning is enough: these builders are a flat list of ``key:value`` pairs,
    and the test fails loudly if one is ever rewritten in a way this cannot read.
    """
    source = APP_JS.read_text()
    start = source.index('return {', source.index(f'function {function}(){{')) + len('return ')
    depth = 0
    for index, char in enumerate(source[start:]):
        if char in '([{':
            depth += 1
        elif char in ')]}':
            depth -= 1
            if depth == 0:
                return source[start + 1:start + index]
    raise AssertionError(f'{function}() does not return an object literal in {APP_JS.name}')


def frontend_keys(function):
    """Keys the browser nests in the settings object of one frontend builder."""
    body = builder_body(function)
    keys = object_keys(body)
    # A spread such as ``...roiValues()`` nests that helper's keys as well.
    for spread in re.findall(r'\.\.\.(\w+)\(\)', body):
        keys += frontend_keys(spread)
    return keys


# One valid value per browser control, in the units the browser reads from the DOM.
# mesh_output_mode is listed so that, if the browser were to nest it again, the payload
# replay fails on the schema instead of on a missing sample value.
SAMPLE_VALUES = dict(
    mesh_output_mode='merged', preset=None, voxel_size_m=0.02, sdf_trunc_m=0.06, space_carving=False,
    roi_min_m=None, roi_max_m=None, origin_error_budget_m=None, association_spacing_multiplier=None,
    boundary_margin_m=None, unsupported_observations='exclude', mask_deleted_triangles=True,
    memory_budget_gib=None, preparation_voxel_size_m=0.01, mode='auto', device='auto',
    detail_level=0.5, chunk_size=None, normal_knn=64, normal_drop_angle_deg=85, mise_iter=1)


def browser_body(function='vdbfusionSettings', **overrides):
    """The object the browser builds, key for key, with a valid value per control."""
    keys = frontend_keys(function)
    missing = [key for key in keys if key not in SAMPLE_VALUES]
    assert not missing, f'Add a sample value for the new browser control(s) {missing}'
    body = {key: SAMPLE_VALUES[key] for key in keys}
    body.update(overrides)
    return body


def frontend_presets():
    """VDBFUSION_PRESETS from app.js: preset -> (voxel metres, truncation metres)."""
    literal = re.search(r'const VDBFUSION_PRESETS=(\{[^}]*\})', APP_JS.read_text()).group(1)
    return {name: (float(voxel), float(trunc)) for name, voxel, trunc in
            re.findall(r'(\w+):\[([\d.]+),([\d.]+)\]', literal)}


# --------------------------------------------------------------- the payload contract
def test_frontend_vdbfusion_settings_are_api_fields():
    """Regression: the browser nested mesh_output_mode in an object that forbids extras.

    Both VDBFusion prepare and mesh requests failed with HTTP 422 because the nested
    settings object is validated with ``extra='forbid'``. mesh_output_mode is a top-level
    execution-only field of the mesh request, never a nested TSDF setting.
    """
    keys = frontend_keys('vdbfusionSettings')
    assert 'mesh_output_mode' not in keys
    unknown = [key for key in keys if key not in VDBFusionSettings.model_fields]
    assert not unknown, (f'{APP_JS.name} nests {unknown} in the VDBFusion settings object, but '
                         'VDBFusionSettings forbids unknown fields, so the whole request is rejected with 422')


def test_every_pinned_semantic_setting_is_visible_to_the_browser():
    """The UI can only detect a stale preparation if it sends every semantic setting."""
    assert set(V.SEMANTIC_SETTING_KEYS) <= set(frontend_keys('vdbfusionSettings'))


def test_frontend_nksr_settings_are_mesh_request_fields():
    unknown = [key for key in frontend_keys('nksrSettings') if key not in MeshRequest.model_fields]
    assert not unknown, f'{APP_JS.name} sends NKSR mesh fields the API does not define: {unknown}'


def test_the_schema_still_forbids_unknown_nested_fields():
    """The fix belongs in the payload, not in a weakened schema."""
    for build in (lambda: VDBFusionSettings(**browser_body(mesh_output_mode='merged')),
                  lambda: ReconstructionRequest(trajectory='t', vdbfusion={'mesh_output_mode': 'merged'}),
                  lambda: MeshRequest(vdbfusion={'mesh_output_mode': 'merged'})):
        with pytest.raises(Exception) as error:
            build()
        assert error.value.errors()[0]['type'] == 'extra_forbidden'


def test_the_browser_settings_object_is_accepted_by_both_models():
    settings = browser_body()
    assert ReconstructionRequest(trajectory='t', algorithm='vdbfusion', vdbfusion=settings).vdbfusion
    request = MeshRequest(mesh_output_mode='both', vdbfusion=settings)
    assert request.mesh_output_mode == 'both' and request.vdbfusion.sdf_trunc_m == 0.06


@pytest.fixture
def prepared_session(root):
    """A session with one prepared VDBFusion run and one legacy NKSR run."""
    service = Service(root, True)
    item = service.sessions.create('payload contract', '', service.config)
    session = service.sessions.get(item['id'])
    (session/'raw_bag').mkdir(parents=True)
    (session/'raw_bag/metadata.yaml').write_text('rosbag2_bagfile_information:\n  version: 5\n')
    vdbfusion_run = session/'reconstruction/run_0123456789ab'
    nksr_run = session/'reconstruction/run_fedcba987654'
    for run, job in ((vdbfusion_run, dict(algorithm='vdbfusion', state='PREPARED')),
                     (nksr_run, dict(state='PREPARED'))):
        (run/'input').mkdir(parents=True)
        atomic_json(run/'job.json', job)
    return service, item['id'], vdbfusion_run, nksr_run


@pytest.fixture
def capture_launchers(monkeypatch):
    """Capture the launchers instead of starting a worker for a real bag."""
    from factory_mapping import reconstruction_jobs
    prepare_calls, mesh_calls, nksr_calls = [], [], []

    async def prepare(service, sid, trajectory, size, save_full, *args, **kwargs):
        prepare_calls.append(dict(sid=sid, trajectory=trajectory, size=size,
                                  filter_edited_geometry=args[0] if args else kwargs.get('filter_edited_geometry', False),
                                  edit_id=args[1] if len(args) > 1 else kwargs.get('edit_id'),
                                  filter_tolerance_m=args[2] if len(args) > 2 else kwargs.get('filter_tolerance_m'),
                                  **kwargs))
        return {'state': 'running'}

    async def vdbfusion_mesh(service, sid, rid, request):
        mesh_calls.append(dict(sid=sid, run=rid, request=request))
        return {'state': 'RUNNING'}

    async def nksr_mesh(service, sid, rid, request):
        nksr_calls.append(dict(sid=sid, run=rid, request=request))
        return {'state': 'RUNNING'}

    monkeypatch.setattr(reconstruction_jobs, 'start', prepare)
    monkeypatch.setattr(jobs, 'reconstruct', vdbfusion_mesh)
    monkeypatch.setattr(nksr_jobs, 'reconstruct', nksr_mesh)
    return SimpleNamespace(prepare=prepare_calls, mesh=mesh_calls, nksr=nksr_calls)


# --------------------------------------------------------------- preparation
def test_prepare_accepts_the_browser_request_for_every_preset(root, capture_launchers):
    presets = dict(frontend_presets(), custom=(0.03, 0.09))
    assert set(presets) == {'fast', 'detailed', 'experimental', 'custom'}
    with TestClient(make_app(root, True)) as client:
        for preset, (voxel, trunc) in presets.items():
            settings = browser_body(preset=None if preset == 'custom' else preset,
                                    voxel_size_m=voxel, sdf_trunc_m=trunc)
            body = dict(trajectory=TRAJECTORY, algorithm='vdbfusion', voxel_size_m=0.01,
                        filter_edited_geometry=False, vdbfusion=settings)
            response = client.post('/api/sessions/test/reconstruction', json=body)
            assert response.status_code == 202, response.text
            forwarded = capture_launchers.prepare[-1]['vdbfusion']
            assert forwarded['voxel_size_m'] == voxel and forwarded['sdf_trunc_m'] == trunc
            assert forwarded['preset'] == (None if preset == 'custom' else preset)
            assert 'mesh_output_mode' not in forwarded
            assert capture_launchers.prepare[-1]['algorithm'] == 'vdbfusion'


def test_prepare_bodies_for_edited_and_unedited_browser_requests(root, capture_launchers):
    with TestClient(make_app(root, True)) as client:
        # Unedited: no cleanup, no tolerance, semantic settings only.
        unedited = dict(trajectory=TRAJECTORY, algorithm='vdbfusion', voxel_size_m=0.01,
                        filter_edited_geometry=False, vdbfusion=browser_body())
        assert client.post('/api/sessions/test/reconstruction', json=unedited).status_code == 202
        call = capture_launchers.prepare[-1]
        assert call['filter_edited_geometry'] is False and call['edit_id'] is None
        assert call['filter_tolerance_m'] is None
        # Edited VDBFusion: the cleanup id travels, the NKSR-only tolerance does not.
        edited = dict(trajectory=TRAJECTORY, algorithm='vdbfusion', voxel_size_m=0.01,
                      filter_edited_geometry=True, edit_id='edit_012345abcdef',
                      vdbfusion=browser_body())
        assert client.post('/api/sessions/test/reconstruction', json=edited).status_code == 202
        call = capture_launchers.prepare[-1]
        assert call['filter_edited_geometry'] is True and call['edit_id'] == 'edit_012345abcdef'
        assert call['filter_tolerance_m'] is None
        # Edited NKSR: the tolerance is required and still sent.
        nksr = dict(trajectory=TRAJECTORY, algorithm='nksr', voxel_size_m=0.01,
                    filter_edited_geometry=True, edit_id='edit_012345abcdef', filter_tolerance_m=0.05)
        assert client.post('/api/sessions/test/reconstruction', json=nksr).status_code == 202
        call = capture_launchers.prepare[-1]
        assert call['algorithm'] == 'nksr' and call['filter_tolerance_m'] == 0.05
        assert call['vdbfusion'] is None


def test_the_hidden_tolerance_cannot_be_sent_as_zero():
    """A cleared NKSR tolerance field reads as 0, which the schema refuses.

    The browser only sends filter_tolerance_m for NKSR, whose control is visible and
    validated; a hidden VDBFusion request never carries it.
    """
    with pytest.raises(Exception) as error:
        ReconstructionRequest(trajectory='t', algorithm='vdbfusion', filter_edited_geometry=True,
                              edit_id='edit_012345abcdef', filter_tolerance_m=0, vdbfusion=browser_body())
    assert error.value.errors()[0]['type'] == 'greater_than'
    assert "filtering&&engine==='nksr'?{filter_tolerance_m:tolerance}" in APP_JS.read_text()


# --------------------------------------------------------------- mesh reconstruction
def test_mesh_accepts_the_browser_request_for_every_output_mode(prepared_session, capture_launchers):
    service, sid, vdbfusion_run, nksr_run = prepared_session
    with TestClient(make_app(service.root, True)) as client:
        for mode in ('merged', 'chunks', 'both'):
            body = dict(mesh_output_mode=mode, vdbfusion=browser_body())
            response = client.post(f'/api/sessions/{sid}/reconstruction/{vdbfusion_run.name}/mesh', json=body)
            assert response.status_code == 202, response.text
            request = capture_launchers.mesh[-1]['request']
            # The output mode is execution-only, so the route takes it from the top level.
            assert request['mesh_output_mode'] == mode
            assert request['voxel_size_m'] == 0.02 and request['sdf_trunc_m'] == 0.06
            assert request['chunk_size'] is None
            assert not capture_launchers.nksr
        assert capture_launchers.mesh[-1]['run'] == vdbfusion_run.name


def test_mesh_keeps_nksr_requests_on_the_nksr_path(prepared_session, capture_launchers):
    service, sid, vdbfusion_run, nksr_run = prepared_session
    body = browser_body('nksrSettings', mesh_output_mode='chunks', chunk_size=2.0)
    with TestClient(make_app(service.root, True)) as client:
        response = client.post(f'/api/sessions/{sid}/reconstruction/{nksr_run.name}/mesh', json=body)
        assert response.status_code == 202, response.text
    assert not capture_launchers.mesh
    request = capture_launchers.nksr[-1]['request']
    assert request['mesh_output_mode'] == 'chunks' and request['chunk_size'] == 2.0
    assert 'vdbfusion' not in request


def test_prepared_settings_are_compared_but_the_output_mode_is_not(prepared_session, capture_launchers):
    """A semantic change is stale; the mesh output mode may change without re-preparing."""
    service, sid, vdbfusion_run, nksr_run = prepared_session
    settings = browser_body()
    record = dict(state='PREPARED', settings=validate_vdbfusion_settings(settings),
                  identity=dict(source=dict()))
    assert jobs.enforce_prepared_identity(vdbfusion_run, record, {}, settings)['voxel_size_m'] == 0.02
    for mode in ('merged', 'chunks', 'both'):
        effective = jobs.enforce_prepared_identity(vdbfusion_run, record, {}, dict(settings, mesh_output_mode=mode))
        assert effective['mesh_output_mode'] == mode
    with pytest.raises(ValueError) as error:
        jobs.enforce_prepared_identity(vdbfusion_run, record, {}, dict(settings, voxel_size_m=0.01))
    assert 'PREPARED_SETTINGS_STALE' in str(error.value) and 'voxel_size_m' in str(error.value)


def test_disk_preflight_plans_for_the_requested_output_mode(tmp_path, monkeypatch):
    """chunks and both publish more than merged, and the preflight decides on that mode."""
    import psutil
    disk = SimpleNamespace(total=10 * 1024 ** 3, used=0, free=0, percent=0)
    monkeypatch.setattr(psutil, 'virtual_memory', lambda: SimpleNamespace(available=64 * 1024 ** 3,
                                                                         total=64 * 1024 ** 3))
    monkeypatch.setattr(psutil, 'disk_usage', lambda path: disk)
    settings = V.resolve_settings(dict(browser_body()))
    observed = dict(observed_points=4_000_000, observed_bbox=([-10.0, -10.0, 0.0], [10.0, 10.0, 4.0]))

    def report(mode, free):
        disk.free = int(free)
        return V.preflight(tmp_path, dict(settings, mesh_output_mode=mode), **observed)

    disk.free = 64 * 1024 ** 3
    merged, chunks, both = (report(mode, disk.free) for mode in ('merged', 'chunks', 'both'))
    assert (merged['estimated_disk_requirement_bytes'] < chunks['estimated_disk_requirement_bytes']
            < both['estimated_disk_requirement_bytes'])
    assert merged['ok'] and chunks['ok'] and both['ok']
    # Between the two requirements the merged set fits and "both" is refused, so the
    # selected output mode - not the preparation default - decides the disk preflight.
    midpoint = (merged['estimated_disk_requirement_bytes'] + both['estimated_disk_requirement_bytes'])//2
    assert both['estimated_disk_requirement_bytes'] - merged['estimated_disk_requirement_bytes'] > 1024 ** 2
    assert report('merged', midpoint)['ok'] is True
    refused = report('both', midpoint)
    assert refused['ok'] is False and refused['code'] == 'RESOURCE_PREFLIGHT_FAILED'
    assert 'both output set' in refused['failures'][0]


# --------------------------------------------------------------- readable errors
def test_rejected_requests_return_readable_field_paths(root, capture_launchers):
    """The 422 body names the field and the reason, so the UI can render both."""
    with TestClient(make_app(root, True)) as client:
        cases = [
            (dict(trajectory='t', algorithm='vdbfusion', vdbfusion={'mesh_output_mode': 'merged'}),
             ['body', 'vdbfusion', 'mesh_output_mode'], 'extra_forbidden'),
            (dict(trajectory='t', algorithm='vdbfusion', vdbfusion=dict(browser_body(), voxel_size_m=-1)),
             ['body', 'vdbfusion', 'voxel_size_m'], 'greater_than'),
            (dict(trajectory='t', algorithm='poisson'),
             ['body', 'algorithm'], 'literal_error'),
            (dict(trajectory='t', filter_edited_geometry=True, edit_id='e', filter_tolerance_m=0),
             ['body', 'filter_tolerance_m'], 'greater_than'),
            ({}, ['body', 'trajectory'], 'missing'),
        ]
        for body, loc, kind in cases:
            response = client.post('/api/sessions/test/reconstruction', json=body)
            assert response.status_code == 422, (body, response.text)
            detail = response.json()['detail']
            assert isinstance(detail, list) and detail
            entry = next((item for item in detail if item['loc'][:len(loc)] == loc), None) or detail[0]
            assert entry['loc'][:len(loc)] == loc and entry['type'] == kind, (body, detail)
            assert isinstance(entry['msg'], str) and entry['msg']
            assert not capture_launchers.prepare
        # An unknown preset is also reported as its own field path.
        response = client.post('/api/sessions/test/reconstruction',
                               json={'trajectory': 't', 'vdbfusion': {'preset': 'unknown'}})
        assert response.status_code == 422, response.text
        assert any(item['loc'] == ['body', 'vdbfusion', 'preset'] for item in response.json()['detail'])
        assert 'traceback' not in response.text.lower()
        assert not capture_launchers.prepare


def test_mesh_rejects_an_unknown_output_mode_readably(prepared_session, capture_launchers):
    service, sid, vdbfusion_run, nksr_run = prepared_session
    with TestClient(make_app(service.root, True)) as client:
        response = client.post(f'/api/sessions/{sid}/reconstruction/{vdbfusion_run.name}/mesh',
                               json=dict(mesh_output_mode='poisson', vdbfusion=browser_body()))
    assert response.status_code == 422
    entry = response.json()['detail'][0]
    assert entry['loc'] == ['body', 'mesh_output_mode'] and entry['msg']
    assert not capture_launchers.mesh


def test_validation_errors_render_as_field_paths_in_the_browser():
    """Run the browser-side formatter against the real 422 detail shape."""
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is not available to run the frontend formatter test')
    test = ROOT/'tests/test_api_errors.mjs'
    result = subprocess.run([node, str(test)], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'API error rendering checks passed' in result.stdout


# ------------------------------------------------------------------- payload audit
def test_no_other_browser_payload_is_rejected_by_a_schema(root):
    """Every other JSON body the browser builds still matches its request model.

    Each endpoint answers with its own business status (200/202/409); a 422 means the
    browser and the API disagree about the payload, which is the defect this file exists
    to catch. The bodies are transcribed from ui/frontend/*.js.
    """
    with TestClient(make_app(root, True)) as client:
        session = client.post('/api/sessions', json={'name': 'payload audit', 'notes': ''}).json()['id']
        bodies = [
            ('POST', 'action', {'action': 'diagnose', 'session': None, 'preset': 'jetson_cpu', 'run': None}),
            ('PATCH', f'sessions/{session}', {'name': 'payload audit', 'notes': 'note'}),
            ('POST', 'tools/open', {'session': session, 'run': 'run_0123456789ab',
                                    'tool': 'offline_viewer', 'additional': []}),
            ('POST', 'capture/action', {'action': 'stop_scan'}),
            ('PUT', 'capture/settings', {'live_glim': False, 'auto_process': True, 'mapping_preset': 'pc_dense'}),
            ('POST', 'camera/enabled', {'enabled': False}),
            ('PUT', 'camera/profile', {'profile': 'auto'}),
            ('POST', f'calibrations/{session}/action', {'action': 'cancel', 'notes': ''}),
            ('POST', 'nksr/check', {'device': 'auto'}),
            ('POST', 'vdbfusion/check', {}),
            ('POST', f'sessions/{session}/colorization', {'allow_unvalidated_calibration': False}),
        ]
        for method, path, body in bodies:
            response = client.request(method, f'/api/{path}', json=body)
            # 404 would mean the route moved, 500 an unreported defect and 422 a payload
            # mismatch; only the business statuses are acceptable here.
            assert response.status_code in (200, 202, 409), (path, body, response.text)
