// Regression: a FastAPI validation body must render as field paths, never as "[object Object]".
// Run with: node tests/test_api_errors.mjs
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../ui/frontend/api-errors.js',import.meta.url),'utf8');
const {apiErrorMessage}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));

// The exact 422 the API returned for the nested mesh output mode before the payload fix.
const extraField={detail:[{type:'extra_forbidden',loc:['body','vdbfusion','mesh_output_mode'],
 msg:'Extra inputs are not permitted',input:'merged',url:'https://errors.pydantic.dev/2.13/v/extra_forbidden'}]};
const rendered=apiErrorMessage(extraField,'Request failed (422)');
assert.equal(rendered,'Invalid request: vdbfusion.mesh_output_mode: Extra inputs are not permitted');
assert.doesNotMatch(rendered,/\[object Object\]/);
// The rejected input and the documentation URL are not echoed into the banner.
assert.doesNotMatch(rendered,/merged|errors\.pydantic\.dev/);

// Field paths keep list indices and drop FastAPI's "body" prefix.
assert.equal(apiErrorMessage({detail:[{loc:['body','roi_min_m',0],msg:'Input should be a valid number'}]},'x'),
 'Invalid request: roi_min_m[0]: Input should be a valid number');
assert.equal(apiErrorMessage({detail:[{loc:['body','vdbfusion','voxel_size_m'],
 msg:'Input should be greater than 0'}]},'x'),
 'Invalid request: vdbfusion.voxel_size_m: Input should be greater than 0');
// Several errors are listed once each, not merged into one opaque string.
assert.equal(apiErrorMessage({detail:[{loc:['body','a'],msg:'one'},{loc:['body','b'],msg:'two'},
 {loc:['body','a'],msg:'one'}]},'x'),'Invalid request: a: one; b: two');
assert.match(apiErrorMessage({detail:Array.from({length:11},(unused,index)=>({loc:['body','k'+index],msg:'bad'}))},'x'),
 /and 3 more errors$/);

// Handled errors, string and object details.
assert.equal(apiErrorMessage({detail:'Prepare VDBFusion point input first'},'x'),'Prepare VDBFusion point input first');
assert.equal(apiErrorMessage({detail:{message:'VDBFusion check already running'}},'x'),'VDBFusion check already running');
assert.equal(apiErrorMessage('Import failed','x'),'Import failed');

// Unknown object shapes fall back instead of dumping internals or printing a placeholder.
assert.equal(apiErrorMessage({traceback:'File "x.py"',stack:'S','exception':'E'},'Request failed (500)'),'Request failed (500)');
assert.equal(apiErrorMessage({detail:{unknown:[1,2,3]}},'Request failed (422)'),'Request failed (422)');
assert.equal(apiErrorMessage(null,'Request failed (422)'),'Request failed (422)');
assert.equal(apiErrorMessage(undefined,undefined),'Request failed');
assert.equal(apiErrorMessage({},''),'Request failed');

// Malformed entries, nested arrays and non-string locations are handled without throwing.
const hostile={detail:[null,{},42,[{loc:['body','deep'],msg:'listed'}],{loc:'not-a-list',msg:'no path'},
 {loc:['body',{nested:true}],msg:'object path'},{msg:{nested:true}},{loc:['body','f'],msg:''}]};
const hostileText=apiErrorMessage(hostile,'x');
assert.doesNotMatch(hostileText,/\[object Object\]|undefined|null/);
assert.match(hostileText,/no path/);
assert.match(hostileText,/object path/);

// A very long upstream message is clipped instead of filling the page.
assert.equal(apiErrorMessage('x'.repeat(2000),'y').length,600);
assert.doesNotMatch(apiErrorMessage({detail:[{loc:['body','a'],msg:'y'.repeat(2000)}]},'z'),/\[object Object\]/);

console.log('API error rendering checks passed');
