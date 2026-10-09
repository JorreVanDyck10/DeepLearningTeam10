"""Publish only the checksum-pinned deployment package as a GitHub release asset.

Uses the existing Git Credential Manager account; credentials never enter output.
"""
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = 'JorreVanDyck10/DeepLearningTeam10'
TAG = 'raoul-citibike-rf-20261009'


def main():
    receipt = json.loads((ROOT/'backend/citibike_artifact.json').read_text(encoding='utf-8'))
    archive = ROOT/'SolutionRaoul/Citybike/models/raoul-citibike-rf-20261009.zip'
    from backend.raoul_citibike import checksum
    if checksum(archive) != receipt['archive_sha256']:
        raise ValueError('Deployment package differs from tested checksum')
    credential = subprocess.run(['git','credential','fill'], input='protocol=https\nhost=github.com\n\n',
                                cwd=ROOT, capture_output=True, text=True, check=True, timeout=60)
    values = dict(line.split('=',1) for line in credential.stdout.splitlines() if '=' in line)
    token = values.get('password')
    if not token:
        raise RuntimeError('No GitHub credential available')
    headers = {'Authorization': 'Bearer '+token, 'Accept': 'application/vnd.github+json',
               'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'DeepLearningTeam10'}
    base = f'https://api.github.com/repos/{REPOSITORY}/releases'
    def api(url, payload=None, method=None):
        if not url.startswith('https://api.github.com/'):
            raise ValueError('Unexpected API destination')
        request = urllib.request.Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                                         headers={**headers,'Content-Type':'application/json'}, method=method)
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    releases = api(base+'?per_page=100')
    release = next((item for item in releases if item['tag_name']==TAG),None)
    if release is None:
        release = api(base, {'tag_name': TAG, 'target_commitish': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                             'name': 'Raoul Citi Bike — evaluated Random Forest deployment artifact',
                             'body': 'Lossless inference export of the selected 320-tree NYC Random Forest. Same tree splits and leaf values; no retraining. Original model SHA-256: '+receipt['original_model_sha256']+'. Package SHA-256: '+receipt['archive_sha256']+'. Includes development-only demonstration history. Model files remain outside Git.',
                             'draft':False,'prerelease':False,'make_latest':'false'}, 'POST')
    existing = next((item for item in release['assets'] if item['name']==archive.name),None)
    if existing:
        if existing['size'] != archive.stat().st_size or existing.get('digest') != 'sha256:'+receipt['archive_sha256']:
            raise ValueError('Existing release asset differs; never overwrite a pinned artifact')
        result = existing
    else:
        # Stream bytes; no secrets in URL, file or console.
        import http.client
        upload_path = f'/repos/{REPOSITORY}/releases/{release["id"]}/assets?name={archive.name}'
        connection = http.client.HTTPSConnection('uploads.github.com',timeout=600)
        connection.putrequest('POST',upload_path)
        for name, value in {**headers,'Content-Type':'application/zip','Content-Length':str(archive.stat().st_size)}.items():
            connection.putheader(name,value)
        connection.endheaders()
        with archive.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                connection.send(block)
        response = connection.getresponse()
        body = response.read()
        if response.status != 201:
            raise RuntimeError(f'GitHub artifact upload failed: HTTP {response.status}')
        result = json.loads(body)
        connection.close()
    if result['browser_download_url'] != receipt['url'] or result.get('digest') != 'sha256:'+receipt['archive_sha256']:
        raise ValueError('Published artifact lineage/checksum mismatch')
    print(json.dumps({'release':release['html_url'],'asset':result['browser_download_url'],
                      'size_bytes':result['size'],'sha256':receipt['archive_sha256']},indent=2))


if __name__ == '__main__':
    main()
