"""Release integrity failures must never install or replace the selected model."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from backend import raoul_citibike


class ArtifactTests(unittest.TestCase):
    def receipt(self, root, archive, files, digest=None):
        path = root/'receipt.json'
        path.write_text(json.dumps({'url': 'https://github.com/test/pinned.zip',
                                   'archive_sha256': digest or hashlib.sha256(archive).hexdigest(),
                                   'files': files}), encoding='utf-8')
        return path

    def test_wrong_archive_checksum_is_rejected_and_staging_is_cleaned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); body = b'not the selected model'
            receipt = self.receipt(root, body, {'forest.npy': '0'*64}, digest='1'*64)
            with patch.object(raoul_citibike, 'RECEIPT', receipt), patch.object(raoul_citibike.urllib.request, 'urlopen', return_value=io.BytesIO(body)):
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    raoul_citibike.ensure_artifact(root/'model')
            self.assertFalse((root/'model').exists())
            self.assertFalse((root/'model.partial').exists())

    def test_unexpected_archive_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); stream = io.BytesIO()
            with zipfile.ZipFile(stream,'w') as archive:
                archive.writestr('../escape.npy', b'unsafe')
            body = stream.getvalue()
            receipt = self.receipt(root, body, {'forest.npy':'0'*64})
            with patch.object(raoul_citibike, 'RECEIPT', receipt), patch.object(raoul_citibike.urllib.request, 'urlopen', return_value=io.BytesIO(body)):
                with self.assertRaisesRegex(ValueError, 'Unexpected archive members'):
                    raoul_citibike.ensure_artifact(root/'model')
            self.assertFalse((root/'escape.npy').exists())
            self.assertFalse((root/'model.partial').exists())

    def test_existing_changed_model_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); directory = root/'model'; directory.mkdir()
            (directory/'forest.npy').write_bytes(b'changed local file')
            content = b'new file'; stream = io.BytesIO()
            with zipfile.ZipFile(stream,'w') as archive:
                archive.writestr('forest.npy',content)
            body=stream.getvalue()
            receipt = self.receipt(root,body,{'forest.npy':hashlib.sha256(content).hexdigest()})
            with patch.object(raoul_citibike, 'RECEIPT', receipt), patch.object(raoul_citibike.urllib.request, 'urlopen', return_value=io.BytesIO(body)):
                with self.assertRaisesRegex(ValueError, 'do not overwrite'):
                    raoul_citibike.ensure_artifact(directory)
            self.assertEqual((directory/'forest.npy').read_bytes(),b'changed local file')


if __name__ == '__main__':
    unittest.main()
