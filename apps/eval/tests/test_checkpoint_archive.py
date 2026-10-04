"""Restore preserves bytes and rejects corruption and local changes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('checkpoint_archive', Path(__file__).parents[1] / 'scripts/checkpoint_archive.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)

class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        archive.ROOT = root / 'source'
        archive.GROUPS = {'sample': 'sample'}
        (archive.ROOT / 'sample').mkdir(parents=True)
        (archive.ROOT / 'sample/result.bin').write_bytes(b'original\x00result\n')
        self.output = root / 'assets'
        self.target = root / 'restored'
        archive.pack('sample', self.output)
        self.manifest = self.output / 'sample.manifest.json'

    def test_restore_and_idempotent_reverification(self):
        archive.verify(self.manifest, self.output, self.target)
        self.assertEqual((self.target / 'sample/result.bin').read_bytes(), b'original\x00result\n')
        archive.verify(self.manifest, self.output, self.target)

    def test_materialize_linked_store_contents(self):
        import os
        source = archive.ROOT / 'sample'
        os.link(source / 'result.bin', source / 'hardlink.bin')
        external = archive.ROOT / 'external'
        external.mkdir()
        (external / 'state.json').write_text('{"cost": 2}')
        (source / 'alias').symlink_to(external, target_is_directory=True)
        output = self.output / 'linked'
        archive.pack('sample', output)
        archive.verify(output / 'sample.manifest.json', output, self.target)
        self.assertEqual((self.target / 'sample/hardlink.bin').read_bytes(), b'original\x00result\n')
        self.assertEqual((self.target / 'sample/alias/state.json').read_text(), '{"cost": 2}')
        self.assertFalse((self.target / 'sample/alias').is_symlink())

    def test_refuse_existing_changed_file(self):
        (self.target / 'sample').mkdir(parents=True)
        path = self.target / 'sample/result.bin'
        path.write_bytes(b'user edits')
        with self.assertRaisesRegex(ValueError, 'Existing file differs'):
            archive.verify(self.manifest, self.output, self.target)
        self.assertEqual(path.read_bytes(), b'user edits')

    def test_refuse_corrupt_archive_before_restore(self):
        with (self.output / 'sample.tar.gz').open('ab') as f:
            f.write(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            archive.verify(self.manifest, self.output, self.target)
        self.assertFalse(self.target.exists())

if __name__ == '__main__':
    unittest.main()
