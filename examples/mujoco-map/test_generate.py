import importlib.util
import pathlib
import tempfile
import unittest

SCRIPT = pathlib.Path(__file__).with_name("generate.py")
spec = importlib.util.spec_from_file_location("occupancy_generate", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PublicationTests(unittest.TestCase):
    def test_dataset_is_published_only_when_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            output = pathlib.Path(temp) / "dataset"
            module.publish_dataset(output, [("depth_000.f32le", b"\x00\x00\x80\x3f")], {"schema_version": 1})
            self.assertEqual((output / "depth_000.f32le").read_bytes(), b"\x00\x00\x80\x3f")
            self.assertIn('"schema_version": 1', (output / "manifest.json").read_text())
            with self.assertRaises(FileExistsError):
                module.publish_dataset(output, [], {})


if __name__ == "__main__":
    unittest.main()
