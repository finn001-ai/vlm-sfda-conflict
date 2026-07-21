import unittest
from pathlib import Path


class GTRClassRoutingDefaultsTest(unittest.TestCase):
    def test_existing_methods_keep_class_routing_disabled(self):
        config_source = Path("conf.py").read_text()
        self.assertIn("_C.DCCL.GTR_CLASS_ROUTING = False", config_source)

        for path in Path("cfgs").rglob("*.yaml"):
            self.assertNotIn("GTR_CLASS_ROUTING: True", path.read_text(), str(path))


if __name__ == "__main__":
    unittest.main()
