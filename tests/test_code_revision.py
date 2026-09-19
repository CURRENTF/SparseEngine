from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sparseengine.utils.code_revision import _verified_source_repo_root
from sparseengine.utils.code_revision import code_revision_info


class CodeRevisionTest(unittest.TestCase):
    def tearDown(self):
        code_revision_info.cache_clear()

    @patch("sparseengine.utils.code_revision._git_command", return_value=None)
    @patch("sparseengine.utils.code_revision.version", return_value="0.1.0")
    def test_uses_published_distribution_name(self, mock_version, _mock_git):
        code_revision_info.cache_clear()

        revision = code_revision_info()

        mock_version.assert_called_once_with("sparseengine")
        self.assertEqual(revision["package_version"], "0.1.0")
        self.assertIsNone(revision["git_commit"])

    def test_rejects_an_enclosing_unrelated_git_root(self):
        with TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "source"
            module_path = (
                source_root
                / "src"
                / "sparseengine"
                / "utils"
                / "code_revision.py"
            )
            module_path.parent.mkdir(parents=True)
            module_path.touch()

            with patch(
                "sparseengine.utils.code_revision._git_value",
                return_value=str(Path(tmp)),
            ):
                repo_root = _verified_source_repo_root(module_path)

        self.assertIsNone(repo_root)

    def test_accepts_the_matching_source_git_root(self):
        with TemporaryDirectory() as tmp:
            source_root = Path(tmp) / "source"
            module_path = (
                source_root
                / "src"
                / "sparseengine"
                / "utils"
                / "code_revision.py"
            )
            module_path.parent.mkdir(parents=True)
            module_path.touch()

            with patch(
                "sparseengine.utils.code_revision._git_value",
                return_value=str(source_root),
            ):
                repo_root = _verified_source_repo_root(module_path)

        self.assertEqual(repo_root, source_root)


if __name__ == "__main__":
    unittest.main()
