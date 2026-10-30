"""Known pipeline/security regressions must not silently return."""

import tomllib
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def version(value):
    return tuple(int(n) for n in value.split("."))


class DesktopUpdater(unittest.TestCase):
    def test_updater_can_resolve_a_zip_version_with_the_reported_security_fix(self):
        manifest = tomllib.loads((ROOT / "apps/desktop/src-tauri/Cargo.toml").read_text(encoding="utf-8"))
        self.assertGreaterEqual(version(manifest["dependencies"]["tauri-plugin-updater"]), (2, 14, 1))
        lock = tomllib.loads((ROOT / "apps/desktop/src-tauri/Cargo.lock").read_text(encoding="utf-8"))
        packages = lock["package"]
        updater = next(p for p in packages if p["name"] == "tauri-plugin-updater")
        self.assertGreaterEqual(version(updater["version"]), (2, 14, 1))
        zips = [p for p in packages if p["name"] == "zip"]
        self.assertTrue(zips)
        for package in zips:
            self.assertGreaterEqual(version(package["version"]), (9, 0, 5))

    def test_declared_rust_minimum_matches_the_updaters_existing_minimum(self):
        manifest = tomllib.loads((ROOT / "apps/desktop/src-tauri/Cargo.toml").read_text(encoding="utf-8"))
        self.assertGreaterEqual(version(manifest["package"]["rust-version"]), (1, 90))


class DependencyPRTitles(unittest.TestCase):
    def test_security_updates_keep_working_without_enabling_version_update_churn(self):
        path = ROOT / ".github/dependabot.yml"
        self.assertTrue(path.is_file(), "Dependabot titles need the same area prefix as human PRs")
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["version"], 2)
        updates = config["updates"]
        self.assertEqual({(u["package-ecosystem"], u["directory"]) for u in updates},
                         {("pip", "/"), ("npm", "/apps/host"), ("cargo", "/apps/desktop/src-tauri")})
        for update in updates:
            self.assertEqual(update["open-pull-requests-limit"], 0)
            self.assertEqual(update["commit-message"]["prefix"], "[Other] build(deps)")
            self.assertNotIn("ignore", update)


if __name__ == "__main__":
    unittest.main()
