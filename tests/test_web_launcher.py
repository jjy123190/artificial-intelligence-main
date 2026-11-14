"""Windows website entry scripts must parse before any runtime setup."""
from __future__ import annotations

import codecs
import os
from pathlib import Path
import subprocess
from unittest import TestCase, skipUnless

from scripts.export_source import ROOT, source_files


def powershell_sources() -> list[Path]:
    return [path for path in source_files() if path.suffix.lower() == ".ps1"]


class WindowsWebsiteLauncherTests(TestCase):
    def test_non_ascii_powershell_sources_have_utf8_bom(self):
        # Windows PowerShell 5.1 otherwise reads UTF-8 as the system ANSI codepage.
        # Setting chcp or Console.OutputEncoding cannot fix -File source decoding.
        files = powershell_sources()
        self.assertIn(ROOT / "start_web.ps1", files)
        for path in files:
            with self.subTest(script=str(path.relative_to(ROOT))):
                raw = path.read_bytes()
                text = raw.decode("utf-8-sig")
                if not text.isascii():
                    self.assertTrue(raw.startswith(codecs.BOM_UTF8),
                                    "Non-ASCII PowerShell source requires a UTF-8 BOM")

    @skipUnless(os.name == "nt", "Requires the actual Windows PowerShell launcher")
    def test_sources_parse_in_windows_powershell_5_1(self):
        executable = (Path(os.environ.get("SystemRoot", r"C:\Windows"))
                      / "System32/WindowsPowerShell/v1.0/powershell.exe")
        self.assertTrue(executable.is_file(), str(executable))
        quoted_paths = ",".join("'" + str(path).replace("'", "''") + "'"
                                for path in powershell_sources())
        command = (
            "[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false); "
            "$failed = @(); foreach ($path in @(" + quoted_paths + ")) { "
            "$tokens = $null; $errors = $null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            "$path, [ref]$tokens, [ref]$errors) | Out-Null; "
            "foreach ($error in $errors) { $failed += [pscustomobject]@{"
            "file=$path; line=$error.Extent.StartLineNumber; message=$error.Message} } }; "
            "if ($failed.Count) { $failed | ConvertTo-Json -Compress; exit 1 }; exit 0"
        )
        result = subprocess.run(
            [str(executable), "-NoLogo", "-NoProfile", "-NonInteractive",
             "-ExecutionPolicy", "Bypass", "-Command", command],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
