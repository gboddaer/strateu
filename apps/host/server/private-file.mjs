/** Owner-only files. POSIX: mode 0600. Windows: a protected ACL naming only the current user (NTFS has no mode bits). */
import { chmodSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';

// Name the Windows tools by path: Git for Windows puts a different whoami first on PATH.
const system32 = name => join(process.env.SystemRoot || 'C:\\Windows', 'System32', name);
const windowsPowershell = join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe');

let userSid = null;
/** The current user's SID, read once with whoami (language independent). */
export function currentUserSid() {
  if (userSid) return userSid;
  const out = execFileSync(system32('whoami.exe'), ['/user', '/fo', 'csv', '/nh'], { encoding: 'utf8', timeout: 10000, windowsHide: true });
  const match = /S-1-\d+(?:-\d+)+/.exec(out);
  if (!match) throw new Error('Could not read the current Windows user.');
  return (userSid = match[0]);
}

export function makePrivate(path, { platform = process.platform } = {}) {
  if (platform !== 'win32') { chmodSync(path, 0o600); return; }
  // Windows PowerShell cannot autoload its Security module (Get-Acl, Set-Acl live there) when the
  // user's PSModulePath points at PowerShell 7 folders, so the child starts with a clean module path.
  const env = { ...process.env }; delete env.PSModulePath;
  // Replace the whole DACL: protected (inherited entries gone, nothing copied), and exactly one
  // entry, this user.  icacls's /inheritance:r alone turns inherited entries into explicit copies
  // of Everyone, SYSTEM and Administrators on some machines, which is what a runner run showed.
  const quoted = path.replace(/'/g, "''");
  execFileSync(windowsPowershell, ['-NoProfile', '-NonInteractive', '-Command',
    `$a = Get-Acl -LiteralPath '${quoted}'; $a.SetAccessRuleProtection($true, $false); ` +
    `$a.SetAccessRule((New-Object System.Security.AccessControl.FileSystemAccessRule('${currentUserSid()}', 'FullControl', 'Allow'))); ` +
    `Set-Acl -LiteralPath '${quoted}' -Value $a`],
    { encoding: 'utf8', timeout: 20000, windowsHide: true, env });
  // The owner too, so a file an elevated run made is not left owned by Administrators (needs the right to; best effort).
  try { execFileSync(system32('icacls.exe'), [path, '/setowner', `*${currentUserSid()}`], { stdio: 'ignore', timeout: 15000, windowsHide: true }); } catch { /* the DACL above already opens it to the user */ }
}
