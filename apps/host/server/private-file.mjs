/** Owner-only files. POSIX: mode 0600. Windows: a protected ACL naming only the current user (NTFS has no mode bits). */
import { chmodSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join } from 'node:path';

// Name the Windows tools by path: Git for Windows puts a different whoami first on PATH.
const system32 = name => join(process.env.SystemRoot || 'C:\\Windows', 'System32', name);

let userSid = null;
/** The current user's SID, read once with whoami (language independent). */
export function currentUserSid() {
  if (userSid) return userSid;
  const out = execFileSync(system32('whoami.exe'), ['/user', '/fo', 'csv', '/nh'], { encoding: 'utf8', timeout: 10000, windowsHide: true });
  const match = /S-1-\d+(?:-\d+)+/.exec(out);
  if (!match) throw new Error('Could not read the current Windows user.');
  return (userSid = match[0]);
}

function icacls(path, args) {
  execFileSync(system32('icacls.exe'), [path, ...args], { stdio: 'ignore', timeout: 15000, windowsHide: true });
}

export function makePrivate(path, { platform = process.platform } = {}) {
  if (platform !== 'win32') { chmodSync(path, 0o600); return; }
  // Three fast calls, in this order.  In one call, /inheritance:r next to /grant:r left the
  // inherited entries standing as explicit copies of SYSTEM and Administrators (a runner run's
  // Get-Acl printed D:PAI(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;LA)).  Split: reset to the parent's
  // rules, strip every inherited entry (a protected, empty DACL for the moment), then give this
  // user full control.  The end state is one full-control entry, the user's.
  icacls(path, ['/reset']);
  try {
    icacls(path, ['/inheritance:r']);
    icacls(path, ['/grant:r', `*${currentUserSid()}:F`]);
  } catch (e) {
    try { icacls(path, ['/reset']); } catch { /* leave the file as the reset put it */ }
    throw e;
  }
  // The owner too, so a file an elevated run made is not left owned by Administrators (needs the right to; best effort).
  try { icacls(path, ['/setowner', `*${currentUserSid()}`]); } catch { /* the DACL above already opens it to the user */ }
}
