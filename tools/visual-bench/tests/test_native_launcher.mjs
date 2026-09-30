import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, mkdir, copyFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

const launcher = new URL('../run-native.sh', import.meta.url);

test('native launcher sets the runtime opt-out before exec and preserves arguments and failure', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'navigate-native-launcher-'));
  try {
    await mkdir(join(directory, 'target', 'release'), { recursive: true });
    await copyFile(launcher, join(directory, 'run-native.sh'));
    await writeFile(join(directory, 'target', 'release', 'visual-bench'),
      '#!/bin/sh\nprintf "%s\\n" "$ORT_DISABLE_TELEMETRY" "$#" "$1" "$2"\nexit 23\n', { mode: 0o755 });
    for (const supplied of [undefined, '0', '1']) {
      const env = { ...process.env };
      if (supplied === undefined) delete env.ORT_DISABLE_TELEMETRY;
      else env.ORT_DISABLE_TELEMETRY = supplied;
      const result = spawnSync('sh', [join(directory, 'run-native.sh'), 'image with spaces.png', '--device=cpu'],
        { cwd: tmpdir(), env, encoding: 'utf8' });
      assert.equal(result.error, undefined);
      assert.equal(result.status, 23);
      assert.equal(result.stdout, '1\n2\nimage with spaces.png\n--device=cpu\n');
      assert.equal(result.stderr, '');
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
