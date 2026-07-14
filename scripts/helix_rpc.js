#!/usr/bin/env node
'use strict';
/**
 * Cosmic CLI ↔ T2Helix RPC
 * stdin: JSON { "op": "...", ...args }
 * stdout: JSON result
 *
 * Env:
 *   T2HELIX_ROOT   — path to t2helix checkout (default: ~/t2helix)
 *   T2HELIX_DATA_DIR — chronicle data dir (shared with Claude plugin when set)
 */

const fs = require('fs');
const path = require('path');
const os = require('os');

function resolveRoot() {
  if (process.env.T2HELIX_ROOT) return process.env.T2HELIX_ROOT;
  const home = process.env.HOME || os.homedir();
  const candidates = [
    path.join(home, 't2helix'),
    path.join(home, 'code', 't2helix'),
    path.join(home, 'src', 't2helix'),
  ];
  for (const c of candidates) {
    if (fs.existsSync(path.join(c, 'lib', 'grok-adapter.js'))) return c;
  }
  return path.join(home, 't2helix');
}

function resolveDataDir(home) {
  if (process.env.T2HELIX_DATA_DIR) return process.env.T2HELIX_DATA_DIR;
  const plugin = path.join(
    home,
    '.claude',
    'plugins',
    'data',
    't2helix-templetwo-t2helix'
  );
  if (fs.existsSync(plugin)) return plugin;
  return path.join(home, '.t2helix-data');
}

async function main() {
  const home = process.env.HOME || os.homedir();
  const root = resolveRoot();
  process.env.T2HELIX_DATA_DIR = resolveDataDir(home);

  let body = '';
  for await (const chunk of process.stdin) body += chunk;
  let req;
  try {
    req = JSON.parse(body || '{}');
  } catch (e) {
    console.log(JSON.stringify({ ok: false, error: 'invalid json: ' + e.message }));
    process.exit(2);
  }

  const op = req.op || process.argv[2];
  if (!op) {
    console.log(JSON.stringify({ ok: false, error: 'missing op' }));
    process.exit(2);
  }

  let g, chronicle;
  try {
    g = require(path.join(root, 'lib', 'grok-adapter.js'));
    chronicle = require(path.join(root, 'lib', 'chronicle.js'));
  } catch (e) {
    console.log(
      JSON.stringify({
        ok: false,
        error: 'cannot load t2helix from ' + root + ': ' + e.message,
        hint: 'clone https://github.com/templetwo/t2helix and set T2HELIX_ROOT',
      })
    );
    process.exit(3);
  }

  try {
    let result;
    switch (op) {
      case 'init':
        result = g.grokInit();
        break;
      case 'boot':
        result = await g.grokBoot({
          query: req.query || 'current context',
          topK: req.topK || 8,
        });
        break;
      case 'recall':
        result = g.grokRecall(req.query || '', {
          topK: req.topK || 6,
          layer: req.layer,
          include_meta: !!req.include_meta,
        });
        break;
      case 'record':
        result = g.grokRecord(req.content || '', {
          session_id: req.session_id,
          layer: req.layer || 'hypothesis',
          domain: req.domain || 'cosmic-cli',
          tags: req.tags || ['source:cosmic-cli'],
          intensity: req.intensity || 0.6,
        });
        break;
      case 'witness': {
        // Prefer structured tool actions so compass Bash-scoped rules match.
        // Plain strings are tagged tool_name:'Grok' and skip 8/9 rules
        // (Claude PAUSE experiment 2026-07-14).
        let action = req.action !== undefined ? req.action : req.query || '';
        if (req.tool_name) {
          action = {
            tool_name: req.tool_name,
            tool_input: req.tool_input || {},
          };
        }
        result = await g.grokWitness(action, {
          session_id: req.session_id,
          domain: req.domain || 'cosmic-cli',
          tags: req.tags || ['source:cosmic-cli'],
        });
        break;
      }
      case 'confirm_pending': {
        const token = req.token;
        if (!token) {
          result = { ok: false, error: 'token required' };
          break;
        }
        result = chronicle.approveConfirmation({ token });
        break;
      }
      case 'list_pending':
        result = chronicle.listPendingConfirmations({
          session_id: req.session_id,
          limit: req.limit || 20,
        });
        break;
      case 'set_goal':
        result = chronicle.setGoal({
          session_id: req.session_id || chronicle.readCurrentSession() || 'cosmic-cli',
          goal: req.goal || '',
          why: req.why || 'cosmic-cli mission',
          acceptance_criteria: req.acceptance_criteria,
        });
        break;
      case 'open_thread':
        result = chronicle.openThread({
          question: req.question || '',
          domain: req.domain || 'cosmic-cli',
          context: req.context || '',
        });
        break;
      case 'resolve_thread':
        result = chronicle.resolveThread({
          id: req.id,
          resolution: req.resolution || '',
        });
        break;
      case 'get_state':
        result = chronicle.getState(
          req.session_id || chronicle.readCurrentSession() || undefined
        );
        break;
      case 'health':
        result = {
          ok: true,
          root,
          dataDir: process.env.T2HELIX_DATA_DIR,
          health: typeof chronicle.health === 'function' ? chronicle.health() : null,
          session: chronicle.readCurrentSession && chronicle.readCurrentSession(),
        };
        break;
      default:
        result = { ok: false, error: 'unknown op: ' + op };
        process.exitCode = 2;
    }
    console.log(JSON.stringify({ ok: true, op, result }));
  } catch (e) {
    console.log(JSON.stringify({ ok: false, op, error: e.message, stack: e.stack }));
    process.exit(1);
  }
}

main();
