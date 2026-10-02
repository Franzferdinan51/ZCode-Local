// ============================================================
// `zcode decide` — ask the SystemOne decision engine directly
// ============================================================
//
// Thin, zero-config CLI over the SystemOne shim's decide endpoint
// (POST /v1/systemone/decide), mirroring grok-local's `decide` command.
// Answers typed decision questions — choice, score, or noul (yes/no) —
// backed by the decider model. The shim URL resolves exactly like the
// rest of ZCode ($SYSTEMONE_SHIM_URL, else the bundled/local shim);
// the CLI entry point already auto-starts the bundled shim when nothing
// answers, so this command works out of the box with no setup.
//
// Fail-open: kill-switches (ZCODE_SYSTEMONE=0, ZCODE_SYSTEMONE_DECIDE=0),
// an unreachable shim, or an unparseable answer all produce a clear
// stderr message and a nonzero exit — never a hang, never a throw.
// No model IDs appear anywhere here: selection is by the decider's own
// labels only.

import { readFileSync } from "node:fs";
import { parseArgs } from "node:util";
import type { RunContext } from "@zcode/shared-types";
import {
  buildDecideChoiceRequest,
  decide,
  judgeBatch,
  permute,
  resolveSystemOneBatchEndpoint,
  resolveSystemOneDecideEndpoint,
  resolveSystemOnePermuteEndpoint,
  type SystemOneBatchResults,
  type SystemOneDecideAnswer,
  type SystemOneDecideRequest,
  type SystemOneDecideType,
  type SystemOnePermuteVerdict,
} from "@zcode/core";

const USAGE = `Usage:
  zcode-local decide --type choice --state <text> --instructions <text> --criteria <label>=<description> [--criteria ...] [--gold <label>] [--verify] [--json]
  zcode-local decide --type noul   --state <text> --instructions <text> [--gold yes|no] [--json]
  zcode-local decide --type score  --state <text> --instructions <text> --criteria 0=<desc> --criteria 1=<desc> ... [--gold <n>] [--json]
  zcode-local decide --batch <file> [--json]

Ask the SystemOne decision engine a typed question:

  choice  pick one of the labeled options (--criteria label=description, repeatable)
  noul    yes/no question (no criteria)
  score   rate on a scale (--criteria 0..n-1 descriptions, labels must be 0,1,2...)

Options:
  --state <text>          the decision state (what is being decided)
  --instructions <text>   how the decider should judge
  --criteria <k=v>        one option per flag (choice/score only)
  --gold <label>          known-correct label: recorded in the decision log
                          for SystemOne's calibration battery
  --verify                re-run a choice under 8 option orders and report
                          whether the winner is stable (choice only): the
                          Decide -> verify -> Act gate for high-stakes choices
  --batch <file>          judge a JSON array of /v1/systemone bodies
                          (1..32) in one call; per-item failures are reported
                          per item. Takes no single-question flags.
  --json                  print the raw answer as JSON
  --timeout-ms <n>        decide lookup timeout (default 8000)
  -h, --help              show this help

Examples:
  zcode-local decide --type choice --state "Pick tonight's deploy slot" \\
    --instructions "Prefer the slot with the least user impact" \\
    --criteria a="Deploy at 2am, low traffic" --criteria b="Deploy at 6pm, high traffic"
  zcode-local decide --type noul --state "Is the API healthy?" --instructions "Answer from the probe results"
  zcode-local decide --type score --state "Rate this plan" --instructions "0=unusable, 4=perfect" \\
    --criteria 0="unusable" --criteria 1="poor" --criteria 2="ok" --criteria 3="good" --criteria 4="perfect" --json

Decision records are appended to ~/.config/zcode/systemone-decision-records.jsonl
($ZCODE_SYSTEMONE_DECISION_LOG overrides, "0" disables).
`;

function fail(ctx: RunContext, message: string): number {
  ctx.stderr.write(`[zcode] decide: ${message}\n`);
  return 1;
}

function parseCriterionFlag(
  raw: string,
): { label: string; description: string } | { error: string } {
  const eq = raw.indexOf("=");
  if (eq <= 0)
    return { error: `bad --criteria ${JSON.stringify(raw)}: expected <label>=<description>` };
  const label = raw.slice(0, eq).trim();
  const description = raw.slice(eq + 1).trim();
  if (!label) return { error: `bad --criteria ${JSON.stringify(raw)}: empty label` };
  if (!description) return { error: `bad --criteria ${JSON.stringify(raw)}: empty description` };
  return { label, description };
}

function formatDistribution(answer: SystemOneDecideAnswer): string {
  const entries = Object.entries(answer.probabilities).sort((a, b) => b[1] - a[1]);
  const width = 24;
  return entries
    .map(([label, prob]) => {
      const bar = "█".repeat(Math.max(1, Math.round(prob * width)));
      const marker = label === answer.label ? " ←" : "";
      return `  ${label}: ${prob.toFixed(3)} ${bar}${marker}`;
    })
    .join("\n");
}

export async function runDecideCommand(ctx: RunContext): Promise<number> {
  const argv = ctx.argv.slice(1);
  let parsed: ReturnType<typeof parseDecideArgs>;
  try {
    parsed = parseDecideArgs(argv);
  } catch (error) {
    return fail(ctx, error instanceof Error ? error.message : String(error));
  }
  if (parsed.values.help) {
    ctx.stdout.write(USAGE);
    return 0;
  }
  if (parsed.values.batch !== undefined) {
    return runBatchCommand(ctx, parsed);
  }
  const type = parsed.values.type as SystemOneDecideType | undefined;
  if (type !== "choice" && type !== "score" && type !== "noul") {
    return fail(ctx, `--type must be one of choice|score|noul\n\n${USAGE}`);
  }
  const wantVerify = parsed.values.verify === true;
  if (wantVerify && type !== "choice") {
    return fail(ctx, "--verify needs --type choice: only choices have option orders to permute");
  }
  const state = (parsed.values.state as string | undefined) ?? "";
  const instructions = (parsed.values.instructions as string | undefined) ?? "";
  if (!state.trim()) return fail(ctx, "--state is required");
  if (!instructions.trim()) return fail(ctx, "--instructions is required");

  const rawCriteria = (parsed.values.criteria as string[] | undefined) ?? [];
  const criteria: Record<string, string> = {};
  for (const raw of rawCriteria) {
    const entry = parseCriterionFlag(raw);
    if ("error" in entry) return fail(ctx, entry.error);
    if (criteria[entry.label] !== undefined) {
      return fail(ctx, `duplicate --criteria label ${JSON.stringify(entry.label)}`);
    }
    criteria[entry.label] = entry.description;
  }
  const labels = Object.keys(criteria);
  if (type === "noul") {
    if (labels.length > 0) return fail(ctx, "--criteria is not used with --type noul");
  } else {
    if (labels.length === 0)
      return fail(ctx, `--type ${type} requires at least one --criteria <label>=<description>`);
  }
  if (type === "score") {
    for (let i = 0; i < labels.length; i += 1) {
      if (labels[i] !== String(i)) {
        return fail(
          ctx,
          `--type score needs labels 0..n-1 in order; got ${JSON.stringify(labels[i])} at position ${i}`,
        );
      }
    }
  }
  if (wantVerify && labels.length < 2) {
    return fail(ctx, "--verify needs at least two --criteria options to permute");
  }

  const gold = parsed.values.gold as string | undefined;
  if (gold !== undefined) {
    const valid = type === "noul" ? ["yes", "no"] : labels;
    if (!valid.includes(gold)) {
      return fail(ctx, `--gold ${JSON.stringify(gold)} is not one of: ${valid.join(", ")}`);
    }
  }

  const timeoutRaw = parsed.values["timeout-ms"] as string | undefined;
  let timeoutMs: number | undefined;
  if (timeoutRaw !== undefined) {
    const n = Number(timeoutRaw);
    if (!Number.isFinite(n) || n <= 0) return fail(ctx, `--timeout-ms must be a positive number`);
    timeoutMs = Math.floor(n);
  }

  let request: SystemOneDecideRequest;
  if (type === "choice") {
    request = buildDecideChoiceRequest(state, instructions, criteria);
  } else {
    request = {
      state: state.slice(0, 6000),
      instructions,
      ...(type === "score" ? { criteria } : {}),
      type,
    };
  }

  let answer: SystemOneDecideAnswer | undefined;
  try {
    answer = await decide(request, {
      endpoint: resolveSystemOneDecideEndpoint(),
      ...(timeoutMs !== undefined ? { timeoutMs } : {}),
      ...(gold !== undefined ? { goldLabel: gold } : {}),
    });
  } catch {
    answer = undefined;
  }
  if (!answer) {
    return fail(
      ctx,
      "no answer from the SystemOne decide engine (shim down or unreachable? " +
        "set $SYSTEMONE_SHIM_URL or start the bundled shim; ZCODE_SYSTEMONE=0 disables)",
    );
  }

  let verdict: SystemOnePermuteVerdict | undefined;
  if (wantVerify) {
    try {
      verdict = await permute(
        { state, instructions, criteria },
        {
          endpoint: resolveSystemOnePermuteEndpoint(),
          ...(timeoutMs !== undefined ? { timeoutMs } : {}),
        },
      );
    } catch {
      verdict = undefined;
    }
  }

  if (parsed.values.json) {
    ctx.stdout.write(
      wantVerify
        ? `${JSON.stringify({ decision: answer, verification: verdict ?? null }, null, 2)}\n`
        : `${JSON.stringify(answer, null, 2)}\n`,
    );
  } else {
    const latency = answer.latencyMs !== undefined ? ` · ${Math.round(answer.latencyMs)}ms` : "";
    const backend = answer.backend ? ` · ${answer.backend}` : "";
    ctx.stdout.write(
      `decide(${answer.type}) → ${JSON.stringify(answer.label)}  confidence ${answer.confidence.toFixed(3)}${latency}${backend}\n` +
        `${formatDistribution(answer)}\n`,
    );
    if (verdict) ctx.stdout.write(`${formatVerification(verdict)}\n`);
  }
  if (wantVerify && !verdict) {
    return fail(
      ctx,
      "verification unavailable: the permute endpoint did not answer (the decision above still stands)",
    );
  }
  return 0;
}

function formatVerification(verdict: SystemOnePermuteVerdict): string {
  const first = verdict.runs[0]?.choice;
  const agree = verdict.runs.filter((r) => r.choice === first).length;
  const spread = (verdict.maxSpread * 100).toFixed(1);
  if (verdict.stable) {
    return `verification: STABLE — ${agree}/${verdict.runs.length} orders agree (max spread ${spread}%)`;
  }
  const lines = [
    `verification: UNSTABLE — the winner flips across option orders (max spread ${spread}%); do not act on this decision`,
  ];
  verdict.runs.forEach((run, i) => {
    const probs = Object.values(run.probabilities);
    const top = probs.length > 0 ? Math.max(0, ...probs) : 0;
    lines.push(
      `  order ${i + 1} [${run.order.join(", ")}] -> ${run.choice} (${Math.round(top * 100)}%)`,
    );
  });
  return lines.join("\n");
}

function formatBatch(results: SystemOneBatchResults): string {
  const n = results.results.length;
  const lines = [
    `batch: ${n} item${n === 1 ? "" : "s"} judged${results.model ? ` (model: ${results.model})` : ""}`,
  ];
  results.results.forEach((item, i) => {
    if (item.status === 200) {
      lines.push(`item ${i}: ok`);
      const payload = item.payload as Record<string, unknown> | null;
      const answers = payload?.["answers"];
      if (answers !== undefined && answers !== null) {
        try {
          lines.push(`  answers: ${JSON.stringify(answers)}`);
        } catch {
          // keep the ok line without the answers detail
        }
      }
    } else if (typeof item.error === "string") {
      lines.push(`item ${i}: status ${item.status}: ${item.error}`);
    } else {
      lines.push(`item ${i}: status ${item.status}`);
    }
  });
  return lines.join("\n");
}

type ParsedDecideArgs = ReturnType<typeof parseDecideArgs>;

async function runBatchCommand(ctx: RunContext, parsed: ParsedDecideArgs): Promise<number> {
  const values = parsed.values;
  if (values.type !== undefined) {
    return fail(
      ctx,
      "--batch takes no --type: each item in the file carries its own question types",
    );
  }
  if (values.state !== undefined) {
    return fail(ctx, "--batch takes no --state: each item in the file carries its own state");
  }
  if (values.instructions !== undefined) {
    return fail(
      ctx,
      "--batch takes no --instructions: each item in the file carries its own instructions",
    );
  }
  const criteria = values.criteria as string[] | undefined;
  if (criteria !== undefined && criteria.length > 0) {
    return fail(ctx, "--batch takes no --criteria: each item in the file carries its own criteria");
  }
  if (values.gold !== undefined) return fail(ctx, "--batch takes no --gold");
  if (values.verify === true) {
    return fail(
      ctx,
      "--verify needs a single --type choice question and cannot be combined with --batch",
    );
  }
  const path = values.batch as string;
  let items: unknown;
  try {
    items = JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return fail(ctx, `--batch cannot read ${JSON.stringify(path)}: not a readable JSON file`);
  }
  if (!Array.isArray(items)) {
    return fail(ctx, "--batch file must hold a JSON array of /v1/systemone bodies");
  }
  const timeoutRaw = values["timeout-ms"] as string | undefined;
  let timeoutMs: number | undefined;
  if (timeoutRaw !== undefined) {
    const n = Number(timeoutRaw);
    if (!Number.isFinite(n) || n <= 0) return fail(ctx, "--timeout-ms must be a positive number");
    timeoutMs = Math.floor(n);
  }

  let results: SystemOneBatchResults | undefined;
  try {
    results = await judgeBatch(items, {
      endpoint: resolveSystemOneBatchEndpoint(),
      ...(timeoutMs !== undefined ? { timeoutMs } : {}),
    });
  } catch {
    results = undefined;
  }
  if (!results) {
    return fail(
      ctx,
      "no answer from the SystemOne batch endpoint (shim down or unreachable? " +
        "set $SYSTEMONE_SHIM_URL or start the bundled shim; ZCODE_SYSTEMONE=0 disables)",
    );
  }
  if (values.json) {
    ctx.stdout.write(`${JSON.stringify(results, null, 2)}\n`);
    return 0;
  }
  ctx.stdout.write(`${formatBatch(results)}\n`);
  return 0;
}

function parseDecideArgs(args: string[]) {
  return parseArgs({
    allowPositionals: false,
    args,
    options: {
      type: { type: "string" },
      state: { type: "string" },
      instructions: { type: "string" },
      criteria: { type: "string", multiple: true },
      gold: { type: "string" },
      json: { type: "boolean" },
      verify: { type: "boolean" },
      batch: { type: "string" },
      "timeout-ms": { type: "string" },
      help: { type: "boolean", short: "h" },
    },
  });
}
