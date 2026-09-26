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

import { parseArgs } from "node:util";
import type { RunContext } from "@zcode/shared-types";
import {
  buildDecideChoiceRequest,
  decide,
  resolveSystemOneDecideEndpoint,
  type SystemOneDecideAnswer,
  type SystemOneDecideRequest,
  type SystemOneDecideType,
} from "@zcode/core";

const USAGE = `Usage:
  zcode decide --type choice --state <text> --instructions <text> --criteria <label>=<description> [--criteria ...] [--gold <label>] [--json]
  zcode decide --type noul   --state <text> --instructions <text> [--gold yes|no] [--json]
  zcode decide --type score  --state <text> --instructions <text> --criteria 0=<desc> --criteria 1=<desc> ... [--gold <n>] [--json]

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
  --json                  print the raw answer as JSON
  --timeout-ms <n>        decide lookup timeout (default 8000)
  -h, --help              show this help

Examples:
  zcode decide --type choice --state "Pick tonight's deploy slot" \\
    --instructions "Prefer the slot with the least user impact" \\
    --criteria a="Deploy at 2am, low traffic" --criteria b="Deploy at 6pm, high traffic"
  zcode decide --type noul --state "Is the API healthy?" --instructions "Answer from the probe results"
  zcode decide --type score --state "Rate this plan" --instructions "0=unusable, 4=perfect" \\
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
  if (eq <= 0) return { error: `bad --criteria ${JSON.stringify(raw)}: expected <label>=<description>` };
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
  const type = parsed.values.type as SystemOneDecideType | undefined;
  if (type !== "choice" && type !== "score" && type !== "noul") {
    return fail(ctx, `--type must be one of choice|score|noul\n\n${USAGE}`);
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
    if (labels.length === 0) return fail(ctx, `--type ${type} requires at least one --criteria <label>=<description>`);
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

  if (parsed.values.json) {
    ctx.stdout.write(`${JSON.stringify(answer, null, 2)}\n`);
    return 0;
  }
  const latency = answer.latencyMs !== undefined ? ` · ${Math.round(answer.latencyMs)}ms` : "";
  const backend = answer.backend ? ` · ${answer.backend}` : "";
  ctx.stdout.write(
    `decide(${answer.type}) → ${JSON.stringify(answer.label)}  confidence ${answer.confidence.toFixed(3)}${latency}${backend}\n` +
      `${formatDistribution(answer)}\n`,
  );
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
      "timeout-ms": { type: "string" },
      help: { type: "boolean", short: "h" },
    },
  });
}
