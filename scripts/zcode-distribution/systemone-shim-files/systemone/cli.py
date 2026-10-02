"""systemone CLI — agent-friendly command line for a live SystemOne shim.

Two modes:

Shim-client commands (talk to the deployed shim over HTTP; they never load
a model themselves — the shim URL comes from $SYSTEMONE_SHIM_URL or
--shim-url, default http://127.0.0.1:8765):

    systemone route "summarize this quarter's revenue" [--json]
    systemone decide --type choice --state "..." --instructions "..." \\
        --criteria a="first option" --criteria b="second option" [--json]
    systemone status [--json]            # shim + sidecar health
    systemone battery [--mode live]      # regression battery vs the shim

Local commands (run against this machine's engine):

    systemone local [--load]            # engine config check (old `status`)
    systemone ask --state "..." --questions questions.json
    systemone serve [--port 8765]       # run the shim (foreground)

Every output command accepts --json for agent-consumable output; the
default is human-readable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .patterns import MAX_STATE_CHARS, MODEL_CANDIDATES, SystemOneError

try:
    from .api import SystemOne, default_device
except ImportError:  # slim install: shim-client + direct-SGLang commands only
    SystemOne = None  # type: ignore[assignment,misc]

    def default_device() -> str:
        return "cpu"

from .client import ShimError, SystemOneClient, default_shim_url

_LOCAL_EXTRA_HINT = "pip install 'systemone[local]' for the torch/GLiClass engine"


def _env_model() -> str:
    return (os.environ.get("SYSTEMONE_MODEL") or "").strip() or MODEL_CANDIDATES[0]


def _env_device() -> str:
    return (os.environ.get("SYSTEMONE_DEVICE") or "auto").strip()


def _make_client(args: argparse.Namespace) -> SystemOneClient:
    return SystemOneClient(base_url=args.shim_url, timeout=args.timeout)


# -- shim-client commands -------------------------------------------------


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def cmd_route(args: argparse.Namespace) -> int:
    """Tier routing for a task description via the live shim."""
    try:
        payload = _make_client(args).route(
            args.task, cost_bias=args.cost_bias, tiers=args.tiers
        )
    except ShimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    route = payload.get("route", {})
    if args.json:
        _print_json(route)
        return 0
    tier = route.get("tier", "?")
    conf = route.get("confidence")
    margin = route.get("margin")
    conf_s = f"{conf:.4f}" if isinstance(conf, (int, float)) else "n/a"
    margin_s = f"{margin:.4f}" if isinstance(margin, (int, float)) else "n/a"
    print(f"tier       : {tier}")
    print(f"confidence : {conf_s}  (margin {margin_s})")
    print(f"model      : {route.get('model_id', 'n/a')}")
    print(f"effort     : {route.get('effort', 'n/a')}")
    uncertain = route.get("uncertain")
    if uncertain is not None:
        print(f"uncertain  : {uncertain}")
    print(f"rationale  : {route.get('rationale', 'n/a')}")
    return 0


def _parse_criteria(args: argparse.Namespace) -> object:
    """Build the decide criteria from --criteria-json or --criteria pairs."""
    if args.criteria_json is not None:
        try:
            return json.loads(args.criteria_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"error: --criteria-json is not valid JSON: {exc}")
    pairs = []
    for raw in args.criteria or []:
        if "=" in raw:
            label, desc = raw.split("=", 1)
            pairs.append((label.strip(), desc.strip() or None))
        else:
            pairs.append((raw.strip(), None))
    pairs = [(lab, desc) for lab, desc in pairs if lab]
    if args.type == "score":
        # Score levels are positional on the shim: the ORDER of the
        # descriptions defines the levels, low -> high.
        return [desc if desc else lab for lab, desc in pairs] or None
    return {lab: desc for lab, desc in pairs} or None


def _fmt_prob_table(dist: dict) -> list:
    lines = []
    for label, prob in dist.items():
        try:
            bar = "#" * int(round(float(prob) * 20))
        except (TypeError, ValueError):
            bar = ""
        lines.append(f"    {label:<24} {float(prob):.4f} {bar}")
    return lines


def _decide_direct_sglang(args: argparse.Namespace) -> dict:
    """One typed decision straight from SGLang, no shim involved.

    Builds the api-style question from the decide flags, judges it with
    SGLangBackend (SGLANG_BASE_URL / SGLANG_MODEL), and maps the answer
    onto the decide payload shape the printer below expects.
    """
    from .sglang_backend import SGLangBackend, SGLangError

    criteria = _parse_criteria(args)
    if args.type == "choice":
        labels = list(criteria) if isinstance(criteria, dict) else []
        if len(labels) < 2:
            raise SystemExit(
                "error: --direct-sglang choice needs >= 2 --criteria labels"
            )
        question = {
            "name": "decision", "type": "choice", "options": labels,
            "prompt": args.instructions,
        }
    elif args.type == "score":
        levels = list(criteria) if isinstance(criteria, list) else []
        if len(levels) < 2:
            raise SystemExit(
                "error: --direct-sglang score needs >= 2 --criteria levels"
            )
        question = {
            "name": "decision", "type": "score", "levels": levels,
            "prompt": args.instructions,
        }
    else:
        question = {
            "name": "decision", "type": "noul",
            "statement": args.instructions,
        }
    try:
        out = SGLangBackend().systemone(args.state, [question])
    except SGLangError as exc:
        raise SystemExit(f"error: {exc}")
    ans, meta = out["decision"], out.get("_meta", {})
    decision: dict = {
        "type": args.type,
        "backend": "sglang-direct",
        "model": meta.get("model"),
        "latency_ms": meta.get("latency_ms"),
        "confidence": ans.get("confidence"),
        "label_mass": ans.get("label_mass"),
    }
    if args.type == "choice":
        decision.update({
            "label": ans["choice"], "probabilities": ans["probabilities"],
        })
    elif args.type == "score":
        decision.update({
            "level": ans["level"], "distribution": ans["distribution"],
            "score": ans.get("score"),
        })
    else:
        p = float(ans["probability"])
        decision.update({
            "label": "yes" if ans["answer"] else "no", "noul": p,
            "probabilities": {"yes": p, "no": 1.0 - p},
        })
    return decision


def cmd_decide(args: argparse.Namespace) -> int:
    """One typed decision via the live shim's decide endpoint."""
    if getattr(args, "direct_sglang", False):
        decision = _decide_direct_sglang(args)
    else:
        criteria = _parse_criteria(args)
        try:
            decision = _make_client(args).decide(
                args.state, args.instructions, criteria=criteria, type=args.type
            )
        except ShimError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    if args.json:
        _print_json(decision)
        return 0
    dtype = decision.get("type", args.type)
    conf = decision.get("confidence")
    conf_s = f"{conf:.4f}" if isinstance(conf, (int, float)) else "n/a"
    print(f"backend    : {decision.get('backend', 'n/a')}")
    if dtype == "choice":
        print(f"choice     : {decision.get('label', '?')}  (confidence {conf_s})")
        print("probabilities:")
        print("\n".join(_fmt_prob_table(decision.get("probabilities", {}) or {})))
    elif dtype == "score":
        print(f"level      : {decision.get('level', '?')}  (confidence {conf_s})")
        print("distribution:")
        print("\n".join(_fmt_prob_table(decision.get("distribution", {}) or {})))
    else:  # noul
        probs = decision.get("probabilities", {}) or {}
        p_yes = probs.get("yes", decision.get("noul"))
        p_s = f"{float(p_yes):.4f}" if isinstance(p_yes, (int, float)) else "n/a"
        print(f"answer     : {decision.get('label', '?')}  "
              f"(P(yes)={p_s}, confidence {conf_s})")
    latency = decision.get("latency_ms")
    if latency is not None:
        print(f"latency    : {latency} ms")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Shim + sidecar health: liveness, engine model, decision backend."""
    try:
        info = _make_client(args).status(probe=not args.no_probe)
    except ShimError as exc:  # status() is defensive; this is belt-and-braces
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print_json(info)
        return 0
    shim = info.get("shim") or {}
    print(f"shim       : {info.get('shim_url', '?')}")
    if shim.get("ok"):
        backend = shim.get("backend")
        backend_s = f", backend: {backend}" if backend else ""
        print(f"  ok       : yes  (engine model: {shim.get('model', 'n/a')}"
              f"{backend_s})")
    else:
        print(f"  ok       : NO — {shim.get('error', 'unknown error')}")
        return 1
    decision = info.get("decision")
    if decision is None:
        print("  decide   : probe skipped (--no-probe)")
    elif "error" in decision:
        print(f"  decide   : probe failed — {decision['error']}")
    else:
        print(f"  decide   : backend={decision.get('backend', 'n/a')}  "
              f"latency={decision.get('latency_ms', 'n/a')} ms  "
              f"confidence={decision.get('confidence', 'n/a')}")
    return 0


def cmd_jevbench(args: argparse.Namespace) -> int:
    """Score a JevBench jsonl split with a SystemOne engine."""
    from .jevbench import run_file
    from .shim import create_engine

    try:
        engine = create_engine(getattr(args, "engine", None))
    except Exception as exc:
        print(f"error: cannot build engine — {exc}", file=sys.stderr)
        return 1
    try:
        summary = run_file(args.items, engine, out=args.out, limit=args.limit)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.json:
        _print_json(summary)
        return 0
    acc = summary["accuracy"]
    acc_s = f" ({acc:.1%})" if acc is not None else ""
    print(f"jevbench: {summary['correct']}/{summary['scored']} correct{acc_s} "
          f"over {summary['n']} items [{summary['model']}]")
    if summary["mean_latency_ms"] is not None:
        print(f"  mean latency : {summary['mean_latency_ms']} ms")
    for fam, stats in sorted(summary["by_family"].items()):
        facc = stats["accuracy"]
        print(f"  {fam:<14}: {stats['correct']}/{stats['n']}"
              + (f" ({facc:.1%})" if facc is not None else ""))
    if summary.get("predictions"):
        print(f"  predictions  : {summary['predictions']}")
    return 0


def cmd_battery(args: argparse.Namespace) -> int:
    """Run the regression battery against the live shim (passthrough args)."""
    from .battery import run as battery_run

    passthrough = list(getattr(args, "passthrough", None) or [])
    saved = sys.argv
    sys.argv = ["systemone battery", *passthrough]
    try:
        return int(battery_run.main() or 0)
    finally:
        sys.argv = saved


# -- local commands -------------------------------------------------------


def cmd_local(args: argparse.Namespace) -> int:
    """Config/health check of the LOCAL engine (no shim involved).

    Only loads a model with --load (one at a time).
    """
    try:
        import torch
    except ImportError:
        print("systemone local")
        print(f"  device setting   : {_env_device()} (torch not installed)")
        print(f"  problem          : {_LOCAL_EXTRA_HINT}")
        return 1

    print("systemone local")
    print(f"  configured model : {_env_model()}")
    mps = getattr(torch.backends, "mps", None)
    mps_ok = bool(mps is not None and mps.is_available())
    print(f"  device setting   : {_env_device()} "
          f"(resolved: {default_device()}, "
          f"cuda: {torch.cuda.is_available()}, mps: {mps_ok})")
    print(f"  max state chars  : {MAX_STATE_CHARS}")

    cal_path = os.environ.get("SYSTEMONE_CALIBRATOR", "").strip()
    if cal_path:
        ok = os.path.exists(cal_path)
        print(f"  calibrator       : {cal_path} ({'found' if ok else 'MISSING'})")
    else:
        print("  calibrator       : none (raw softmax)")

    try:
        import mcp  # noqa: F401

        print(f"  mcp lib          : ok "
              f"({mcp.__version__ if hasattr(mcp, '__version__') else 'installed'})")
    except Exception as exc:
        print(f"  mcp lib          : PROBLEM ({exc})")

    if args.load:
        if SystemOne is None:
            print(f"  model load       : FAILED — {_LOCAL_EXTRA_HINT}")
            return 1
        try:
            eng = SystemOne()
        except SystemOneError as exc:
            print(f"  model load       : FAILED — {exc}")
            return 1
        print(f"  model load       : ok ({eng.model_name} on {eng.device})")
        # tiny latency probe, one batched pass
        out = eng.systemone(
            "status probe",
            [{"name": "ok", "type": "noul", "statement": "Is this a probe?"}],
        )
        print(f"  probe latency    : {out['_meta']['latency_ms']} ms "
              f"(P(yes)={out['ok']['probability']:.3f})")
    else:
        print("  model load       : skipped (use --load to verify)")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """One-shot typed judgments over a state, Jev-style (local engine)."""
    src = args.questions
    try:
        raw = sys.stdin.read() if src == "-" else open(src, encoding="utf-8").read()
    except OSError as exc:
        print(f"error: cannot read questions file: {exc.strerror or exc}", file=sys.stderr)
        return 2
    try:
        questions = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"error: questions file is not valid JSON: {exc}", file=sys.stderr)
        return 2
    if not isinstance(questions, list) or not questions:
        print("error: questions must be a non-empty JSON list", file=sys.stderr)
        return 2

    # Reuse the MCP server's Jev-compatible validation/conversion so the CLI
    # and the tool accept exactly the same question shape.
    try:
        from .mcp_server import _convert_questions, _to_jev_answers, get_engine
    except ImportError as exc:
        print(f"error: ask cannot start ({exc}) — {_LOCAL_EXTRA_HINT} "
              f"(plus 'systemone[mcp]' for MCP support)", file=sys.stderr)
        return 1

    try:
        converted, level_orders = _convert_questions(questions)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        engine = get_engine(model_name=args.model)
        out = engine.systemone(args.state, converted)
        result = _to_jev_answers(out, questions, level_orders)
    except SystemOneError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the shim; --with-jeff1 also starts the decision sidecar.

    Single-device story: one command brings up the GLiClass router shim on
    --port and the decision sidecar on --jeff1-port, both in the foreground.
    Ctrl-C stops the shim and terminates the sidecar subprocess.
    (Flag names are historical — the sidecar backend is decider-4b.)"""
    import subprocess

    procs = []
    try:
        if args.with_jeff1:
            env = dict(os.environ)
            env.setdefault("JEFF1_PORT", str(args.jeff1_port))
            proc = subprocess.Popen(
                [sys.executable, "-m", "systemone.jeff1_sidecar",
                 "--port", str(args.jeff1_port)],
                env=env,
            )
            procs.append(proc)
            print(f"decision sidecar starting on http://127.0.0.1:{args.jeff1_port} "
                  f"(pid {proc.pid}; model loads lazily on first request)")
        from .shim import serve as shim_serve

        server = shim_serve(args.port, engine_name=getattr(args, "engine", None))
        print(f"systemone shim on http://127.0.0.1:{args.port}/v1/systemone "
              f"and /v1/systemone/route (model {server.engine.model_name}, "
              f"backend {server.engine_backend})")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        for proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="systemone",
        description="SystemOne agent CLI — route tasks and make typed "
                    "decisions via a live shim (or drive the local engine).",
    )
    parser.add_argument(
        "--shim-url", default=None,
        help=f"shim base URL (default: $SYSTEMONE_SHIM_URL or {default_shim_url()})",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="HTTP timeout in seconds for shim calls (default 120)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="command")

    # -- shim-client commands --
    p_route = sub.add_parser(
        "route", help="route a task to the cheapest sufficient tier (via shim)")
    p_route.add_argument("task", help="task description to route")
    p_route.add_argument(
        "--cost-bias", default="balanced",
        choices=("economy", "balanced", "quality"),
        help="cost/capability bias (default balanced)")
    p_route.add_argument(
        "--tiers", default=None,
        help="comma-separated tier subset to route over, e.g. economy,balanced")
    p_route.add_argument("--json", action="store_true",
                         help="emit the raw route dict as JSON")
    p_route.set_defaults(func=cmd_route)

    p_decide = sub.add_parser(
        "decide", help="one typed decision: choice | noul | score (via shim)")
    p_decide.add_argument("--type", required=True,
                          choices=("choice", "noul", "score"),
                          help="decision type")
    p_decide.add_argument("--instructions", required=True,
                          help="the judgment to make")
    p_decide.add_argument("--state", required=True,
                          help="state text the decision is about")
    p_decide.add_argument(
        "--criteria", action="append", default=[],
        metavar="label=description",
        help="repeatable criterion. choice/noul: label=description pairs; "
             "score: descriptions in level order (label part ignored).")
    p_decide.add_argument(
        "--criteria-json", default=None, metavar="JSON",
        help='criteria as JSON (choice: {"label": "desc"}, '
              'score: ["level 1", "level 2"], '
              'noul: {"yes": "..", "no": ".."} or {"true": "..", "false": ".."})')
    p_decide.add_argument("--json", action="store_true",
                          help="emit the raw decide payload as JSON")
    p_decide.add_argument(
        "--direct-sglang", action="store_true",
        help="judge with SGLang directly (SGLANG_BASE_URL), no shim involved")
    p_decide.set_defaults(func=cmd_decide)

    p_status = sub.add_parser(
        "status", help="shim + sidecar health, decision backend in use")
    p_status.add_argument("--no-probe", action="store_true",
                          help="skip the tiny decide probe (shim liveness only)")
    p_status.add_argument("--json", action="store_true",
                          help="emit the status dict as JSON")
    p_status.set_defaults(func=cmd_status)

    p_jevbench = sub.add_parser(
        "jevbench", help="score a JevBench jsonl split with an engine")
    p_jevbench.add_argument("--items", required=True,
                            help="path to a JevBench-format .jsonl split")
    p_jevbench.add_argument("--out", default=None,
                            help="write per-item predictions as jsonl")
    p_jevbench.add_argument("--limit", type=int, default=None,
                            help="score at most N items")
    p_jevbench.add_argument("--engine", default=None,
                            choices=("auto", "local", "sglang", "jevk5", "onnx", "jev"),
                            help="engine (default: $SYSTEMONE_ENGINE or auto)")
    p_jevbench.add_argument("--json", action="store_true",
                            help="emit the raw summary as JSON")
    p_jevbench.set_defaults(func=cmd_jevbench)

    p_battery = sub.add_parser(
        "battery", help="regression battery vs the live shim (args pass through)")
    p_battery.add_argument("args", nargs=argparse.REMAINDER,
                           help="arguments forwarded to systemone.battery.run")
    p_battery.set_defaults(func=cmd_battery)

    # -- local commands --
    p_local = sub.add_parser(
        "local", help="local engine health check (no shim)")
    p_local.add_argument("--load", action="store_true",
                         help="also load the model and run a latency probe")
    p_local.set_defaults(func=cmd_local)

    p_ask = sub.add_parser("ask", help="one-shot typed judgments (local engine)")
    p_ask.add_argument("--state", required=True, help="state text to judge")
    p_ask.add_argument("--questions", required=True,
                       help='JSON file with [{"id","type","instructions","criteria"}] '
                            'or "-" for stdin')
    p_ask.add_argument("--model", default=None,
                       help="GLiClass checkpoint (default: env/smallest)")
    p_ask.set_defaults(func=cmd_ask)

    p_serve = sub.add_parser("serve", help="run the shim server (foreground)")
    p_serve.add_argument("--port", type=int, default=8765,
                         help="shim port (default 8765)")
    p_serve.add_argument("--with-jeff1", action="store_true",
                         help="also start the decision sidecar as a subprocess")
    p_serve.add_argument("--jeff1-port", type=int, default=8079,
                         help="decision sidecar port (default 8079)")
    p_serve.add_argument(
        "--engine", default=None,
        choices=("auto", "local", "sglang", "jevk5", "onnx", "jev"),
        help="decision engine to serve (default: $SYSTEMONE_ENGINE or auto)")
    p_serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    # Split the battery passthrough verbatim: everything after the `battery`
    # subcommand token goes straight to battery.run.main, in order, so the
    # battery's own flags never touch this parser.
    raw = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] | None = None
    for i, tok in enumerate(raw):
        if tok == "battery":
            passthrough = raw[i + 1:]
            if passthrough[:1] == ["--"]:
                passthrough = passthrough[1:]
            raw = raw[: i + 1]
            break
    args = parser.parse_args(raw)
    if args.cmd == "battery":
        args.passthrough = passthrough or []
    tiers = getattr(args, "tiers", None)
    if tiers is not None:  # normalize "a,b" -> ["a", "b"]
        args.tiers = [t.strip() for t in tiers.split(",") if t.strip()] or None
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
