import { useState } from "react";
import type { MlRouteBackendId } from "@zcode/shared/systemone-scorer";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select.js";
import { Switch } from "@/components/ui/switch.js";
import { Input } from "@/components/ui/input.js";
import { SettingsGroupCard, SettingsRow } from "@/settings/SettingsPageParts.js";
import {
  readMlRoutePreference,
  writeMlRoutePreference,
  type MlRoutePreference,
} from "@/lib/mlRoutePreference.js";

const BACKEND_OPTIONS: Array<{ value: MlRouteBackendId; label: string; hint: string }> = [
  {
    value: "laya",
    label: "Laya (local)",
    hint: "Spawns a persistent python3 bridge (pip install laya). ~0.6GB weights, fastest CPU option.",
  },
  {
    value: "jeff-1",
    label: "Jeff-1 (local server)",
    hint: "POSTs to your legacy Jeff-1 jev_clf_server on 127.0.0.1:8079. 4B model, needs MPS/CUDA + ~8GB RAM. (The modern :8079 decision sidecar is decide-only and never answers route-choice.)",
  },
  {
    value: "systemone",
    label: "SystemOne (local server)",
    hint: "POSTs to your SystemOne shim on 127.0.0.1:8765. Smallest weights (32M+), pure local.",
  },
  {
    value: "custom",
    label: "Custom endpoint",
    hint: "POSTs the SystemOne shape to the endpoint below. Loopback only.",
  },
];

function backendHint(backend: MlRouteBackendId): string {
  return BACKEND_OPTIONS.find((option) => option.value === backend)?.hint ?? "";
}

/**
 * Local ML routing card (Settings > General). Self-contained: prefs live in
 * localStorage, the host sidecar is only contacted per route request.
 * Default OFF — local ML runtimes cost GBs of RAM and slow down machines
 * that did not opt in knowingly.
 */
export function LocalMlRouteSetting() {
  const [preference, setPreference] = useState<MlRoutePreference>(() => readMlRoutePreference());

  const update = (next: MlRoutePreference): void => {
    setPreference(next);
    writeMlRoutePreference(next);
  };

  const showEndpoint = preference.backend !== "laya";
  const effectiveEndpoint =
    preference.endpoint.trim() ||
    (preference.backend === "jeff-1"
      ? "http://127.0.0.1:8079/v1/systemone (default)"
      : preference.backend === "systemone"
        ? "http://127.0.0.1:8765/v1/systemone (default)"
        : "required for custom backends");
  return (
    <SettingsGroupCard>
      <SettingsRow
        label="Local ML routing (experimental)"
        description="Re-rank new-chat model picks with an on-device decision model (Jev-compatible). Needs a Python ML runtime + downloaded weights; uses GBs of RAM and can slow this machine down. Only enable if you know what you are doing — failures always keep the instant heuristic pick."
        control={
          <Switch
            checked={preference.enabled}
            onCheckedChange={(enabled) => update({ ...preference, enabled })}
            aria-label="Local ML routing (experimental)"
          />
        }
      />
      {preference.enabled ? (
        <>
          <SettingsRow
            label="ML backend"
            description={backendHint(preference.backend)}
            control={
              <Select
                value={preference.backend}
                onValueChange={(value) =>
                  update({ ...preference, backend: value as MlRouteBackendId })
                }
              >
                <SelectTrigger size="lg" className="w-[260px] min-w-0 justify-between">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {BACKEND_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            }
          />
          {showEndpoint ? (
            <SettingsRow
              label="Endpoint override"
              description={`Loopback http(s) only. Empty = ${effectiveEndpoint}.`}
              control={
                <Input
                  value={preference.endpoint}
                  placeholder={effectiveEndpoint}
                  onChange={(event) => update({ ...preference, endpoint: event.target.value })}
                  className="w-[320px]"
                />
              }
            />
          ) : null}
        </>
      ) : null}
    </SettingsGroupCard>
  );
}
