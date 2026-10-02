import type {
  IntegratedTerminalShellOption,
  IntranetProbeRequest,
  IntranetProbeResult,
  SystemInfo,
} from "@zcode/shared";
import { ServiceChannels } from "@zcode/shared";
import type { MlRouteServiceRequest, MlRouteServiceResponse } from "@zcode/shared/systemone-scorer";
import { createServiceDescriptor } from "../descriptors.js";

export interface ISystemService {
  info(): Promise<SystemInfo>;
  listIntegratedTerminalShells(): Promise<IntegratedTerminalShellOption[]>;
  probeIntranet(request: IntranetProbeRequest): Promise<IntranetProbeResult>;
  /**
   * Resolve command names against PATH on the host running this service.
   * Returns the absolute executable path per command, or null when not found.
   * Used by the Harness Router tab to detect installed external harness CLIs.
   */
  resolveCommands(request: { commands: string[] }): Promise<Record<string, string | null>>;
  /**
   * Re-rank route candidates via an opt-in local ML backend (Jeff-1 /
   * SystemOne / Laya sidecar; SystemOne on :8765 is primary). Hard
   * filters stay in TS; a null suggestion means the caller keeps its
   * heuristic result (`reason` explains why).
   */
  suggestMlRoute(request: MlRouteServiceRequest): Promise<MlRouteServiceResponse>;
}

export const ISystemService = createServiceDescriptor<ISystemService>(ServiceChannels.System);
