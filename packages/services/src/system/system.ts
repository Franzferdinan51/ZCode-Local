import type {
  IntegratedTerminalShellOption,
  IntranetProbeRequest,
  IntranetProbeResult,
  SystemInfo,
} from "@zcode/shared";
import { ServiceChannels } from "@zcode/shared";
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
}

export const ISystemService = createServiceDescriptor<ISystemService>(ServiceChannels.System);
