/* oxlint-disable eslint(max-lines) -- 路由/Consent/自动路由共用同一 View 与 Notice 状态机；拆组件会让跨区 pending/notice 同步更脆弱。 */
import { useCallback, useEffect, useState } from "react";
import {
  CheckIcon,
  Loader2Icon,
  RotateCcwIcon,
  RouteIcon,
  ShieldCheckIcon,
  ShieldOffIcon,
  SparklesIcon,
  StarIcon,
  TerminalIcon,
} from "lucide-react";
import type { ModelSelection } from "@zcode/shared";
import { Button } from "@/components/ui/button.js";
import { useZCodeIntl } from "@/i18n/IntlProvider.js";
import { useServices } from "@/hooks/useServices.js";
import { useModelSelectionServiceView } from "@/hooks/useModelSelectionView.js";
import { V4_DRAFT_SCOPE_ROOT } from "@/v4/composer/composerDraftStore.js";
import { logger } from "@/logger.js";
import { cn } from "@/components/lib/utils.js";
import {
  buildHarnessProviderGroups,
  describeRouteTarget,
  resolveHarnessRouteOption,
} from "@/harness-router/harnessRouterModel.js";
import {
  EXTERNAL_HARNESSES,
  externalHarnessBinaries,
  harnessRouteTarget,
  resolveExternalHarnessStatus,
  type HarnessRouteTarget,
} from "@/harness-router/externalHarnesses.js";
import { requestSessionRoute } from "@/harness-router/sessionRouteRequests.js";

interface HarnessRouterSidePaneProps {
  workspacePath: string;
  workspaceIdentity?: string;
  enabled?: boolean;
  /** Open session in the main chat area (null = draft scope). v4: sessionId === taskId. */
  activeTaskId?: string | null;
  onOpenTerminalTabWithCommand?: (params: { command: string; title: string }) => void;
}

export function HarnessRouterSidePane({
  workspacePath,
  workspaceIdentity,
  enabled = true,
  activeTaskId = null,
  onOpenTerminalTabWithCommand,
}: HarnessRouterSidePaneProps) {
  const { intl } = useZCodeIntl();
  const { modelSelectionService, providerSettingsService, systemService } = useServices();
  const { state, reload } = useModelSelectionServiceView(modelSelectionService ?? null, enabled);
  const [pendingKey, setPendingKey] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [sessionNotice, setSessionNotice] = useState<string | null>(null);
  const [resolvedCommands, setResolvedCommands] = useState<Record<string, string | null> | null>(
    null,
  );

  useEffect(() => {
    if (!enabled || !systemService) {
      return;
    }
    let cancelled = false;
    void systemService
      .resolveCommands({ commands: externalHarnessBinaries() })
      .then((resolved) => {
        if (!cancelled) {
          setResolvedCommands(resolved);
        }
      })
      .catch((error) => {
        logger.error("[harness-router] resolve external harness commands failed", error);
      });
    return () => {
      cancelled = true;
    };
  }, [enabled, systemService]);

  const saveRoute = useCallback(
    async (selection: ModelSelection | undefined) => {
      if (!modelSelectionService) {
        return;
      }
      const key = describeRouteTarget(selection) ?? "automatic";
      setPendingKey(key);
      setSaveError(null);
      setSessionNotice(null);
      try {
        await modelSelectionService.saveDefaultModelSelection(selection);
        logger.info("[harness-router] default route saved", {
          route: key,
          workspacePath,
        });
      } catch (error) {
        logger.error("[harness-router] save default route failed", error);
        setSaveError(
          error instanceof Error
            ? error.message
            : intl.formatMessage({ id: "harnessRouter.saveError" }),
        );
      } finally {
        setPendingKey((current) => (current === key ? null : current));
      }
    },
    [intl, modelSelectionService, workspacePath],
  );

  const routeSession = useCallback(
    (selection: ModelSelection) => {
      const key = describeRouteTarget(selection) ?? "unknown";
      setSaveError(null);
      const handled = requestSessionRoute({
        scopeId: activeTaskId ?? V4_DRAFT_SCOPE_ROOT,
        selection,
      });
      if (handled) {
        logger.info("[harness-router] open session routed", { route: key, workspacePath });
        setSessionNotice(intl.formatMessage({ id: "harnessRouter.sessionRouted" }, { route: key }));
      } else {
        setSessionNotice(intl.formatMessage({ id: "harnessRouter.sessionNotOpen" }));
      }
    },
    [activeTaskId, intl, workspacePath],
  );

  const toggleHarnessConsent = useCallback(
    async (target: HarnessRouteTarget, harnessName: string, granted: boolean) => {
      if (!providerSettingsService) {
        return;
      }
      const key = `consent:${target.providerId}`;
      setPendingKey(key);
      setSaveError(null);
      setSessionNotice(null);
      try {
        await providerSettingsService.savePersonalProviderOverlay(target.providerId, {
          access: {
            type: "external-harness",
            driverId: target.driverId,
            consentGranted: granted,
          },
        });
        setSessionNotice(
          intl.formatMessage(
            { id: granted ? "harnessRouter.consentGranted" : "harnessRouter.consentRevoked" },
            { harness: harnessName },
          ),
        );
        await reload();
      } catch (error) {
        logger.error("[harness-router] save harness consent failed", error);
        setSaveError(
          error instanceof Error
            ? error.message
            : intl.formatMessage({ id: "harnessRouter.saveError" }),
        );
      } finally {
        setPendingKey((current) => (current === key ? null : current));
      }
    },
    [intl, providerSettingsService, reload],
  );

  const launchExternalHarness = useCallback(
    (harnessId: string) => {
      if (!onOpenTerminalTabWithCommand) {
        return;
      }
      const harness = EXTERNAL_HARNESSES.find((entry) => entry.id === harnessId);
      if (!harness || harness.builtin) {
        return;
      }
      setSaveError(null);
      onOpenTerminalTabWithCommand({
        command: harness.launchCommand,
        title: harness.name,
      });
      logger.info("[harness-router] external harness launched", {
        harness: harness.id,
        workspacePath,
      });
      setSessionNotice(
        intl.formatMessage({ id: "harnessRouter.externalLaunched" }, { harness: harness.name }),
      );
    },
    [intl, onOpenTerminalTabWithCommand, workspacePath],
  );

  if (state.status === "loading") {
    return (
      <div className="flex h-full items-center justify-center gap-2 text-ui-xs text-foreground-subtle">
        <Loader2Icon className="size-3.5 animate-spin" />
        {intl.formatMessage({ id: "harnessRouter.loading" })}
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center">
        <p className="text-ui-xs text-foreground-subtle">
          {intl.formatMessage({ id: "harnessRouter.loadError" })}
        </p>
        <Button size="sm" variant="outline" onClick={reload}>
          {intl.formatMessage({ id: "common.retry" })}
        </Button>
      </div>
    );
  }

  if (state.status === "unavailable") {
    return (
      <div className="flex h-full items-center justify-center px-6 text-center text-ui-xs text-foreground-subtle">
        {intl.formatMessage({ id: "harnessRouter.unavailable" })}
      </div>
    );
  }

  const groups = buildHarnessProviderGroups(state.view);
  const routed = state.view.preferredSelection;

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="shrink-0 border-b border-border px-3 py-2.5">
        <div className="flex items-center gap-1.5">
          <RouteIcon className="size-3.5 text-foreground-subtle" />
          <h2 className="text-ui-sm font-medium text-foreground">
            {intl.formatMessage({ id: "sidePane.harnessRouter" })}
          </h2>
        </div>
        <p className="pt-1 text-ui-xs text-foreground-subtle">
          {intl.formatMessage({ id: "harnessRouter.description" })}
        </p>
        <p className="pt-1 text-ui-xs text-foreground-subtle">
          {intl.formatMessage(
            { id: "harnessRouter.currentRoute" },
            {
              route: routed
                ? `${routed.providerId}/${routed.modelId}`
                : intl.formatMessage({ id: "harnessRouter.automatic" }),
            },
          )}
        </p>
        {sessionNotice && (
          <p className="pt-1 text-ui-xs text-foreground">{sessionNotice}</p>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-2">
        <button
          type="button"
          disabled={pendingKey !== null}
          onClick={() => void saveRoute(undefined)}
          className={cn(
            "flex w-full items-center gap-2 rounded-md border px-2.5 py-2 text-left text-ui-xs",
            "border-border hover:bg-accent disabled:opacity-60",
            !routed && "border-primary",
          )}
        >
          {pendingKey === "automatic" ? (
            <Loader2Icon className="size-3.5 animate-spin" />
          ) : (
            <RotateCcwIcon className="size-3.5 text-foreground-subtle" />
          )}
          <span className="min-w-0 flex-1">
            <span className="block font-medium text-foreground">
              {intl.formatMessage({ id: "harnessRouter.automatic" })}
            </span>
            <span className="block truncate text-foreground-subtle">
              {intl.formatMessage({ id: "harnessRouter.automaticHint" })}
            </span>
          </span>
          {!routed && <CheckIcon className="size-3.5 shrink-0 text-primary" />}
        </button>

        {groups.length === 0 ? (
          <p className="px-1 pt-3 text-ui-xs text-foreground-subtle">
            {intl.formatMessage({ id: "harnessRouter.noProviders" })}
          </p>
        ) : (
          groups.map((group) => (
            <section key={group.providerId} className="pt-3">
              <h3 className="px-1 pb-1.5 text-ui-xs font-medium text-foreground-subtle">
                {group.providerName}
              </h3>
              <ul className="flex flex-col gap-1">
                {group.models.map((model) => {
                  const key = `${model.providerId}/${model.modelId}`;
                  const pending = pendingKey === key;
                  return (
                    <li
                      key={model.modelId}
                      className={cn(
                        "flex items-center gap-1 rounded-md border border-border",
                        model.isRouted && "border-primary",
                      )}
                    >
                      <button
                        type="button"
                        disabled={!model.enabled || pendingKey !== null}
                        onClick={() =>
                          routeSession({ providerId: model.providerId, modelId: model.modelId })
                        }
                        title={`${intl.formatMessage({ id: "harnessRouter.routeSession" })}: ${key}`}
                        className={cn(
                          "flex min-w-0 flex-1 items-center gap-2 rounded-md px-2.5 py-1.5 text-left",
                          "hover:bg-accent disabled:cursor-not-allowed disabled:opacity-60",
                        )}
                      >
                        <span
                          className={cn(
                            "size-1.5 shrink-0 rounded-full",
                            model.isRouted ? "bg-primary" : "bg-border",
                          )}
                        />
                        <span className="min-w-0 flex-1 truncate font-mono text-ui-xs text-foreground">
                          {model.modelId}
                        </span>
                        {!model.enabled && (
                          <span className="shrink-0 text-ui-xs text-foreground-subtle">
                            {intl.formatMessage({ id: "harnessRouter.disabled" })}
                          </span>
                        )}
                        {model.isRouted && (
                          <CheckIcon className="size-3.5 shrink-0 text-primary" />
                        )}
                      </button>
                      <button
                        type="button"
                        disabled={!model.enabled || pendingKey !== null}
                        onClick={() =>
                          void saveRoute({ providerId: model.providerId, modelId: model.modelId })
                        }
                        title={intl.formatMessage({ id: "harnessRouter.setDefault" })}
                        aria-label={`${intl.formatMessage({ id: "harnessRouter.setDefault" })}: ${key}`}
                        className="shrink-0 rounded-md p-1.5 hover:bg-accent disabled:cursor-not-allowed disabled:opacity-60"
                      >
                        {pending ? (
                          <Loader2Icon className="size-3.5 animate-spin" />
                        ) : (
                          <StarIcon
                            className={cn(
                              "size-3.5",
                              model.isRouted ? "fill-current text-primary" : "text-foreground-subtle",
                            )}
                          />
                        )}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </section>
          ))
        )}

        <section className="pt-4">
          <h3 className="px-1 pb-1 text-ui-xs font-medium text-foreground-subtle">
            {intl.formatMessage({ id: "harnessRouter.externalTitle" })}
          </h3>
          <p className="px-1 pb-1.5 text-ui-xs text-foreground-subtle">
            {intl.formatMessage({ id: "harnessRouter.externalHint" })}
          </p>
          <ul className="flex flex-col gap-1">
            {EXTERNAL_HARNESSES.map((harness) => {
              const status = resolveExternalHarnessStatus(
                harness,
                resolvedCommands?.[harness.binary],
                resolvedCommands !== null,
              );
              const launchable =
                status === "installed" && Boolean(onOpenTerminalTabWithCommand);
              const target = harnessRouteTarget(harness.id);
              const option = target
                ? resolveHarnessRouteOption(state.view, target.providerId, target.modelId)
                : null;
              const routable = target !== null && option !== null && option.enabled;
              const routeKey = target ? `${target.providerId}/${target.modelId}` : harness.id;
              const consentPending = target !== null && pendingKey === `consent:${target.providerId}`;
              const statusLabel =
                status === "builtin"
                  ? intl.formatMessage({ id: "harnessRouter.externalBuiltin" })
                  : status === "installed"
                    ? intl.formatMessage({ id: "harnessRouter.externalInstalled" })
                    : status === "missing"
                      ? intl.formatMessage({ id: "harnessRouter.externalMissing" })
                      : intl.formatMessage({ id: "harnessRouter.externalUnknown" });
              return (
                <li
                  key={harness.id}
                  className={cn(
                    "flex items-center gap-1 rounded-md border border-border",
                    option?.isRouted && "border-primary",
                  )}
                >
                  <button
                    type="button"
                    disabled={routable ? status !== "installed" : !launchable}
                    onClick={() =>
                      routable && option
                        ? routeSession({ providerId: option.providerId, modelId: option.modelId })
                        : launchExternalHarness(harness.id)
                    }
                    title={
                      routable
                        ? `${intl.formatMessage({ id: "harnessRouter.routeSession" })}: ${routeKey}`
                        : `${intl.formatMessage({ id: "harnessRouter.externalLaunch" })}: ${harness.launchCommand}`
                    }
                    className={cn(
                      "flex min-w-0 flex-1 items-center gap-2 rounded-md px-2.5 py-1.5 text-left",
                      "hover:bg-accent disabled:cursor-not-allowed disabled:opacity-60",
                    )}
                  >
                    <span
                      className={cn(
                        "size-1.5 shrink-0 rounded-full",
                        status === "installed" || status === "builtin"
                          ? "bg-primary"
                          : "bg-border",
                      )}
                    />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-ui-xs font-medium text-foreground">
                        {harness.name}
                      </span>
                      <span className="block truncate text-ui-xs text-foreground-subtle">
                        {routable
                          ? intl.formatMessage({ id: "harnessRouter.routableHint" })
                          : harness.hint}
                      </span>
                    </span>
                    {option?.isRouted ? (
                      <span className="shrink-0 text-ui-xs font-medium text-primary">
                        {intl.formatMessage({ id: "harnessRouter.routedBadge" })}
                      </span>
                    ) : (
                      <span className="shrink-0 text-ui-xs text-foreground-subtle">
                        {statusLabel}
                      </span>
                    )}
                    {routable ? (
                      <RouteIcon className="size-3.5 shrink-0 text-primary" />
                    ) : (
                      launchable && <TerminalIcon className="size-3.5 shrink-0 text-primary" />
                    )}
                  </button>
                  {routable && target && option ? (
                    <button
                      type="button"
                      disabled={pendingKey !== null}
                      onClick={() =>
                        void toggleHarnessConsent(target, harness.name, !option.consentGranted)
                      }
                      title={intl.formatMessage({
                        id: option.consentGranted
                          ? "harnessRouter.consentRevoke"
                          : "harnessRouter.consentGrant",
                      })}
                      aria-label={`${intl.formatMessage({
                        id: option.consentGranted
                          ? "harnessRouter.consentRevoke"
                          : "harnessRouter.consentGrant",
                      })}: ${harness.name}`}
                      className="shrink-0 rounded-md p-1.5 hover:bg-accent disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {consentPending ? (
                        <Loader2Icon className="size-3.5 animate-spin" />
                      ) : option.consentGranted ? (
                        <ShieldCheckIcon className="size-3.5 text-primary" />
                      ) : (
                        <ShieldOffIcon className="size-3.5 text-foreground-subtle" />
                      )}
                    </button>
                  ) : null}
                  {routable && launchable ? (
                    <button
                      type="button"
                      onClick={() => launchExternalHarness(harness.id)}
                      title={`${intl.formatMessage({ id: "harnessRouter.externalLaunch" })}: ${harness.launchCommand}`}
                      aria-label={`${intl.formatMessage({ id: "harnessRouter.externalLaunch" })}: ${harness.name}`}
                      className="shrink-0 rounded-md p-1.5 hover:bg-accent"
                    >
                      <TerminalIcon className="size-3.5 text-foreground-subtle" />
                    </button>
                  ) : null}
                </li>
              );
            })}
          </ul>
          <p className="px-1 pt-2 text-ui-xs text-foreground-subtle">
            {intl.formatMessage({ id: "harnessRouter.apiHint" })}
          </p>
        </section>

        {saveError && (
          <p className="px-1 pt-3 text-ui-xs text-destructive">{saveError}</p>
        )}
      </div>

      <div className="shrink-0 border-t border-border px-3 py-2">
        <p className="text-ui-xs text-foreground-subtle">
          {intl.formatMessage({ id: "harnessRouter.footerHint" })}
        </p>
      </div>
    </div>
  );
}
