// Tests for speedstack/systemone-shim-url.ts — node:test, no external deps.
//
// The shim base URL is env-overridable with a localhost default; route,
// decide, and the wizard all resolve through here.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  DEFAULT_SYSTEMONE_SHIM_URL,
  SYSTEMONE_SHIM_URL_ENV,
  isLoopbackShimUrl,
  isPlausibleShimUrl,
  resolveSystemOneShimUrl,
  systemOneShimEndpoint,
} from "./systemone-shim-url.js";

test("resolveSystemOneShimUrl: localhost default when unset", () => {
  assert.equal(resolveSystemOneShimUrl({}), DEFAULT_SYSTEMONE_SHIM_URL);
  assert.equal(DEFAULT_SYSTEMONE_SHIM_URL, "http://127.0.0.1:8765");
});

test("resolveSystemOneShimUrl: env override wins, trims junk", () => {
  assert.equal(
    resolveSystemOneShimUrl({ [SYSTEMONE_SHIM_URL_ENV]: "http://macmini:8765" }),
    "http://macmini:8765",
  );
  assert.equal(
    resolveSystemOneShimUrl({ [SYSTEMONE_SHIM_URL_ENV]: "http://macmini:8765///" }),
    "http://macmini:8765",
  );
  assert.equal(
    resolveSystemOneShimUrl({ [SYSTEMONE_SHIM_URL_ENV]: "   " }),
    DEFAULT_SYSTEMONE_SHIM_URL,
  );
});

test("systemOneShimEndpoint: joins paths onto the resolved base", () => {
  assert.equal(
    systemOneShimEndpoint("/v1/systemone/route", {}),
    "http://127.0.0.1:8765/v1/systemone/route",
  );
  assert.equal(
    systemOneShimEndpoint("/healthz", {
      [SYSTEMONE_SHIM_URL_ENV]: "http://macmini:8765",
    }),
    "http://macmini:8765/healthz",
  );
});

test("isPlausibleShimUrl: http(s) hosts pass, junk fails", () => {
  assert.equal(isPlausibleShimUrl("http://127.0.0.1:8765"), true);
  assert.equal(isPlausibleShimUrl("https://macmini:8765/"), true);
  assert.equal(isPlausibleShimUrl("not a url"), false);
  assert.equal(isPlausibleShimUrl(""), false);
  assert.equal(isPlausibleShimUrl("ftp://host/x"), false);
});

test("isLoopbackShimUrl: localhost forms are loopback, remotes are not", () => {
  assert.equal(isLoopbackShimUrl("http://127.0.0.1:8765"), true);
  assert.equal(isLoopbackShimUrl("http://localhost:8765/"), true);
  assert.equal(isLoopbackShimUrl("http://[::1]:8765"), true);
  assert.equal(isLoopbackShimUrl("http://192.168.1.50:8765"), false);
  assert.equal(isLoopbackShimUrl("http://macmini:8765"), false);
  assert.equal(isLoopbackShimUrl("https://example.com:8765"), false);
  assert.equal(isLoopbackShimUrl("not a url"), false);
});
