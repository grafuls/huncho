# Quick task complete: Enforce bearer auth on every route (GitHub issue #1)

Date: 2026-10-09

`huncho_api::router(state)` now serves the API as one fallback service
wrapped by a `require_auth` middleware, so the configured bearer token is
checked once, before routing and before any request body is read, on every
path and method. `/health`, `/v1/models` and `/metrics` therefore return 401
like `/v1/systemone`, and rejections are identical for known and unknown
paths. Wrapping each route instead (`Router::layer`) would have leaked
axum's `Allow` header on denied requests, which enumerates routes. The
handler-level check in `systemone` was removed. Auth remains optional: with
no token every route behaves as before.

A 401 keeps the JSON `unauthorized` body, adds `WWW-Authenticate: Bearer`,
and adds `Connection: close` when the unread request body would otherwise
make hyper drop a keep-alive connection silently. `OPTIONS` requests are
answered by the outer CORS layer without a token, identically for every path.

`router()` now takes the state (public API change). All workspace callers
were updated; `serve()` layers CORS and tracing on top as before.

Docs: `docs/API.md`, `docs/operations.md` (probe and Prometheus scrape
examples), `packaging/rpm/huncho.1`, both env templates and `serve --help`.
CLI integration-test spawners no longer inherit `HUNCHO_AUTH_TOKEN`.

Verification:

- `cargo test -p huncho-api --offline`: all pass, including three new tests
  (every route 401s without/with wrong/schemeless/Basic credentials and
  succeeds with the token; denials precede routing and body parsing and are
  header-identical across paths; routes stay open without a token).
- Mutation checks: dropping the layer fails the new tests; per-route
  `Router::layer` fails the uniformity test.
- `cargo test -p huncho-cli --offline`: all pass; the feature-gated CLI tests
  compile with `clef,qualification` and `vllm`.
- Live checks against `huncho serve --mock --auth-token secret` covering
  methods, path forms, CORS preflight and keep-alive with large bodies.
- Two adversarial review rounds (5 + 4 lenses, 3 skeptics per finding); no
  bypass found in the second round.

Follow-ups deliberately left out of scope:

- `serve --help` prints the `HUNCHO_AUTH_TOKEN` value (clap
  `hide_env_values` is off).
- An empty token (`HUNCHO_AUTH_TOKEN=` or `--auth-token ""`) fails closed
  and locks out every route; reject it at startup.
- `check_auth` matches only `Bearer `/`bearer ` with exactly one space.
- `CorsLayer::permissive()` does not allow the `Authorization` header for
  credentialed cross-origin requests.
- No test of the composed `serve()` stack (CORS + trace around auth).

Changes are left in the working tree; no commit was created.
