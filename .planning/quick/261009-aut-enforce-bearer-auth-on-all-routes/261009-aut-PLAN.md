# Quick task: Enforce bearer auth on every route (GitHub issue #1)

Date: 2026-10-09

Bearer auth (API-03) is only checked inside the `/v1/systemone` handler, and
only after its JSON body has been extracted. `/health`, `/v1/models` and
`/metrics` are reachable without a token even when `auth_token` is set, which
contradicts `docs/API.md` ("every request must carry ...") and is reachable
in production (`HUNCHO_BIND=0.0.0.0`).

1. Enforce `check_auth` once, as a router-level middleware layered after all
   routes so it also covers method-not-allowed and unknown-path fallbacks.
   `router()` takes the shared state and returns a ready router, so no public
   constructor can produce an unauthenticated app. Remove the now-redundant
   handler check; authentication runs before any body is read.
2. Keep auth optional: with no token configured every route behaves as before.
   Keep the existing JSON `unauthorized` error body and add the RFC 9110/6750
   `WWW-Authenticate: Bearer` challenge on 401.
3. Add API tests asserting 401 for `/health`, `/v1/models` and `/metrics` (and
   the other bypass shapes) without/with a wrong token, success with the right
   token, and unchanged unauthenticated behaviour when no token is set.
4. Update API/operations docs and the man page: health probes and Prometheus
   scrapes must send the token when one is configured.
5. Run the `huncho-api` and `huncho-cli` test suites and clippy.
