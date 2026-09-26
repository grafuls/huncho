# API

`s1` exposes the Jev `/v1/systemone` wire contract plus operations endpoints.
It is byte-compatible with the public TypeSafe / Jev HTTP contract, so an
unmodified TypeSafe Python SDK works against `s1` with only a base-URL change.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/systemone` | Evaluate a state against typed questions. |
| `GET`  | `/health` | Liveness and version. |
| `GET`  | `/v1/models` | List registered models and their capabilities. |
| `GET`  | `/metrics` | Prometheus metrics. |

All responses are JSON.

## `POST /v1/systemone`

### Request

```json
{
  "state": "string | object | array",
  "model": "mock",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team handles this?",
      "criteria": { "returns": "Refund request", "billing": "Charging issue" }
    },
    "is_refund": {
      "type": "noul",
      "instructions": "Is the customer requesting a refund?",
      "criteria": { "true": "Requests money back", "false": "Not a refund" }
    },
    "severity": {
      "type": "score",
      "instructions": "Rate severity",
      "criteria": ["low", "mid", "high"]
    }
  }
}
```

- `state`: a string, object, or array (the Jev contract forbids null, booleans,
  and numbers here).
- `model`: the registered model name.
- `questions`: a map of `question-id -> question`. Answers come back under the
  same ids. Each question has a `type` of `choice`, `score`, or `noul` and an
  `instructions` field.

Question types:

- **choice**: `criteria` is a map of `label -> description`. Returns the chosen
  label, the full probability distribution, and a confidence.
- **score**: `criteria` is an ordered array of levels. Returns per-level
  probabilities, a probability-weighted position (can fall between levels), a
  legend mapping level index to description, and a confidence.
- **noul**: yes/no. Returns the probability that the answer is yes.

### Response

The response shape is strictly Jev-shaped by default:

```json
{
  "model": "mock",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "returns",
      "probabilities": { "billing": 0.4965, "returns": 0.5034 },
      "confidence": 0.0068
    },
    "is_refund": { "type": "noul", "noul": 0.4965 },
    "severity": {
      "type": "score",
      "probabilities": { "0": 0.3303, "1": 0.3348, "2": 0.3347 },
      "score": 1.0044,
      "legend": { "0": "low", "1": "mid", "2": "high" },
      "confidence": 0.0023
    }
  },
  "usage": { "input_tokens": 70, "output_tokens": 0 }
}
```

`s1` is prefill-only, so `output_tokens` is always `0`.

### Errors

| Status | Meaning |
|---|---|
| `400` | Malformed request (bad state/instructions, validation failure). |
| `401` | Missing or invalid bearer token (when auth is configured). |
| `422` | Unknown model name, or a package/backend validation failure. |
| `500` | Backend or calibration failure. |

Errors use a structured body:

```json
{ "error": { "code": "model_not_found", "message": "unknown model `nope`" } }
```

## Engine extensions (API-05)

Engine-specific extras are **off by default**, so default responses stay strictly
Jev-shaped. Request them with the `x-s1-extensions` header, or set
`default_extensions` in the server config:

```http
x-s1-extensions: true
```

When enabled, the response gains an `extensions` object:

```json
"extensions": {
  "backend": "onnx",
  "dtype": "fp32",
  "calibration_status": "fit",
  "confidence_definition": "peak",
  "prompt_contract_hash": "mock-hash",
  "raw_logits": { "department": [0.6279, 0.6416] }
}
```

`raw_logits` are the per-candidate, pre-softmax (pre-temperature) logits.
`confidence_definition` reports which confidence formula was applied (peak,
entropy, or a custom definition).

## Auth (API-03)

Configure a bearer token with the `--auth-token` serve flag
(see [operations.md](operations.md)). If set, every request must carry
`Authorization: Bearer <token>`; otherwise it returns `401`. Token comparison
is constant-time.

## Metrics (API-04, P1)

`GET /metrics` exposes Prometheus text metrics: request latency (p50/p95/p99),
counts and token usage per model, queue depth, and fork counts.
