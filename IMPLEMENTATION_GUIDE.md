# RAG Platform setup, deployment, and validation

This is the setup and deployment procedure for RAG Platform. For the connected OpsDesk
deployment order, start with the [portfolio guide](https://github.com/adedaramola/ops-platform/blob/main/PORTFOLIO_IMPLEMENTATION_GUIDE.md).
Run the local steps in order; use a fresh terminal from the parent workspace.

## RAG-01 — Enter and inspect the repository

```bash
cd rag-platform
git status --short --branch
git rev-parse HEAD
python3.11 --version
```

Expected result: the supplied branch/revision is checked out and Python is at least 3.11. Stop for
unexpected worktree changes.

## RAG-02 — Create the development environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -c "import rag; print('rag import OK')"
```

Do not create `.env` until deterministic tests have passed. This prevents accidental paid calls.

## RAG-03 — Run deterministic quality gates

```bash
make lint
make typecheck
make test-unit
pytest tests/integration/ -v -m "not weaviate"
```

Expected result: Ruff, strict mypy, unit tests, and embedded-Chroma integration tests pass. Provider,
HTTP, model, and store dependencies must be mocked or local. Record live-Weaviate and judge tests as
unexecuted unless their prerequisites are intentionally available.

## RAG-04 — Choose and configure a local embedding path

Copy the template:

```bash
cp .env.example .env
```

Choose one embedding backend:

1. OpenAI: leave `RAG_PLATFORM_EMBED_BACKEND=openai` and provide a valid local
   `RAG_PLATFORM_OPENAI_API_KEY`. This may incur charges.
2. Local BGE: install `python -m pip install -e ".[local-embed]"`, set
   `RAG_PLATFORM_EMBED_BACKEND=local`, and set the configured embedding dimensions to match the
   selected BGE model. This downloads a large model.

For standalone `/query`, provide a valid Anthropic key. For retrieval-only `/v1/search`, the
current settings still require the Anthropic field at startup, but the search route must not call
the generator.

Set a nonproduction service key and at least one approved source ID:

```dotenv
RAG_PLATFORM_API_KEY=<random-local-service-key>
RAG_PLATFORM_APPROVED_SOURCE_IDS=["opsdesk-demo-runbook"]
RAG_PLATFORM_STORE_BACKEND=chroma
RAG_PLATFORM_CACHE_BACKEND=none
```

Keep `.env` ignored. Never expose the embedded Chroma HTTP server; local Chroma must remain
in-process.

## RAG-05 — Ingest one approved document

Use a PDF, text file, DOCX file, URL, or directory that you are allowed to process. Its normalized
filename must match the configured approved source ID.

```bash
rag-platform-ingest --source /path/to/opsdesk-demo-runbook.pdf
```

Expected result: the command reports document/chunk counts without printing document text. Repeat
the command once and confirm re-ingestion is idempotent. Do not commit `.chroma`, downloaded models,
raw private documents, or reports containing content.

## RAG-06 — Start and verify the local API

Terminal 1:

```bash
source .venv/bin/activate
make api
```

Terminal 2:

```bash
curl -fsS http://localhost:8000/health

curl -i -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"VPN connection issue","source_ids":["opsdesk-demo-runbook"],"max_chunks":5}'

curl -fsS -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <local-service-key>" \
  -d '{"query":"VPN connection issue","source_ids":["opsdesk-demo-runbook"],"max_chunks":5}'
```

Expected results:

- health returns 200;
- the request without a key returns 403;
- authenticated search returns a `query_id` and no more than five chunks;
- citations are sequential `C1..Cn` and contain only the approved source;
- excerpts do not exceed the configured bound;
- logs contain lengths/counts, not the query or excerpts.

Send a request for an unapproved source and confirm 403. `/v1/search` must not make an LLM call.

## RAG-07 — Run optional evaluation deliberately

Retrieval benchmark:

```bash
make benchmark
```

Judge-backed evaluation requires Ollama or an enabled paid judge:

```bash
make ollama-pull
make eval
```

Record the backend, model, corpus revision, metrics, and number of runs. Do not compare results from
different corpora or embedding models as though they are the same experiment. Skip model downloads or judge execution when they are outside the intended validation scope.

## RAG-08 — Validate the Terraform package

```bash
cd terraform
terraform fmt -check -recursive
terraform init -backend=false -input=false
terraform validate
cp terraform.tfvars.example terraform.tfvars
```

Populate the ignored file with region, environment/domain, approved source IDs, instance settings,
and enabled provider credentials. Protect the state because current provider inputs may be copied
into Secrets Manager. Never commit `terraform.tfvars`, state, or a saved plan.

Expected result: formatting and validation pass before any AWS plan is created.

## RAG-09 — Plan and deploy only after authorization

Continue from the `terraform` directory used in RAG-08. Configure any remote backend
before initialization; see the [state notes](terraform/README.md#notes).

```bash
aws sts get-caller-identity
terraform init
terraform plan -out=rag.tfplan
```

Verify the AWS account/region and review resource count, ingress rules, DNS/TLS, EC2 types, S3,
persistent Weaviate EBS, and destruction behavior. Apply the exact reviewed plan only after approval:

```bash
terraform apply rag.tfplan
terraform plan
```

Expected result: the second plan reports no changes. Record only these non-secret outputs for the
portfolio integration:

```bash
terraform output -raw search_base_url
terraform output -raw opsdesk_agent_api_key_secret_arn
terraform output -raw alb_arn_suffix
```

The second value must be an ARN. Never retrieve or copy its raw key into another repository.

## RAG-10 — Ingest and verify the deployed service

1. Upload the approved document to the Terraform-created S3 bucket.
2. Follow the [host ingestion procedure](terraform/README.md#ingest-documents) to copy it to the API host.
3. Run `rag-platform-ingest` as the service user.
4. Restart the API/UI services if needed to reload persisted BM25 state.
5. Verify health, missing-auth 403, unapproved-source 403, and approved search 200.
6. Confirm the API reaches Weaviate over its private allowed path and data persists across API host
   restart/replacement.
7. Confirm logs and public evidence exclude keys, raw queries, excerpts, and document content.

Return to the portfolio guide with the search URL, scoped credential ARN, ALB ARN suffix, commit SHA,
and sanitized verification results.

For standalone question-answering, start the optional interface with `make ui` from the
repository root after configuring the generation provider. API usage examples and optional backends are
described in [README.md](README.md#query-via-python).

## RAG-11 — Stop or destroy only under a separate decision

Stopping an API host, replacing infrastructure, deleting S3 objects, removing the Weaviate volume,
or running `terraform destroy` are separate destructive/cost decisions. Before teardown, decide
which documents, index data, volume snapshots, logs, and state must be retained. Do not use teardown
as ordinary validation.
