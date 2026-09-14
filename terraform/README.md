# RAG Platform — AWS Infrastructure

Defines the EC2, storage, network, and HTTPS resources for the RAG portfolio deployment.
For the setup and deployment sequence, use [IMPLEMENTATION_GUIDE.md](../IMPLEMENTATION_GUIDE.md#rag-08--validate-the-terraform-package).
This document covers the resource layout and host operations.

## Infrastructure

```
Internet
   │
   ▼
ALB (rag-platform-prod-alb)
   ├── :80   → HTTPS redirect
   ├── :443  → FastAPI  (port 8000)
   └── :8501 → Streamlit UI (port 8501)
   │
   ▼
EC2 t3.small (rag-platform-prod-api)
   ├── rag-platform-api.service  (uvicorn, configurable workers; default 1)
   └── rag-platform-ui.service   (streamlit)
   │
   ▼ (private VPC traffic)
EC2 t3.medium (rag-platform-prod-weaviate)
   └── Weaviate 1.27.0 (Docker, 20 GB EBS data volume)

S3 (rag-platform-prod-docs-*)
   └── Raw document storage
```

## Prerequisites

- [Terraform](https://developer.hashicorp.com/terraform/install) >= 1.7
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html) configured (`aws configure`)
- An EC2 key pair created in your target region
- Anthropic API key, OpenAI API key, GitHub PAT (repo scope)
- An existing public Route 53 zone for the HTTPS API hostname

## Deployment and integration outputs

Use [RAG-08 through RAG-10](../IMPLEMENTATION_GUIDE.md#rag-08--validate-the-terraform-package)
for configuration validation, a reviewed deployment plan, and service verification.

The OpsDesk integration uses `search_base_url`, `approved_source_ids`,
`opsdesk_agent_api_key_secret_arn`, and `alb_arn_suffix`. The credential output is a
Secrets Manager ARN. Grant the Agent role access to that specific secret without
copying its value into an overlay.

## Ingest documents

Start in the local `terraform` directory with the deployed state initialized. Steps 1–2
run locally; steps 3–4 run on the API host. Exit the remote session before running
further Terraform commands.

```bash
# 1. Upload to S3
DOCUMENTS_BUCKET=$(terraform output -raw documents_bucket_name)
aws s3 cp my-doc.pdf "s3://$DOCUMENTS_BUCKET/raw/"

# 2. SSH into the API instance
$(terraform output -raw ssh_api)

# 3. On the API host, use the bucket name returned in step 1.
DOCUMENTS_BUCKET="<bucket-name-from-step-1>"
sudo aws s3 cp "s3://$DOCUMENTS_BUCKET/raw/my-doc.pdf" \
  /opt/rag-platform/data/raw/my-doc.pdf
sudo chown rag-platform:rag-platform /opt/rag-platform/data/raw/my-doc.pdf
sudo -u rag-platform bash -c \
  'cd /opt/rag-platform && .venv/bin/rag-platform-ingest --source /opt/rag-platform/data/raw/my-doc.pdf'

# 4. Restart services to reload the BM25 corpus
sudo systemctl restart rag-platform-api rag-platform-ui
```

## File overview

| File | Purpose |
|---|---|
| `main.tf` | Provider config, AMI data source, optional S3 backend |
| `vpc.tf` | VPC, two public subnets, internet gateway, and routing |
| `security_groups.tf` | ALB, API, and Weaviate security group rules |
| `alb.tf` | Application Load Balancer + target groups |
| `dns.tf` | ACM certificate validation and Route 53 alias |
| `ec2.tf` | API and Weaviate instances, EBS data volume, user-data scripts |
| `iam.tf` | Instance profile and S3 read/write policy for the EC2 instances |
| `s3.tf` | Documents bucket with versioning enabled |
| `secrets.tf` | Generated scoped OpsDesk Agent credential in Secrets Manager |
| `variables.tf` | All input variables with defaults and descriptions |
| `outputs.tf` | URLs, IPs, SSH commands, bucket name |
| `terraform.tfvars.example` | Template — copy to `terraform.tfvars` and fill in secrets |

## Notes

**SSH access** — the `allowed_ssh_cidr` variable defaults to `0.0.0.0/0`.
Restrict it to your own IP in production:

```bash
# Find your IP
curl https://checkip.amazonaws.com
```

**Remote state** — for team or CI use, uncomment the `backend "s3"` block in
`main.tf` and point it at a state bucket. Never share `terraform.tfstate`
directly.

**Weaviate network access** — both EC2 instances are in public subnets and
receive public IP addresses. The API communicates with Weaviate over its private
VPC address. Ports 8080 and 50051 accept traffic only from the API security
group; SSH is limited by `allowed_ssh_cidr`.

## Tear down

Back up the Weaviate data volume before teardown if its contents must survive.
The documents bucket uses `force_destroy = false`, so Terraform will not delete
a non-empty bucket or its versioned objects.

To destroy everything, empty every object version and delete marker from the
documents bucket first, then run:

```bash
terraform destroy
```

To retain the documents bucket and all of its settings, first back up the
Terraform state securely, then detach all S3-related resources from Terraform
management:

```bash
terraform state rm \
  aws_s3_bucket.documents \
  aws_s3_bucket_versioning.documents \
  aws_s3_bucket_server_side_encryption_configuration.documents \
  aws_s3_bucket_public_access_block.documents \
  random_id.bucket_suffix

terraform destroy
```

The retained bucket then exists independently of this Terraform configuration.
Do not run `terraform apply` again without first importing it or changing the
configuration, because Terraform will plan a replacement documents bucket.
