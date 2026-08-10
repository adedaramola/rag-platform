# RAG Platform — AWS Infrastructure

Provisions a production-ready two-tier deployment on AWS using Terraform.

## Infrastructure

```
Internet
   │
   ▼
ALB (rag-platform-prod-alb)
   ├── :80   → FastAPI  (port 8000)
   └── :8501 → Streamlit UI (port 8501)
   │
   ▼
EC2 t3.small (rag-platform-prod-api)
   ├── rag-platform-api.service  (uvicorn, 2 workers)
   └── rag-platform-ui.service   (streamlit)
   │
   ▼ (private network)
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

## Deploy

```bash
cd terraform/

# 1. Copy and populate the variables file
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars — fill in key_pair_name and all secrets

# 2. Initialise providers
terraform init

# 3. Preview the plan
terraform plan

# 4. Apply
terraform apply
```

Terraform prints the live URLs on completion:

```
api_endpoint             = "http://<alb-dns>/query"
api_docs_url             = "http://<alb-dns>/docs"
ui_url                   = "http://<alb-dns>:8501"
ssh_api                  = "ssh -i ~/.ssh/<key>.pem ec2-user@<ip>"
ssh_weaviate             = "ssh -i ~/.ssh/<key>.pem ec2-user@<ip>"
documents_bucket_name    = "rag-platform-prod-docs-<suffix>"
```

## Ingest documents

```bash
# 1. Upload to S3
aws s3 cp my-doc.pdf s3://$(terraform output -raw documents_bucket_name)/raw/

# 2. SSH into the API instance
$(terraform output -raw ssh_api)

# 3. Download and ingest
sudo aws s3 cp s3://$(terraform output -raw documents_bucket_name)/raw/my-doc.pdf \
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
| `vpc.tf` | VPC, subnets (public + private), routing, NAT gateway |
| `security_groups.tf` | ALB, API, and Weaviate security group rules |
| `alb.tf` | Application Load Balancer + target groups |
| `ec2.tf` | API and Weaviate instances, EBS data volume, user-data scripts |
| `iam.tf` | Instance profile and S3 read policy for the API instance |
| `s3.tf` | Documents bucket with versioning enabled |
| `variables.tf` | All input variables with defaults and descriptions |
| `outputs.tf` | URLs, IPs, SSH commands, bucket name |
| `terraform.tfvars.example` | Template — copy to `terraform.tfvars` and fill in secrets |

## Notes

**SSH access** — `allowed_ssh_cidr` defaults to `0.0.0.0/0` in the example.
Restrict it to your own IP in production:

```bash
# Find your IP
curl https://checkip.amazonaws.com
```

**Remote state** — for team or CI use, uncomment the `backend "s3"` block in
`main.tf` and point it at a state bucket. Never share `terraform.tfstate`
directly.

**Weaviate is on a private subnet** — it is not publicly reachable. The API
instance communicates with it over the VPC private network. SSH via the API
instance if direct access is needed.

## Tear down

```bash
terraform destroy
```

This removes all resources including the EBS data volume. Back up your Weaviate
data before destroying if you want to retain it.
