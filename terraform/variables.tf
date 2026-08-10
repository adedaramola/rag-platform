variable "aws_region" {
  description = "AWS region to deploy into"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name — used as a prefix on every resource"
  type        = string
  default     = "rag-platform"
}

variable "environment" {
  description = "Deployment environment (used in resource names and tags)"
  type        = string
  default     = "prod"
}

variable "key_pair_name" {
  description = "Name of an existing EC2 key pair for SSH access"
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR block permitted to SSH into the instances. Restrict to your IP in production."
  type        = string
  default     = "0.0.0.0/0"
}

variable "instance_type_weaviate" {
  description = "EC2 instance type for the Weaviate node"
  type        = string
  default     = "t3.medium" # 2 vCPU / 4 GB — minimum for HNSW index under moderate load
}

variable "instance_type_api" {
  description = "EC2 instance type for the FastAPI server"
  type        = string
  default     = "t3.small" # 2 vCPU / 2 GB — sufficient for cross-encoder + uvicorn
}

variable "weaviate_data_volume_size_gb" {
  description = "Size of the dedicated EBS data volume attached to the Weaviate instance (GB)"
  type        = number
  default     = 20
}

variable "weaviate_version" {
  description = "Weaviate Docker image version to run"
  type        = string
  default     = "1.27.0"
}

variable "anthropic_api_key" {
  description = "Anthropic API key — written to the instance .env at boot (RAG_PLATFORM_ANTHROPIC_API_KEY)"
  type        = string
  sensitive   = true
}

variable "openai_api_key" {
  description = "OpenAI API key — written to the instance .env at boot (RAG_PLATFORM_OPENAI_API_KEY)"
  type        = string
  sensitive   = true
}

variable "github_token" {
  description = "GitHub PAT with repo scope — used to clone the private repo at boot, then stripped from git remote"
  type        = string
  sensitive   = true
}

variable "github_repo" {
  description = "Full GitHub repo URL (without credentials)"
  type        = string
  default     = "https://github.com/adedaramola/rag-platform.git"
}
