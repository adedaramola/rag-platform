# ---------------------------------------------------------------------------
# EC2 — Weaviate node + API node
#
# Both instances use the same IAM profile and AMI.
# Weaviate gets a dedicated data EBS volume so it survives instance replacement.
# The API instance has a 20 GB root volume to hold the Python venv and the
# cross-encoder model cache (~180 MB).
# ---------------------------------------------------------------------------

# ---- Weaviate ---------------------------------------------------------------

resource "aws_instance" "weaviate" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.instance_type_weaviate
  subnet_id              = aws_subnet.public[0].id
  vpc_security_group_ids = [aws_security_group.weaviate.id]
  key_name               = var.key_pair_name
  iam_instance_profile   = aws_iam_instance_profile.ec2.name

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 30 # AL2023 AMI requires >= 30 GB
    encrypted             = true
    delete_on_termination = true
  }

  user_data = base64encode(templatefile("${path.module}/user_data/weaviate.sh", {
    weaviate_version = var.weaviate_version
  }))

  tags = { Name = "${var.project_name}-${var.environment}-weaviate" }
}

# Dedicated data volume — separate lifecycle from the instance
resource "aws_ebs_volume" "weaviate_data" {
  availability_zone = aws_instance.weaviate.availability_zone
  size              = var.weaviate_data_volume_size_gb
  type              = "gp3"
  encrypted         = true

  tags = { Name = "${var.project_name}-${var.environment}-weaviate-data" }
}

resource "aws_volume_attachment" "weaviate_data" {
  device_name  = "/dev/xvdf"
  volume_id    = aws_ebs_volume.weaviate_data.id
  instance_id  = aws_instance.weaviate.id
  force_detach = false
}

# ---- API -------------------------------------------------------------------

resource "aws_instance" "api" {
  ami                    = data.aws_ami.amazon_linux_2023.id
  instance_type          = var.instance_type_api
  subnet_id              = aws_subnet.public[0].id
  vpc_security_group_ids = [aws_security_group.api.id]
  key_name               = var.key_pair_name
  iam_instance_profile   = aws_iam_instance_profile.ec2.name

  root_block_device {
    volume_type           = "gp3"
    volume_size           = 60 # 30 GB base + ~30 GB for Python venv, torch, sentence-transformers
    encrypted             = true
    delete_on_termination = true
  }

  user_data = base64encode(templatefile("${path.module}/user_data/api.sh", {
    weaviate_host     = aws_instance.weaviate.private_ip
    weaviate_port     = 8080
    project_name      = var.project_name
    anthropic_api_key = var.anthropic_api_key
    openai_api_key    = var.openai_api_key
    embed_backend     = var.embed_backend
    cache_backend     = var.cache_backend
    api_rate_limit    = var.api_rate_limit
    api_workers       = var.api_workers
    # Authenticated clone URL (token stripped from git remote after clone)
    github_clone_url = "https://x-access-token:${var.github_token}@${replace(var.github_repo, "https://", "")}"
    github_repo      = var.github_repo
    github_ref       = var.github_ref
  }))
  user_data_replace_on_change = true

  # Weaviate must be running before the API starts attempting to connect
  depends_on = [aws_instance.weaviate]

  tags = { Name = "${var.project_name}-${var.environment}-api" }
}
