# ---------------------------------------------------------------------------
# Security groups
#
# Traffic path:  Internet → ALB (80/8501) → API SG (8000/8501) → Weaviate SG (8080/50051)
# Weaviate service ports are reachable only from the API security group.
# SSH is allowed separately from allowed_ssh_cidr.
# ---------------------------------------------------------------------------

resource "aws_security_group" "alb" {
  name        = "${var.project_name}-${var.environment}-alb-sg"
  description = "ALB: accept HTTP from anywhere; forward to API"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS (terminate here once certificate is attached)"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Streamlit UI"
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-${var.environment}-alb-sg" }
}

resource "aws_security_group" "api" {
  name_prefix = "${var.project_name}-${var.environment}-api-sg-"
  description = "FastAPI: port 8000 from ALB; Streamlit: port 8501 from ALB; SSH from allowed CIDR"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "FastAPI from ALB"
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description     = "Streamlit UI from ALB"
    from_port       = 8501
    to_port         = 8501
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-${var.environment}-api-sg" }

  # create_before_destroy prevents the DependencyViolation that occurs when
  # Weaviate's ingress rules reference this SG by ID. Terraform creates the
  # new SG first, updates Weaviate's rules, then deletes the old SG.
  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_security_group" "weaviate" {
  name        = "${var.project_name}-${var.environment}-weaviate-sg"
  description = "Weaviate: ports 8080/50051 from API SG only; SSH from allowed CIDR"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Weaviate HTTP from API"
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
  }

  ingress {
    description     = "Weaviate gRPC from API"
    from_port       = 50051
    to_port         = 50051
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
  }

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${var.project_name}-${var.environment}-weaviate-sg" }
}
