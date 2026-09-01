resource "random_password" "opsdesk_agent_api_key" {
  length  = 48
  special = false
}

resource "aws_secretsmanager_secret" "opsdesk_agent_api_key" {
  name        = "${var.project_name}/${var.environment}/opsdesk-agent-api-key"
  description = "Scoped credential for authenticated OpsDesk retrieval requests"
}

resource "aws_secretsmanager_secret_version" "opsdesk_agent_api_key" {
  secret_id     = aws_secretsmanager_secret.opsdesk_agent_api_key.id
  secret_string = random_password.opsdesk_agent_api_key.result
}
