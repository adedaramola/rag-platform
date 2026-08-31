output "api_endpoint" {
  description = "URL for the FastAPI query endpoint"
  value       = "http://${aws_lb.main.dns_name}/query"
}

output "api_docs_url" {
  description = "FastAPI interactive docs (Swagger UI)"
  value       = "http://${aws_lb.main.dns_name}/docs"
}

output "ui_url" {
  description = "Streamlit chat UI"
  value       = "http://${aws_lb.main.dns_name}:8501"
}

output "alb_dns_name" {
  description = "Raw DNS name of the Application Load Balancer"
  value       = aws_lb.main.dns_name
}

output "api_instance_public_ip" {
  description = "Public IP of the API instance (for SSH / debugging)"
  value       = aws_instance.api.public_ip
}

output "weaviate_instance_private_ip" {
  description = "Private VPC IP used by the API to reach the Weaviate instance"
  value       = aws_instance.weaviate.private_ip
}

output "documents_bucket_name" {
  description = "S3 bucket name for raw document uploads"
  value       = aws_s3_bucket.documents.id
}

output "ssh_weaviate" {
  description = "SSH command for the Weaviate instance"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${aws_instance.weaviate.public_ip}"
}

output "ssh_api" {
  description = "SSH command for the API instance"
  value       = "ssh -i ~/.ssh/${var.key_pair_name}.pem ec2-user@${aws_instance.api.public_ip}"
}
