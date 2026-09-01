data "aws_route53_zone" "public" {
  name         = var.route53_zone_name
  private_zone = false
}

resource "aws_acm_certificate" "rag" {
  domain_name       = var.rag_domain_name
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_route53_record" "rag_certificate_validation" {
  for_each = {
    for option in aws_acm_certificate.rag.domain_validation_options : option.domain_name => {
      name   = option.resource_record_name
      record = option.resource_record_value
      type   = option.resource_record_type
    }
  }

  zone_id = data.aws_route53_zone.public.zone_id
  name    = each.value.name
  type    = each.value.type
  ttl     = 60
  records = [each.value.record]
}

resource "aws_acm_certificate_validation" "rag" {
  certificate_arn         = aws_acm_certificate.rag.arn
  validation_record_fqdns = [for record in aws_route53_record.rag_certificate_validation : record.fqdn]
}

resource "aws_route53_record" "rag" {
  zone_id = data.aws_route53_zone.public.zone_id
  name    = var.rag_domain_name
  type    = "A"

  alias {
    name                   = aws_lb.main.dns_name
    zone_id                = aws_lb.main.zone_id
    evaluate_target_health = true
  }
}
