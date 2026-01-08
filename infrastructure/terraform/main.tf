# AEC Agent - AWS Infrastructure
# Terraform configuration for G4dn instances with NVIDIA T4 GPUs

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment to use S3 backend for state
  # backend "s3" {
  #   bucket = "your-terraform-state-bucket"
  #   key    = "aec-agent/terraform.tfstate"
  #   region = "us-east-1"
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "AEC-Agent"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# =============================================================================
# Data Sources
# =============================================================================

data "aws_availability_zones" "available" {
  state = "available"
}

# Windows Server 2022 AMI
data "aws_ami" "windows_server" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Windows_Server-2022-English-Full-Base-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# =============================================================================
# VPC Configuration
# =============================================================================

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = {
    Name = "${var.project_name}-vpc"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.project_name}-igw"
  }
}

resource "aws_subnet" "public" {
  count                   = var.az_count
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.project_name}-public-${count.index + 1}"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.project_name}-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  count          = var.az_count
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# =============================================================================
# Security Groups
# =============================================================================

resource "aws_security_group" "cad_instances" {
  name        = "${var.project_name}-cad-sg"
  description = "Security group for CAD instances"
  vpc_id      = aws_vpc.main.id

  # RDP Access
  ingress {
    description = "RDP from allowed IPs"
    from_port   = 3389
    to_port     = 3389
    protocol    = "tcp"
    cidr_blocks = var.allowed_rdp_cidrs
  }

  # Chainlit UI
  ingress {
    description = "Chainlit UI"
    from_port   = 8000
    to_port     = 8100
    protocol    = "tcp"
    cidr_blocks = var.allowed_rdp_cidrs
  }

  # All outbound
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-cad-sg"
  }
}

# Sidecar ports - localhost only (internal to instance)
# These are not exposed externally as sidecars bind to 127.0.0.1

# =============================================================================
# IAM Role for Instances
# =============================================================================

resource "aws_iam_role" "cad_instance" {
  name = "${var.project_name}-cad-instance-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ssm" {
  role       = aws_iam_role.cad_instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "cloudwatch" {
  name = "${var.project_name}-cloudwatch-policy"
  role = aws_iam_role.cad_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "cad_instance" {
  name = "${var.project_name}-cad-instance-profile"
  role = aws_iam_role.cad_instance.name
}

# =============================================================================
# EC2 Instances (G4dn with NVIDIA T4)
# =============================================================================

resource "aws_instance" "cad_server" {
  count = var.instance_count

  ami                    = data.aws_ami.windows_server.id
  instance_type          = var.instance_type
  key_name               = var.key_pair_name
  subnet_id              = aws_subnet.public[count.index % var.az_count].id
  vpc_security_group_ids = [aws_security_group.cad_instances.id]
  iam_instance_profile   = aws_iam_instance_profile.cad_instance.name

  # Root volume (OS + Applications)
  root_block_device {
    volume_size           = var.root_volume_size
    volume_type           = "gp3"
    iops                  = 3000
    throughput            = 125
    encrypted             = true
    delete_on_termination = true
  }

  # Instance store for temp files (NVMe)
  # Note: g4dn.4xlarge has 225GB NVMe instance store automatically attached

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"  # IMDSv2
    http_put_response_hop_limit = 1
  }

  user_data = base64encode(templatefile("${path.module}/scripts/user_data.ps1", {
    environment      = var.environment
    cloudwatch_group = var.cloudwatch_log_group
  }))

  tags = {
    Name        = "${var.project_name}-cad-${count.index + 1}"
    UserCount   = var.users_per_instance
    InstanceNum = count.index + 1
  }

  lifecycle {
    ignore_changes = [ami]  # Don't recreate on AMI updates
  }
}

# =============================================================================
# CloudWatch Log Group
# =============================================================================

resource "aws_cloudwatch_log_group" "aec_agent" {
  name              = var.cloudwatch_log_group
  retention_in_days = var.log_retention_days

  tags = {
    Name = "${var.project_name}-logs"
  }
}

# =============================================================================
# CloudWatch Alarms
# =============================================================================

resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  count = var.instance_count

  alarm_name          = "${var.project_name}-high-cpu-${count.index + 1}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_description   = "CPU utilization above 80% for 15 minutes"

  dimensions = {
    InstanceId = aws_instance.cad_server[count.index].id
  }

  alarm_actions = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []
}

resource "aws_cloudwatch_metric_alarm" "high_memory" {
  count = var.instance_count

  alarm_name          = "${var.project_name}-high-memory-${count.index + 1}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "MemoryUtilization"
  namespace           = "CWAgent"
  period              = 300
  statistic           = "Average"
  threshold           = 85
  alarm_description   = "Memory utilization above 85% for 15 minutes"

  dimensions = {
    InstanceId = aws_instance.cad_server[count.index].id
  }

  alarm_actions = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []
}

# =============================================================================
# Outputs
# =============================================================================

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "instance_ids" {
  description = "EC2 instance IDs"
  value       = aws_instance.cad_server[*].id
}

output "instance_public_ips" {
  description = "Public IP addresses of CAD instances"
  value       = aws_instance.cad_server[*].public_ip
}

output "instance_private_ips" {
  description = "Private IP addresses of CAD instances"
  value       = aws_instance.cad_server[*].private_ip
}

output "security_group_id" {
  description = "Security group ID for CAD instances"
  value       = aws_security_group.cad_instances.id
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.aec_agent.name
}

output "total_user_capacity" {
  description = "Total user capacity across all instances"
  value       = var.instance_count * var.users_per_instance
}
