# AEC Agent - Terraform Variables

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "aec-agent"
}

variable "environment" {
  description = "Environment (development, staging, production)"
  type        = string
  default     = "development"

  validation {
    condition     = contains(["development", "staging", "production"], var.environment)
    error_message = "Environment must be development, staging, or production."
  }
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-east-1"
}

# =============================================================================
# VPC Configuration
# =============================================================================

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "az_count" {
  description = "Number of availability zones to use"
  type        = number
  default     = 2

  validation {
    condition     = var.az_count >= 1 && var.az_count <= 3
    error_message = "AZ count must be between 1 and 3."
  }
}

# =============================================================================
# Instance Configuration
# =============================================================================

variable "instance_type" {
  description = "EC2 instance type (g4dn family for GPU)"
  type        = string
  default     = "g4dn.4xlarge"

  validation {
    condition     = can(regex("^g4dn\\.", var.instance_type))
    error_message = "Instance type must be from the g4dn family for GPU support."
  }
}

variable "instance_count" {
  description = "Number of CAD server instances"
  type        = number
  default     = 13  # For 50 users at 4 users/instance

  validation {
    condition     = var.instance_count >= 1 && var.instance_count <= 50
    error_message = "Instance count must be between 1 and 50."
  }
}

variable "users_per_instance" {
  description = "Number of concurrent users per instance"
  type        = number
  default     = 4

  validation {
    condition     = var.users_per_instance >= 1 && var.users_per_instance <= 8
    error_message = "Users per instance must be between 1 and 8."
  }
}

variable "root_volume_size" {
  description = "Root volume size in GB"
  type        = number
  default     = 200

  validation {
    condition     = var.root_volume_size >= 100 && var.root_volume_size <= 500
    error_message = "Root volume size must be between 100 and 500 GB."
  }
}

variable "key_pair_name" {
  description = "EC2 key pair name for RDP access"
  type        = string
}

# =============================================================================
# Security Configuration
# =============================================================================

variable "allowed_rdp_cidrs" {
  description = "CIDR blocks allowed to RDP to instances"
  type        = list(string)
  default     = []  # Must be specified

  validation {
    condition     = length(var.allowed_rdp_cidrs) > 0
    error_message = "At least one CIDR block must be specified for RDP access."
  }
}

# =============================================================================
# Logging Configuration
# =============================================================================

variable "cloudwatch_log_group" {
  description = "CloudWatch log group name"
  type        = string
  default     = "/aec-agent/application"
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30

  validation {
    condition = contains([
      1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653
    ], var.log_retention_days)
    error_message = "Log retention must be a valid CloudWatch retention period."
  }
}

# =============================================================================
# Alerting Configuration
# =============================================================================

variable "alarm_sns_topic_arn" {
  description = "SNS topic ARN for CloudWatch alarms (optional)"
  type        = string
  default     = ""
}
