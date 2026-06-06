provider "aws" {
  region = var.aws_region
}


/* -------------------------------------------------
   VPC
---------------------------------------------------*/
data "aws_availability_zones" "available" {}

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "microservices-vpc" }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = { Name = "microservices-igw" }
}

resource "aws_subnet" "public" {
  count             = 3
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  map_public_ip_on_launch = true

  tags = { Name = "public-subnet-${count.index + 1}" }
}

resource "aws_eip" "nat" {
  domain = "vpc"

  tags = { Name = "nat-eip" }
  depends_on = [aws_internet_gateway.main]
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id

  tags = { Name = "microservices-nat" }
}

resource "aws_subnet" "private" {
  count             = 3
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 100}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = { Name = "private-subnet-${count.index + 1}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "public-rt" }
}

resource "aws_route_table_association" "public" {
  count          = 3
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }

  tags = { Name = "private-rt" }
}

resource "aws_route_table_association" "private" {
  count          = 3
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}


/* -------------------------------------------------
   EKS
---------------------------------------------------*/
resource "aws_eks_cluster" "microservices" {
  name     = "microservices-cluster"
  version  = "1.29"
  role_arn = aws_iam_role.eks_cluster.arn

  vpc_config {
    subnet_ids = concat(aws_subnet.public[*].id, aws_subnet.private[*].id)
    endpoint_public_access  = true
    endpoint_private_access = true
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy,
    aws_cloudwatch_log_group.eks
  ]

  tags = { Name = "microservices-cluster" }
}

resource "aws_eks_node_group" "general" {
  cluster_name    = aws_eks_cluster.microservices.name
  node_group_name = "general-workers"
  node_role_arn   = aws_iam_role.eks_nodes.arn
  subnet_ids      = aws_subnet.private[*].id

  instance_types = ["t3.medium"]

  scaling_config {
    desired_size = 2
    max_size     = 4
    min_size     = 1
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_policy,
    aws_iam_role_policy_attachment.eks_worker_cni,
    aws_iam_role_policy_attachment.eks_worker_registry
  ]

  tags = { Name = "general-node-group" }
}


/* -------------------------------------------------
   RDS (PostgreSQL)
---------------------------------------------------*/
resource "aws_db_subnet_group" "main" {
  name       = "microservices-rds-subnet-group"
  subnet_ids = aws_subnet.private[*].id

  tags = { Name = "microservices-rds-subnet-group" }
}

resource "aws_db_instance" "main" {
  identifier             = "microservices-db"
  engine                 = "postgres"
  engine_version         = "15.4"
  instance_class         = "db.t3.medium"
  allocated_storage      = 20
  storage_type           = "gp3"
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  db_name               = "microservices"
  username              = "postgres"
  password              = var.db_password
  skip_final_snapshot   = false
  final_snapshot_identifier = "microservices-db-final-snapshot"

  backup_retention_period = 7
  publicly_accessible     = false

  tags = { Name = "microservices-db" }
}


/* -------------------------------------------------
   ECR Repositories
---------------------------------------------------*/
resource "aws_ecr_repository" "auth" {
  name = "microservices/auth-service"
  image_scanning_configuration { scan_on_push = true }
  force_delete = true
}

resource "aws_ecr_repository" "orders" {
  name = "microservices/order-service"
  image_scanning_configuration { scan_on_push = true }
  force_delete = true
}

resource "aws_ecr_repository" "gateway" {
  name = "microservices/api-gateway"
  image_scanning_configuration { scan_on_push = true }
  force_delete = true
}


/* -------------------------------------------------
   Security Groups
---------------------------------------------------*/
resource "aws_security_group" "eks_cluster" {
  name_prefix = "eks-cluster-sg-"
  vpc_id      = aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "eks-cluster-sg" }
}

resource "aws_security_group" "rds" {
  name_prefix = "rds-sg-"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "rds-sg" }
}


/* -------------------------------------------------
   IAM Roles (EKS + Service Accounts)
---------------------------------------------------*/
resource "aws_iam_role" "eks_cluster" {
  name = "eks-cluster-role-microservices"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "eks.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

resource "aws_cloudwatch_log_group" "eks" {
  name              = "/aws/eks/microservices-cluster/cluster"
  retention_in_days = 7
}

resource "aws_iam_role" "eks_nodes" {
  name = "eks-node-role-microservices"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{ Effect = "Allow", Principal = { Service = "ec2.amazonaws.com" }, Action = "sts:AssumeRole" }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_worker_policy" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}
resource "aws_iam_role_policy_attachment" "eks_worker_cni" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}
resource "aws_iam_role_policy_attachment" "eks_worker_registry" {
  role       = aws_iam_role.eks_nodes.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_openid_connect_provider" "eks" {
  url             = aws_eks_cluster.microservices.identity[0].oidc[0].issuer
  client_id_list  = ["sts.amazonaws.com"]
}


/* -------------------------------------------------
   Outputs
---------------------------------------------------*/
output "eks_cluster_name"     { value = aws_eks_cluster.microservices.name }
output "eks_cluster_endpoint" { value = aws_eks_cluster.microservices.endpoint }
output "rds_endpoint"         { value = aws_db_instance.main.endpoint }
output "ecr_auth_url"         { value = aws_ecr_repository.auth.repository_url }
output "ecr_orders_url"       { value = aws_ecr_repository.orders.repository_url }
output "ecr_gateway_url"      { value = aws_ecr_repository.gateway.repository_url }
