variable "cluster_name" { type = string }
variable "aws_region" { type = string; default = "ap-southeast-1" }

data "aws_eks_cluster" "cluster" { name = var.cluster_name }
data "aws_eks_cluster_auth" "cluster" { name = var.cluster_name }

provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_cert_data   = data.aws_eks_cluster.cluster.certificate_authority[0].data
  token                  = data.aws_eks_cluster_auth.cluster.token
}

resource "kubernetes_namespace" "microservices" {
  metadata {
    name = "microservices"
  }
}

resource "kubernetes_secret" "app_secrets" {
  metadata {
    name      = "app-secrets"
    namespace = kubernetes_namespace.microservices.metadata[0].name
  }

  data = {
    jwt-secret      = var.jwt_secret
    db-password     = var.db_password
    mongodb-uri     = "mongodb://${var.db_username}:${var.db_password}@mongodb:27017/microservices"
  }

  type = "Opaque"
}

resource "kubernetes_service_account" "ecr_robot" {
  metadata {
    name      = "ecr-robot"
    namespace = kubernetes_namespace.microservices.metadata[0].name
    annotations = {
      "eks.amazonaws.com/role-arn" = aws_iam_role.ecr_robot.arn
    }
  }
}

resource "aws_iam_role" "ecr_robot" {
  name = "microservices-ecr-pull"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Federated = data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer }
      Action   = "sts:AssumeRoleWithWebIdentity"
      Condition = {
        StringEquals = { "${data.aws_eks_cluster.cluster.identity[0].oidc[0].issuer}:sub" = "system:serviceaccount:microservices:ecr-robot" }
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecr_readonly" {
  role       = aws_iam_role.ecr_robot.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}
