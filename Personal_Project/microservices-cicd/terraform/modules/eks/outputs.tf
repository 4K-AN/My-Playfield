output "endpoint" { value = aws_eks_cluster.microservices.endpoint }
output "kubeconfig_cert" { value = aws_eks_cluster.microservices.certificate_authority[0].data }
output "cluster_name" { value = aws_eks_cluster.microservices.name }
