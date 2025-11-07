import java.util.Random;

class Node {
    int data;
    Node nodeKiri;
    Node nodeKanan;

    public Node(int dt) {
        data = dt;
        nodeKiri = nodeKanan = null;
    }

    public void sisipDt(int dtSisip) {
        if (dtSisip < data) {
            if (nodeKiri == null)
                nodeKiri = new Node(dtSisip);
            else
                nodeKiri.sisipDt(dtSisip);
        } else if (dtSisip > data) {
            if (nodeKanan == null)
                nodeKanan = new Node(dtSisip);
            else
                nodeKanan.sisipDt(dtSisip);
        }
    }
}

public class TreeLatihan { 
    private Node root;

    public TreeLatihan() {
        root = null;
    }

    public void sisipDtNode(int dtSisip) {
        if (root == null)
            root = new Node(dtSisip);
        else
            root.sisipDt(dtSisip);
    }

 
    public void preorderTraversal() {
        preorder(root);
    }

    private void preorder(Node node) {
        if (node == null)
            return;
        System.out.printf("%d ", node.data);
        preorder(node.nodeKiri);
        preorder(node.nodeKanan);
    }

    public void inorderTraversal() {
        inorder(root);
    }

    private void inorder(Node node) {
        if (node == null)
            return;
        inorder(node.nodeKiri);
        System.out.printf("%d ", node.data);
        inorder(node.nodeKanan);
    }

    public void postorderTraversal() {
        postorder(root);
    }

    private void postorder(Node node) {
        if (node == null)
            return;
        postorder(node.nodeKiri);
        postorder(node.nodeKanan);
        System.out.printf("%d ", node.data);
    }

 
    public int countNodes() {
        return countNodesRec(root);
    }

    private int countNodesRec(Node node) {
        if (node == null) {
            return 0;
        }
   
        return 1 + countNodesRec(node.nodeKiri) + countNodesRec(node.nodeKanan);
    }

   
    public int countLeaves() {
        return countLeavesRec(root);
    }

    private int countLeavesRec(Node node) {
        if (node == null) {
            return 0;
        }
      
        if (node.nodeKiri == null && node.nodeKanan == null) {
            return 1;
        }
    
        return countLeavesRec(node.nodeKiri) + countLeavesRec(node.nodeKanan);
    }


    public int getHeight() {

        return getHeightRec(root);
    }

    private int getHeightRec(Node node) {
        if (node == null) {
            return -1; 
        }

        int leftHeight = getHeightRec(node.nodeKiri);
        int rightHeight = getHeightRec(node.nodeKanan);

        return 1 + Math.max(leftHeight, rightHeight);
    }


    public int getLevelCount() {

        return getHeight() + 1;
    }


    public static void main(String args[]) {
        TreeLatihan Tree = new TreeLatihan(); 
        int nilai;
        Random randomNumber = new Random();
        System.out.println("sisip nilai data berikut : ");

        for (int i = 1; i <= 10; i++) {
            nilai = randomNumber.nextInt(100);
            System.out.print(nilai + " ");
            Tree.sisipDtNode(nilai);
        }

        System.out.println("\n\nPreorder traversal");
        Tree.preorderTraversal();
        System.out.println("\n\nInorder traversal");
        Tree.inorderTraversal();
        System.out.println("\n\nPostorder traversal");
        Tree.postorderTraversal();
        System.out.println();

        System.out.println("\n--- Analisis Latihan ---");

        System.out.println("Banyaknya Node : " + Tree.countNodes());

        System.out.println("Banyaknya Daun : " + Tree.countLeaves());

        System.out.println("Tinggi Pohon   : " + Tree.getHeight());

        System.out.println("Jumlah Level   : " + Tree.getLevelCount());

        System.out.println();
    }
}
